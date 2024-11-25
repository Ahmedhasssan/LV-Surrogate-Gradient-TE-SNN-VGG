import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from models.q_modules import log2
from .methods import QConv2d

def power_quant(x, value_s):
    shape = x.shape
    xhard = x.view(-1)
    value_s = value_s.type_as(x)
    idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]  # project to nearest quantization level
    xhard = value_s[idxs].view(shape)
    return xhard

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SeqToANNContainer(nn.Module):
    # Chunk of code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        out = y_seq.view(y_shape)
        return out

class Layer_LP(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding, wbit):
        super(Layer_LP, self).__init__()
        self.fwd = SeqToANNContainer(
            QConv2d(in_plane, out_plane, kernel_size, stride=stride, padding=padding, bias=True, wbit=wbit, abit=32, infer=False),
            nn.BatchNorm2d(out_plane)
        )
        #self.act = LIFSpike()
        self.act=ZIFArchTan()
    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.mask = None
        self.p = p

    def extra_repr(self):
        return 'p={}'.format(
            self.p
        )

    def create_mask(self, x: torch.Tensor):
        self.mask = F.dropout(torch.ones_like(x.data), self.p, training=True)

    def forward(self, x: torch.Tensor):
        if self.training:
            if self.mask is None:
                self.create_mask(x)

            return x * self.mask
        else:
            return x

    def reset(self):
        self.mask = None


class Layer(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
            nn.BatchNorm2d(out_plane)
        )
        if in_plane > 3:
            self.act = ZIFArchTan()
        else:
            self.act = LIFSpike()
        
    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x

class SConv(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding, pool=False, 
                membit=2, neg=-2.0, wbit=4, thres=1.0, tau=0.5):
        super(SConv, self).__init__()
        if wbit < 32:
            self.fwd = SeqToANNContainer(
                QConv2d(in_plane, out_plane, kernel_size, stride, padding, wbit=wbit, abit=32),
                nn.BatchNorm2d(out_plane)
            )
        else:
            self.fwd = SeqToANNContainer(
                nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
                nn.BatchNorm2d(out_plane)
            )
        self.act = LIFSpike(thresh=thres, tau=tau, gama=1.0, membit=membit, neg=neg)

        if pool:
            self.pool = SeqToANNContainer(nn.AvgPool2d(2))
        else:
            self.pool = nn.Identity()

    def forward(self, x):
        x = self.fwd(x)
        x = self.pool(x)
        x = self.act(x)
        return x

class SConvDW(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding, pool=False, 
                membit=2, neg=-1.0, wbit=4, thres=1.0, tau=0.5):
        super(SConvDW, self).__init__()
        if wbit < 32:
            self.dw = SeqToANNContainer(
                QConv2d(in_plane,in_plane,kernel_size,stride,padding, wbit=wbit, abit=32),
                nn.BatchNorm2d(in_plane)
            )
            self.pw = SeqToANNContainer(
                QConv2d(in_plane, out_plane, 1, stride, padding, wbit=wbit, abit=32),
                nn.BatchNorm2d(out_plane)
            )
        else:
            self.dw = SeqToANNContainer(
                nn.Conv2d(in_plane,in_plane,kernel_size,stride,padding),
                nn.BatchNorm2d(in_plane)
            )
            self.pw = SeqToANNContainer(
                nn.Conv2d(in_plane,out_plane,1,stride,padding),
                nn.BatchNorm2d(out_plane)
            )
        self.act1 = LIFSpike(thresh=thres, tau=tau, membit=membit, neg=neg)
        self.act2 = LIFSpike(thresh=thres, tau=tau, membit=membit, neg=neg)
        
        # self.act=ZIFArchTan()
        
        if pool:
            self.pool = SeqToANNContainer(nn.AvgPool2d(2))
        else:
            self.pool = nn.Identity()

    def forward(self,x):
        x = self.dw(x)
        x = self.act1(x)
        x = self.pw(x)
        x = self.pool(x)
        x = self.act2(x)
        return x

######### Surrogate Gradient Sigmoid ############
def heaviside(x:Tensor):
    return x.ge(0.).float()

class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gama, salpha, thresh):
            out= heaviside(x)
            ctx.alpha = thresh
            L = torch.tensor([gama])
            ctx.save_for_backward(x, out, L, thresh)
            return out
    
    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others, thresh) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp

        sig = torch.sigmoid(input)
        fire = input.ge(0).float()
        grad_thre = grad_input.mul(sig)
        grad_thre = torch.sum(grad_thre.mul(fire)).view(-1).mul(-1)
        return grad_input, None, None, grad_thre

class ZIFArchTan(nn.Module):
    r"""
    Arch Tan function
    """
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
        super(ZIFArchTan, self).__init__()
        self.act = sigmoid.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        self.thresh = nn.Parameter(torch.Tensor([thresh]), requires_grad=True)
        self.salpha = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)

    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[1]
        for t in range(T):
            mem = mem * self.tau + x[:, t, ...]
            #vth = sigm(self.thresh)
            vth = self.thresh
            spike = self.act(mem - vth, self.gama, self.salpha, self.thresh)
            mem = (1 - spike) * mem
           # spike = spike.mul(self.salpha.item())
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)

############################################

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama, salpha, thresh):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None, None, None

def power_quant(x, grid):
        shape = x.shape
        xhard = x.view(-1)
        value_s = grid.type_as(x)
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
        xhard = value_s[idxs].view(shape)
        return xhard

class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0, membit=2, neg=-2.0):
        super(LIFSpike, self).__init__()
        #self.act_alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.act_alpha = 2
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        
        # spike ratio of negative potential
        self.ratio = AverageMeter()
        self.neg = neg
        self.min = AverageMeter()
        self.conv_spars = AverageMeter()
        
        # # # mem quant
        # qrange = self.thresh - self.neg # thresh = 1.0, neg = -2.0
        # self.levels = torch.tensor([self.neg + 0.125*i for i in range(int(qrange//0.125))])
        # self.scale = (2**membit-1) / qrange 


    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[1]
        for t in range(T):
            mem = mem * self.tau + x[:, t, ...]
            spike = self.act(mem - self.thresh, self.gama, 1.0, 1.0)
            mem = (1 - spike) * mem
            # mem = log2(mem, self.act_alpha)
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)
    
    def extra_repr(self) -> str:
        return super().extra_repr()+"thre={:.3f}, tau={:.3f}".format(self.thresh, self.tau)


def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    return x


# ----- For ResNet19 code -----


class tdLayer(nn.Module):
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = SeqToANNContainer(layer)
        self.bn = bn

    def forward(self, x):
        try:
            x_ = self.layer(x)
        except:
            print(self.layer)
            import pdb;pdb.set_trace()
        if self.bn is not None:
            x_ = self.bn(x_)
        return x_


class tdBatchNorm(nn.Module):
    def __init__(self, out_panel):
        super(tdBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(out_panel)
        self.seqbn = SeqToANNContainer(self.bn)

    def forward(self, x):
        y = self.seqbn(x)
        return y
