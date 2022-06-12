import torch
import torch.nn as nn
import torch.nn.functional as F
#from surrogate import *
from torch import Tensor
import math

class TensorNormalization(nn.Module):
    def __init__(self,mean, std):
        super(TensorNormalization, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std
    def forward(self,X):
        return normalizex(X,self.mean,self.std)

def normalizex(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    if mean.device != tensor.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    return tensor.sub(mean).div(std)


class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
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
        return y_seq.view(y_shape)

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

class Dropout2d(Dropout):
    def __init__(self, p=0.2):
        super().__init__(p)

    def create_mask(self, x: torch.Tensor):
        self.mask = F.dropout2d(torch.ones_like(x.data), self.p, training=True)
    def forward(self, x: torch.Tensor):
        if self.training:
            if self.mask is None:
                self.create_mask(x)

            return x * self.mask
        else:
            return x


class Layer(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
            nn.BatchNorm2d(out_plane)
        )
        #self.act = LIFSpike()
        self.act=ZIFArchTan()

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x

class Linear(nn.Module):
    def __init__(self,in_plane,out_plane):
        super(Linear, self).__init__()
        self.L1 = SeqToANNContainer(
            nn.Linear(in_plane,out_plane),
        )
        #self.act = LIFSpike()
        self.act=ZIFArchTan()

    def forward(self,x):
        x = self.L1(x)
        x = self.act(x)
        return x

class nn_Dropout2d(nn.Module):
    def __init__(self,p):
        super(nn_Dropout2d, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Dropout2d(p)
        )
    def forward(self,x):
        x = self.fwd(x)
        return x

class APLayer(nn.Module):
    def __init__(self,kernel_size):
        super(APLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.AvgPool2d(kernel_size),
        )
        #self.act = LIFSpike()
        self.act=ZIFArchTan()
    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x

######### Surrogate Gradient ATAN ############
def heaviside(x:Tensor):
    return x.ge(0.).float()

class atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gama, salpha, thresh):
        #if x.requires_grad:
            out= heaviside(x)
            ctx.alpha = thresh
            L = torch.tensor([gama])
            ctx.save_for_backward(x, out, L, thresh)
            #out=out.mul(salpha.half())
            return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        grad_alpha = None
        grad_input = grad_output.clone()
        if ctx.needs_input_grad[0]:
            grad_input = ctx.alpha / 2 / (1 + (math.pi / 2 * ctx.alpha * ctx.saved_tensors[0]).pow_(2)) * grad_output
        (input, out, others, thresh) = ctx.saved_tensors
        gama = others[0].item()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        # s = ctx.saved_tensors[1]
        # grad_alpha = torch.sum(grad_output.mul(s)).view(-1)

        mask_alpha = input.ge(0).float()
        mask = grad_input*mask_alpha
        grad_vth = torch.sum(grad_output.mul(mask)).view(-1).mul(-1)
        return grad_input, None, None, grad_vth

def sigm(x):
    sig = 1 / (1 + math.exp(-x))
    return sig

######### Surrogate Gradient Sigmoid ############
def heaviside(x:Tensor):
    return x.ge(0.).float()

class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gama, salpha, thresh):
            out= heaviside(x)
            #out=out.mul(salpha.half())
            ctx.alpha = thresh
            L = torch.tensor([gama])
            ctx.save_for_backward(x, out, L, thresh)
            return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        grad_alpha = None
        grad_input = grad_output.clone()
        if ctx.needs_input_grad[0]:
            sgax = (ctx.saved_tensors[0] * ctx.alpha).sigmoid_()
            grad_input = grad_output * (1. - sgax) * sgax * ctx.alpha
        (input, out, others, thresh) = ctx.saved_tensors
        gama = others[0].item()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp

        # s = ctx.saved_tensors[1]
        # grad_alpha = torch.sum(grad_output.mul(s)).view(-1)

        mask_alpha = input.ge(0).float()
        mask = grad_input*mask_alpha
        grad_vth = torch.sum(grad_output.mul(mask)).view(-1).mul(-1)
        return grad_input, None, None, grad_vth

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
        self.salpha = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

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
        out = (input > 0).float()
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

class LVZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama, thresh):
        #input = input-thresh
        out = input.gt(thresh).float().mul(3.3*thresh)
        #out = (input-thresh > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L, thresh)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others, thresh) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        mask_alpha = input.ge(thresh[0].item()).float()
        grad_alpha = torch.sum(grad_output.mul(mask_alpha)).view(-1)
        return grad_input, None, grad_alpha

class SLSZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama, salpha, thresh):
        #input = input-thresh
        out = input.gt(thresh).float()
        #out = (input-thresh > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        out=out.mul(salpha.half())
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        s = ctx.saved_tensors[1]
        grad_alpha = torch.sum(grad_output.mul(s)).view(-1)
        return grad_input, None, grad_alpha, None

class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        # self.k = 10
        # self.act = F.sigmoid
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        #self.thresh = nn.Parameter(torch.Tensor([thresh]), requires_grad=True)
        #self.salpha = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[1]
        for t in range(T):
            mem = mem * self.tau + x[:, t, ...]
            spike = self.act(mem - self.thresh, self.gama)
            #spike = self.act(mem, self.gama, self.salpha, self.thresh)
            # spike = self.act((mem - self.thresh)*self.k)
            #spike = spike.mul(self.salpha.item())
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)


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
        x_ = self.layer(x)
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


# LIFSpike = LIF
