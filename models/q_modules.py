"""
DNN quantization modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RoundQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        output_int = input.mul(scale).round()
        output_float = output_int.div_(scale)
        return output_float

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class WQ(nn.Module):
    def __init__(self, wbit, num_features, infer=False):
        super(WQ, self).__init__()
        self.wbit = wbit
        self.num_features = num_features
        self.register_buffer('alpha_w', torch.tensor(1.))
        self.infer = infer

    def forward(self, input):
        z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}
        z = z_typical[f'{int(self.wbit)}bit']
        n_lv = 2 ** (self.wbit - 1) - 1

        m = input.abs().mean()
        std = input.std()
        
        self.alpha_w = 1/z[0] * std - z[1]/z[0] * m 
        input = input.clamp(-self.alpha_w.item(), self.alpha_w.item())
        self.scale = n_lv / self.alpha_w

        if not self.infer:
            wq = RoundQuant.apply(input, self.scale)
        else:
            wq = input.mul(self.scale).round()
        return wq
    
    def extra_repr(self):
        return super(WQ, self).extra_repr() + 'wbit={}, infer={}'.format(self.wbit, self.infer)


class AQ(nn.Module):
    def __init__(self, abit, num_features, alpha_init):
        super(AQ, self).__init__()
        self.abit = abit
        self.alpha = nn.Parameter(torch.Tensor([alpha_init]))

    def forward(self, input):
        if input.size(1) > 3:
            input = torch.where(input < self.alpha, input, self.alpha)

            n_lv = 2**self.abit - 1
            scale = n_lv / self.alpha

            a_float = RoundQuant.apply(input, scale)
        else:
            a_float = input
        return a_float
    
    def extra_repr(self):
        return super(AQ, self).extra_repr() + 'abit={}'.format(self.abit)


class QConv2d(nn.Conv2d):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1, 
        bias=True, 
        wbit=32, 
        abit=32,
        infer=False
    ):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
        # precisions
        self.abit = abit
        self.wbit = wbit
        num_features = self.weight.data.size(0)
        self.infer = False

        self.WQ = WQ(wbit=wbit, num_features=num_features, infer=self.infer)
        self.AQ = AQ(abit=abit, num_features=num_features, alpha_init=10.0)
        

    def forward(self, input):
        if self.abit < 32:
            input_q = self.AQ(input)
        else:
            input_q = input
        
        if self.wbit < 32:
            weight_q = self.WQ(self.weight)
        else:
            weight_q = self.weight
        
        if self.infer:
            self.wint.data = weight_q
            out = F.conv2d(input_q, self.wint, self.bias, self.stride, self.padding, self.dilation, self.groups)
            self.oint = out
            out = out.div(self.WQ.scale)
        else:
            out = F.conv2d(input_q, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
        return out

class QLinear(nn.Linear):
    r"""
    Fully connected layer with Quantized weight
    """
    def __init__(self, in_features, out_features, bias=True, wbit=32, abit=32, alpha_init=10.0):
        super(QLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        
        # precisions
        self.wbit = wbit
        self.abit = abit
        self.alpha_init = alpha_init
        channels = self.weight.data.size(0)

        # quantizers
        self.WQ = WQ(wbit=wbit, num_features=channels, channel_wise=0)
        self.AQ = AQ(abit=abit, num_features=channels, alpha_init=alpha_init)

    def forward(self, input):
        if self.abit < 32:
            input_q = self.AQ(input)
        else:
            input_q = input
        
        if self.wbit < 32:
            weight_q = self.WQ(self.weight)
        else:
            weight_q = self.weight

        out = F.linear(input_q, weight_q, self.bias)
        return out
