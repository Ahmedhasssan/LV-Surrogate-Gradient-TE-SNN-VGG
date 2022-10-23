"""
DNN quantization modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

################### POT ##########################
##################################################

def build_power_value(B=2, additive=True):
    base_a = [0.]
    base_b = [0.]
    base_c = [0.]
    if additive:
        if B == 2:
            for i in range(3):
                base_a.append(2 ** (-i - 1))
        elif B == 4:
            for i in range(2):
                base_a.append(2 ** (-2 * i - 1))
                base_b.append(2 ** (-2 * i - 2))
        elif B == 6:
            for i in range(3):
                base_a.append(2 ** (-3 * i - 1))
                base_b.append(2 ** (-3 * i - 2))
                base_c.append(2 ** (-3 * i - 3))
        elif B == 3:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-i - 1))
                else:
                    base_b.append(2 ** (-i - 1))
                    base_a.append(2 ** (-i - 2))
        elif B == 5:
            for i in range(3):
                if i < 2:
                    base_a.append(2 ** (-2 * i - 1))
                    base_b.append(2 ** (-2 * i - 2))
                else:
                    base_c.append(2 ** (-2 * i - 1))
                    base_a.append(2 ** (-2 * i - 2))
                    base_b.append(2 ** (-2 * i - 3))
        else:
            pass
    else:
        for i in range(2 ** B - 3):  ### there is 3 instead of 2 for original implementations of 4 bit
            base_a.append(2 ** (-i - 1))
    values = []
    for a in base_a:
        for b in base_b:
            for c in base_c:
                values.append((a + b + c))
    values = torch.Tensor(list(set(values)))
    values = values.mul(1.0 / torch.max(values))
    return values


def weight_quantization(b, grids, power=True):
    def uniform_quant(x, b):
        xdiv = x.mul((2 ** b - 1))
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    def power_quant(x, value_s):
        shape = x.shape
        xhard = x.view(-1)
        value_s = value_s.type_as(x)
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]  # project to nearest quantization level
        xhard = value_s[idxs].view(shape)
        # xout = (xhard - x).detach() + x
        return xhard

    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            alpha=alpha.round_()
            if alpha%2==0:
                 alpha=alpha
            else:
                 alpha=alpha+1
            input=input.div_(alpha)                          # weights are first divided by alpha
            input_c = input.clamp(min=-1, max=1)       # then clipped to [-1,1]
            sign = input_c.sign()
            input_abs = input_c.abs()
            if power:
                grids = [-1.5,-1.25,-1,-0.75,-0.625,-0.5,-0.125,0,0.125,0.25,0.5,0.625,0.75]
                input_q = power_quant(input_abs, grids).mul(sign)  # project to Q^a(alpha, B)
            else:
                input_q = uniform_quant(input_abs, b).mul(sign)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)             # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()             # grad for weights will not be clipped
            input, input_q = ctx.saved_tensors
            i = (input.abs()>1.).float()
            sign = input.sign()
            grad_alpha = (grad_output*(sign*i + (input_q-input)*(1-i))).sum()
            return grad_input, grad_alpha

    return _pq().apply


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit, power=True):
        super(weight_quantize_fn, self).__init__()
        assert (w_bit <=5 and w_bit > 0) or w_bit == 32
        self.w_bit = w_bit-1
        #self.w_bit = w_bit
        #self.power = power if w_bit>2 else False
        self.power = power
        self.grids = build_power_value(self.w_bit, additive=False)
        self.weight_q = weight_quantization(b=self.w_bit, grids=self.grids, power=self.power)
        self.register_parameter('wgt_alpha', nn.Parameter(torch.tensor(4.0)))

    def forward(self, weight):
        if self.w_bit == 32:
            weight_q = weight
        else:
            mean = weight.data.mean()
            std = weight.data.std()
            weight = weight.add(-mean).div(std)      # weights normalization
            weight_q = self.weight_q(weight, self.wgt_alpha)
            weight_q = weight_q.clamp(min=-1.75, max=1.75) ## Use this for the 4 bit
        return weight_q


def act_quantization(b, grid, power=True):

    def uniform_quant(x, b=3):
        xdiv = x.mul(2 ** b - 1)
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    def power_quant(x, grid):
        shape = x.shape
        xhard = x.view(-1)
        value_s = grid.type_as(x)
        idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
        xhard = value_s[idxs].view(shape)
        return xhard

    # class _uq(torch.autograd.Function):
    #     @staticmethod
    #     def forward(ctx, input, alpha=2):
    #         # alpha=alpha.round_()
    #         # if alpha%2==0:
    #         #     alpha=alpha
    #         # else:
    #         #     alpha=alpha+1
    #         #input=input.div(alpha) ### There was alpha instead of 2
    #         print("=======")
    #         input_c = input.clamp(min=-2,max=1)
    #         if power:
    #             grid = torch.Tensor([-2.0,-1.8750,-1.750,-1.6250,-1.50,-1.3750,-1.250,-1.1250,-1.00, -0.8750,
    #             -0.750,-0.6250,-0.50,-0.3750,-0.250,-0.125,0,0.125,0.3750,0.50,0.6250,0.750,0.8750])
    #             input_q = power_quant(input_c, grid)
    #         else:
    #             input_q = uniform_quant(input_c, b)
    #         ctx.save_for_backward(input, input_q)
    #         #input_q = input_q.mul(alpha)
    #         #input_q = input_q.clamp(min=-2 , max=0.875)
    #         # print(input_c.unique())
    #         # print(input_q.unique())
    #         # import pdb;pdb.set_trace()
    #         return input_q
    #     @staticmethod
    #     def backward(ctx, grad_output):
    #         grad_input = grad_output.clone()
    #         input, input_q = ctx.saved_tensors
    #         i = (input > 1.).float()
    #         # grad_alpha = (grad_output * (i + (input_q - input) * (1 - i))).sum()
    #         grad_input = grad_input*(1-i)
    #         return grad_input, None ## there was grad_alpha instead if None previously

    # return _uq().apply

    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha=2):
            input_c = input.clamp(min=-2,max=1)
            if power:
                grid = torch.Tensor([-2.0,-1.8750,-1.750,-1.6250,-1.50,-1.3750,-1.250,-1.1250,-1.00, -0.8750,
                -0.750,-0.6250,-0.50,-0.3750,-0.250,-0.125,0,0.125,0.3750,0.50,0.6250,0.750,0.8750])
                input_q = power_quant(input_c, grid)
            else:
                input_q = uniform_quant(input_c, b)
            ctx.save_for_backward(input, input_q)
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            input, input_q = ctx.saved_tensors
            i = (input > 1.).float()
            grad_input = grad_input*(1-i)
            return grad_input, None

    return _uq().apply

#########################################################
#########################################################
act_grid = build_power_value(4, additive=False)
log2 = act_quantization(4, act_grid, power=True)

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
