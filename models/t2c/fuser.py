import torch
import copy
import torch.nn as nn

from ..methods import QBaseConv2d, ConvBN, QBaseLinear
from ..layers import SeqToANNContainer

class LayerFuser(object):
    def __init__(self, model:nn.Module):
        self.model = model
        # flag
        self.flag = False
        
        # layers
        self.groups = []
        
        # parameters
        self.xscales = []
        self.xbound = []

        # full precision conv layer
        self.fpl = 0

        # full precision classifier
        self.fpc = False
    
    def inference(self, model:nn.Module):
        """
        Switch to inference mode
        """
        for n, m in model.named_modules():
            if hasattr(m, "inference"):
                m.inference()

    def conv_bn(self, conv:QBaseConv2d, bn:nn.BatchNorm2d):
        sq = 1 / conv.wq.scale.data
        # bn
        std = torch.sqrt(bn.running_var.data + bn.eps)

        # merged bn scaler
        sbn = bn.weight.data.mul(sq) / std
        # bn bias
        bbn = bn.bias - bn.weight.mul(bn.running_mean.data).div(std) + bn.weight.mul(conv.bias.data).div(std)
        return sbn, bbn
    
    def fuse(self):
        # initialize the model copy to avoid the mutated dict
        fused_model = copy.deepcopy(self.model) 
        
        for name, module in self.model.named_children():
            for n, m in module.named_children():
                for k, layer in m.named_children():
                    for ln, ll in layer.named_children():
                        if isinstance(ll, SeqToANNContainer):
                            if isinstance(ll.module, nn.Sequential):
                                conv = ll.module[0]
                                bn = ll.module[1]

                                # fuse scalers and biases
                                s, b = self.conv_bn(conv, bn)

                                # get fused module
                                fm = ConvBN(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, 
                                                dilation=conv.dilation, groups=conv.groups, wbit=conv.wbit, abit=conv.abit, train_flag=conv.train_flag)
                                
                                # update scalers 
                                fm.scaler.scale.data = s
                                fm.scaler.bias.data = b
                                if conv.bias is not None:
                                    conv.bias.data.fill_(0.0)

                                # replace the bn
                                setattr(fm, "conv", conv)
                                setattr(fm, "bn", nn.Identity())

                                # replace the module
                                setattr(ll, "module", fm)
                    if isinstance(layer, QBaseLinear):
                        print(layer)

                            
                        setattr(layer, ln, ll)
                    setattr(m, k, layer)
                setattr(module, n, m)
            setattr(fused_model, name, module)
        
        return fused_model


                                
                                


                                