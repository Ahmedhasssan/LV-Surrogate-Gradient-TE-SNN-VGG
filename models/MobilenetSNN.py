"""
Full-sized spiking mobilenet
"""
import math
import torch
import torch.nn as nn
from models.layers import *

class MBNETSNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MBNETSNN, self).__init__()
        self.features = nn.Sequential(
            SConv(3, 32, 3, 1, 1, pool=True),
            SConvDW(32, 64, 3, 1, 1, pool=True),
            SConvDW(64, 64, 3, 1, 1, pool=True),
            SConvDW(64, 128, 3, 1, 1, pool=True),
            SConvDW(128, 128, 3, 1, 1, pool=True),
        )

        W = int(48/2/2/2/2)
        self.classifier1 = SeqToANNContainer(nn.Linear(1152, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier1(x)
        return x