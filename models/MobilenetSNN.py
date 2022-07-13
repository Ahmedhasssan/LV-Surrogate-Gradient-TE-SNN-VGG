"""
Full-sized spiking mobilenet
"""
import math
import torch
import torch.nn as nn
from models.layers import *

class MBNETSNN(nn.Module):
    def __init__(self):
        super(MBNETSNN, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.features = nn.Sequential(
            SConv(3,32,3,1,1),
            pool,
            SConvDW(32,64,3,1,1),
            SConvDW(64,64,3,1,1),
            pool,
            SConvDW(64,128,3,1,1),
            SConvDW(128,128,3,1,1),
            pool,
            SConvDW(128,256,3,1,1),
            SConvDW(256,512,3,1,1),
            pool,
        )

        W = int(48/2/2/2/2)
        self.classifier1 = SeqToANNContainer(nn.Linear(512*W*W*4,512))
        self.classifier2 = SeqToANNContainer(nn.Linear(512,10))
        self.drop = SeqToANNContainer(Dropout(0.5))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier1(x)
        x = self.drop(x)
        x = self.classifier2(x)
        return x