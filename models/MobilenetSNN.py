"""
Full-sized spiking mobilenet
"""
import torch
import torch.nn as nn
from models.layers import *
from .methods import QLinear

class MBNETSNN(nn.Module):
    def __init__(self, num_classes=10, membit=2, neg=-1.0, wbit=4, thres=1.0, tau=0.5):
        super(MBNETSNN, self).__init__()
        self.features = nn.Sequential(
            SConv(3, 32, 3, 1, 1, pool=True, membit=membit, neg=neg, wbit=wbit, thres=thres, tau=tau),
            SConvDW(32, 64, 3, 1, 1, pool=True, membit=membit, neg=neg, wbit=wbit, thres=thres, tau=tau),
            SConvDW(64, 64, 3, 1, 1, pool=True, membit=membit, neg=neg, wbit=wbit, thres=thres, tau=tau),
            SConvDW(64, 128, 3, 1, 1, pool=True, membit=membit, neg=neg, wbit=wbit, thres=thres, tau=tau),
            SConvDW(128, 128, 3, 1, 1, pool=True, membit=membit, neg=neg, wbit=wbit, thres=thres, tau=tau),
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

class MBNETSNNWIDE(nn.Module):
    def __init__(self, num_classes=10):
        super(MBNETSNNWIDE, self).__init__()
        self.features = nn.Sequential(
            SConv(3, 32, 3, 1, 1, pool=True),
            SConvDW(32, 64, 3, 1, 1, pool=True),
            SConvDW(64, 64, 3, 1, 1, pool=True),
            SConvDW(64, 128, 3, 1, 1, pool=True),
            SConvDW(128, 128, 3, 1, 1, pool=True),
            SConvDW(128, 256, 3, 1, 1, pool=True),
        )

        W = int(48/2/2/2/2)
        self.classifier1 = SeqToANNContainer(nn.Linear(1024, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier1(x)
        return x

class MBNETSNNWIDE_PostPool(nn.Module):
    def __init__(self, num_classes=10):
        super(MBNETSNNWIDE_PostPool, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.features = nn.Sequential(
            SConv(3, 32, 3, 1, 1, pool=False),
            pool,
            SConvDW(32, 64, 3, 1, 1, pool=False),
            pool,
            SConvDW(64, 64, 3, 1, 1, pool=False),
            pool,
            SConvDW(64, 128, 3, 1, 1, pool=False),
            pool,
            SConvDW(128, 128, 3, 1, 1, pool=False),
            pool,
            SConvDW(128, 256, 3, 1, 1, pool=False),
            pool,
        )

        W = int(48/2/2/2/2)
        self.classifier1 = SeqToANNContainer(nn.Linear(1024, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier1(x)
        return x

class MBNETSNN_NegQ(nn.Module):
    def __init__(self, num_classes=10):
        super(MBNETSNN_NegQ, self).__init__()
        self.features = nn.Sequential(
            SConv(3, 32, 3, 1, 1, pool=True, neg=-1.0),
            SConvDW(32, 64, 3, 1, 1, pool=True, neg=-1.0),
            SConvDW(64, 64, 3, 1, 1, pool=True, neg=-1.0),
            SConvDW(64, 128, 3, 1, 1, pool=True, neg=-1.0),
            SConvDW(128, 128, 3, 1, 1, pool=True, neg=-1.0),
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

class MBNETSNNWIDE_PostPool_NegQ(nn.Module):
    def __init__(self, num_classes=10):
        super(MBNETSNNWIDE_PostPool_NegQ, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.features = nn.Sequential(
            SConv(3, 32, 3, 1, 1, pool=False, neg=-5.0),
            pool,
            SConvDW(32, 64, 3, 1, 1, pool=False, neg=-5.0),
            pool,
            SConvDW(64, 64, 3, 1, 1, pool=False, neg=-2.0),
            pool,
            SConvDW(64, 128, 3, 1, 1, pool=False, neg=-2.0),
            pool,
            SConvDW(128, 128, 3, 1, 1, pool=False, neg=-2.0),
            pool,
            SConvDW(128, 256, 3, 1, 1, pool=False, neg=-2.0),
            pool,
        )

        W = int(48/2/2/2/2)
        self.classifier1 = SeqToANNContainer(nn.Linear(1024, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier1(x)
        return x

class MBNETSNN_NegQ_LP(nn.Module):
    def __init__(self, num_classes=10):
        super(MBNETSNN_NegQ_LP, self).__init__()
        self.features = nn.Sequential(
            SConv(3, 32, 3, 1, 1, pool=True, neg=-1.0),
            SConvDW(32, 64, 3, 1, 1, pool=True, neg=-1.0),
            SConvDW(64, 64, 3, 1, 1, pool=True, neg=-1.0),
            SConvDW(64, 128, 3, 1, 1, pool=True, neg=-1.0),
            SConvDW(128, 128, 3, 1, 1, pool=True, neg=-1.0),
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
