import random
from models.layers import *

class CIFARVGGSNN9(nn.Module):
    def __init__(self, T):
        super(CIFARVGGSNN9, self).__init__()
        self.T = T
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.features = nn.Sequential(
            Layer(3,32,3,1,1),
            
            Layer(32,64,3,1,1),
            Layer(64,64,3,1,1),
            pool,
            Layer(64,128,3,1,1),
            Layer(128,128,3,1,1),
            pool,
            Layer(128,256,3,1,1),
            Layer(256,256,3,1,1),
            pool,
            Layer(256,512,3,1,1),
            Layer(512,512,3,1,1),
            pool,
        )
        W = int(32/2/2/2/2)
        self.classifier1 = SeqToANNContainer(nn.Linear(512*W*W,512))
        self.classifier2 = SeqToANNContainer(nn.Linear(512,10))
        self.drop = SeqToANNContainer(nn.Dropout(0.5))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = input.unsqueeze(1)
        x = x.repeat(1, self.T, 1, 1, 1)
        x = self.features(x)
        x = torch.flatten(x, 2)
        x = self.classifier1(x)
        x = self.drop(x)
        x = self.classifier2(x)
        return x
    
class CIFARVGGSNN9_4bit(nn.Module):
    def __init__(self, T):
        super(CIFARVGGSNN9_4bit, self).__init__()
        self.T = T
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.features = nn.Sequential(
            Layer_LP(3,32,3,1,1,4),
            
            Layer_LP(32,64,3,1,1,4),
            Layer_LP(64,64,3,1,1,4),
            pool,
            Layer_LP(64,128,3,1,1,4),
            Layer_LP(128,128,3,1,1,4),
            pool,
            Layer_LP(128,256,3,1,1,4),
            Layer_LP(256,256,3,1,1,4),
            pool,
            Layer_LP(256,512,3,1,1,4),
            Layer_LP(512,512,3,1,1,4),
            pool,
        )
        W = int(48/2/2/2/2)
        self.classifier1 = SeqToANNContainer(nn.Linear(512*W*W,512))
        self.classifier2 = SeqToANNContainer(nn.Linear(512,10))
        self.drop = SeqToANNContainer(nn.Dropout(0.5))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = input.unsqueeze(1)
        x = x.repeat(1, self.T, 1, 1, 1)
        x = self.features(x)
        x = torch.flatten(x, 2)
        x = self.classifier1(x)
        x = self.drop(x)
        x = self.classifier2(x)
        return x

