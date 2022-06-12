import random
from models.layers import *

class VGGSNN(nn.Module):
    def __init__(self):
        super(VGGSNN, self).__init__()
        #pool = SeqToANNContainer(nn.MaxPool2d(2))
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.features = nn.Sequential(
            Layer(3,32,3,1,1),
            #Layer(32,32,3,1,1),
            #pool,
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
        W = int(48/2/2/2/2)
        # self.T = 4
        self.classifier1 = SeqToANNContainer(nn.Linear(512*W*W,512))
        self.classifier2 = SeqToANNContainer(nn.Linear(512,10))
        self.drop = SeqToANNContainer(nn.Dropout(0.5))
        #self.drop = SeqToANNContainer(Dropout2d(0.25))
        #self.classifier=Linear(256*W*W,10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier1(x)
        x = self.drop(x)
        x = self.classifier2(x)
        return x

    
