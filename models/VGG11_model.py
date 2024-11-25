import random
from models.layers import *

class VGGSNN11(nn.Module):
    def __init__(self, num_classes):
        super(VGGSNN11, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.features = nn.Sequential(
            Layer(3,64,3,1,1),
            Layer(64,128,3,1,1),
            pool,
            Layer(128,256,3,1,1),
            Layer(256,256,3,1,1),
            pool,
            Layer(256,512,3,1,1),
            Layer(512,512,3,1,1),
            
            Layer(512,512,3,1,1),
            Layer(512,512,3,1,1),
            pool,
            Layer(512,1024,3,1,1),
            pool,
        )
        W = int(48/2/2/2/2)
        self.classifier1 = SeqToANNContainer(nn.Linear(1024*W*W,512))
        self.drop = SeqToANNContainer(Dropout(0.5))
        self.classifier2 = SeqToANNContainer(nn.Linear(512,num_classes))

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

class VGGSNN11_4bit(nn.Module):
    def __init__(self):
        super(VGGSNN11_4bit, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.features = nn.Sequential(
            Layer_LP(3,64,3,1,1,4),
            Layer_LP(64,128,3,1,1,4),
            pool,
            Layer_LP(128,256,3,1,1,4),
            Layer_LP(256,256,3,1,1,4),
            pool,
            Layer_LP(256,512,3,1,1,4),
            Layer_LP(512,512,3,1,1,4),

            Layer_LP(512,512,3,1,1,4),
            Layer_LP(512,512,3,1,1,4),
            pool,
            Layer_LP(512,1024,3,1,1,4),
            pool,
        )
        W = int(48/2/2/2/2)
        # self.T = 4
        self.classifier1 = SeqToANNContainer(nn.Linear(1024*W*W,512))
        self.drop = SeqToANNContainer(Dropout(0.5))
        self.classifier2 = SeqToANNContainer(nn.Linear(512,10))

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


if __name__ == '__main__':
    model = VGGSNN11()
    
