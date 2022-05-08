import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models.team02_nlffc.basicblock import *


class Netw(nn.Module):
    def __init__(self):
        super(Netw,self).__init__()
        self.c0 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=1,padding=0)
        
        self.b1 = FFCU(64)
        self.c1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1,padding=0)
        
        self.b2 = FFCU(64)
        self.c2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1,padding=0)
        
        self.b3 = FFCU(64)
        self.c3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1,padding=0)
        
        self.b4 = FFCU(64)
        self.c4 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1,padding=0)
        
        self.b5 = FFCU(64)
        self.c5 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1,padding=0)
        
        self.b6 = FFCU(64)
        self.c6 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1,padding=0)
        
        self.b7 = FFCU(64)
        self.c7 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1,padding=0)
        
        self.b8 = FFCU(64)
        self.c8 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1,padding=0)
        
        self.b9 = FFCU(64)
        self.c9 = nn.Conv2d(in_channels=64,out_channels=3,kernel_size=1,padding=0)
        self.relu = torch.nn.ReLU(inplace=True)
        
        
    def forward(self,x):
        x = torch.nn.Upsample(scale_factor=4,mode='bicubic')(x)
        x1 = self.c0(x)
        x1 = self.c1(self.b1(x1))
        x2 = self.c2(self.b2(x1))
        x3 = self.c3(self.b3(x2))
        x4 = self.c4(self.b4(x3))+x2
        x5 = self.c5(self.b5(x4))
        x6 = self.c6(self.b6(x5))
        x7 = self.c7(self.b7(x6))
        x8 = self.c8(self.b8(x7))+x4
        x9 = self.relu(self.c9(self.b9(x8)))
        
        return x9



