import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

###### Spectral Transformation Module ##########
class SpectralTx(nn.Module):
    
    def __init__(self, inchannels, outchannel):
        super(SpectralTx, self).__init__()
        self.inch = inchannels
        self.outch = outchannel
        
        self.conv = nn.Conv2d(in_channels=inchannels*2, out_channels=outchannel*2,kernel_size=1,padding=0)
        self.relu = torch.nn.LeakyReLU(0.2)
        
    def forward(self,x):
        batch,c,h,w = x.size()
        
        ff = torch.view_as_real(torch.fft.rfft(x, dim=2,norm='ortho'))
        ff = ff.permute(0,1,4,2,3).contiguous()
        ff = ff.view((batch,-1,)+ff.size()[3:])
        
        ff = self.conv(ff)
        ff = self.relu(ff)
        
        ff = torch.view_as_complex(ff.view((batch,-1,2)+ff.size()[2:]).permute(0,1,3,4,2).contiguous())
        
        out = torch.fft.irfft(ff,dim=2,norm="ortho")
        return out




###### Non-local Attention module #######

class GlobalContextNet(nn.Module):
    
    def __init__(self,inchannels):
        super(GlobalContextNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inchannels, out_channels=1,kernel_size=1,padding=0)
        self.conv2 = nn.Conv2d(in_channels=inchannels, out_channels=inchannels,kernel_size=1,padding=0)
        self.conv3 = nn.Conv2d(in_channels=inchannels, out_channels=inchannels,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(in_channels=inchannels, out_channels=inchannels,kernel_size=3,padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.lrelu = torch.nn.LeakyReLU(0.2)
        
    def forward(self,x):
        batch,c,h,w = x.size()
        b1 = self.conv1(x)           ##b,1,h,w
        b1 = nn.ReLU(inplace=True)(b1)
        b1 = b1.view((batch,1,h*w)) ##b,1,h*w
        b1  = nn.Softmax(dim=2)(b1)
        b1  = b1.view((batch,1,h,w))
        b1 = b1.repeat(repeats=(1,c,1,1)) ##b,c,h,w
        
        b2 = torch.mul(x,b1)
        
        b2 = self.conv2(b2)
        b2 = self.lrelu(b2)
        
        b2 = self.conv3(b2)
        b2 = self.lrelu(b2)
        b2 = self.conv4(b2)
        b2 = b2.view((batch,c,h*w))
        b2 = nn.Softmax(dim=2)(b2)
        b2 = b2.view((batch,c,h,w))
        
        return torch.mul(b2,x)




###### Fast Fourier Convolution module #########

class FFCU(nn.Module):
    def __init__(self,inchannels):
        super(FFCU,self).__init__()
        self.glob_ch = int(0.5*inchannels)
        self.loc_ch = inchannels - self.glob_ch
        
        self.GlobContext = GlobalContextNet(self.loc_ch)
        self.conv0 = nn.Conv2d(in_channels=self.loc_ch, out_channels=self.glob_ch,kernel_size=1,padding=0)
        
        self.conv1 = nn.Conv2d(in_channels=self.loc_ch, out_channels=self.loc_ch,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.glob_ch, out_channels=self.loc_ch,kernel_size=3,padding=1)
        self.SpT = SpectralTx(self.glob_ch,self.glob_ch)
        self.relu = torch.nn.LeakyReLU(0.2)
        
    def forward(self,x):
        xl,xg = torch.split(x,[self.loc_ch,self.glob_ch],dim=1)
        c00 = self.relu(self.conv1(xl))
        c01 = self.conv0(self.GlobContext(xl))
        
        c10 = self.relu(self.conv2(xg))
        c11 = self.SpT(xg)

        
        Cl = c00 + c10
        Cg = c01 + c11
        
        return torch.cat([Cl,Cg],dim=1)

