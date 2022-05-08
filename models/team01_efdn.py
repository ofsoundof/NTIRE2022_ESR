
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class EDBB_deploy(nn.Module):
    def __init__(self, inp_planes, out_planes):
        super(EDBB_deploy, self).__init__()

        self.rep_conv = nn.Conv2d(in_channels=inp_planes, out_channels=out_planes, kernel_size=3, stride=1,
                                  padding=1, bias=True)

        self.act = nn.PReLU(num_parameters=out_planes)

    def forward(self, x):
        y = self.rep_conv(x)
        y = self.act(y)

        return y


class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)

        return x * m

          
class conv(nn.Module):
    def __init__(self, n_feats):
        super(conv, self).__init__()
        self.conv1x1 = nn.Conv2d(n_feats, n_feats, 1, 1, 0) 
        self.act = nn.PReLU(num_parameters=n_feats)
    def forward(self, x):
        return self.act(self.conv1x1(x))
    
class Cell(nn.Module):
    def __init__(self, n_feats=48, dynamic = True, deploy = False, L= None, with_13=False):
        super(Cell, self).__init__()
        
        self.conv1 = conv(n_feats)#nn.Conv2d(n_feats, n_feats, 1, 1, 0) 
        self.conv2 = EDBB_deploy(n_feats,n_feats)
        self.conv3 = EDBB_deploy(n_feats,n_feats)

        self.fuse = nn.Conv2d(n_feats*2, n_feats, 1, 1, 0)

        self.att = ESA(n_feats, nn.Conv2d) #MAB(n_feats)# ENLCA(n_feats)  #CoordAtt(n_feats,n_feats,10)# 

        self.branch = nn.ModuleList([nn.Conv2d(n_feats, n_feats//2, 1, 1, 0) for _ in range(4)])

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        # fuse [x, out1, out2, out3]
        out = self.fuse(torch.cat([self.branch[0](x), self.branch[1](out1), self.branch[2](out2), self.branch[3](out3)], dim=1))
        out = self.att(out)
        out += x

        return out


class EFDN(nn.Module):
    def __init__(self, scale=4, in_channels=3, n_feats=48, out_channels=3):
        super(EFDN, self).__init__()
        self.head = nn.Conv2d(in_channels, n_feats, 3, 1, 1)
        # body cells
        self.cells = nn.ModuleList([Cell(n_feats) for _ in range(4)])
        
        # fusion
        self.local_fuse = nn.ModuleList([nn.Conv2d(n_feats*2, n_feats, 1, 1, 0) for _ in range(3)])

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, out_channels*(scale**2), 3, 1, 1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        # head
        out0 = self.head(x)

        # body cells
        out1 = self.cells[0](out0)
        out2 = self.cells[1](out1)
        out2_fuse = self.local_fuse[0](torch.cat([out1, out2], dim=1))
        out3 = self.cells[2](out2_fuse)
        out3_fuse = self.local_fuse[1](torch.cat([out2, out3], dim=1))
        out4 = self.cells[3](out3_fuse)
        out4_fuse = self.local_fuse[2](torch.cat([out2, out4], dim=1))

        out = out4_fuse + out0      

        # tail
        out = self.tail(out)

        return out.clamp(0,1)
   


