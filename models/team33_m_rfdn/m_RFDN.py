import torch.nn as nn
import models.team33_m_rfdn.basicblock as B
import torch
import torch.nn.functional as F


class m_RFDN(nn.Module):
    def __init__(self, in_nc=3, nf=52, num_modules=4, out_nc=3, upscale=4):
        super(m_RFDN, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = B.m_RFDB(in_channels=nf)
        self.B2 = B.m_RFDB(in_channels=nf)
        self.B3 = B.m_RFDB(in_channels=nf)
        self.B4 = B.m_RFDB(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.att1 = B.PA(nf)
        self.HRconv1 = nn.Conv2d(nf, 24, 3, 1, 1, bias=True)
        
        self.upconv2 = nn.Conv2d(24, 24, 3, 1, 1, bias=True)
        self.att2 = B.PA(24)
        self.HRconv2 = nn.Conv2d(24, 24, 3, 1, 1, bias=True)
            
        self.conv_last = nn.Conv2d(24, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.scale_idx = 0


    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        out_lr = self.upconv1(F.interpolate(out_lr, scale_factor=2, mode='nearest'))
        out_lr = self.lrelu(self.att1(out_lr))
        out_lr = self.lrelu(self.HRconv1(out_lr))
        out_lr = self.upconv2(F.interpolate(out_lr, scale_factor=2, mode='nearest'))
        out_lr = self.lrelu(self.att2(out_lr))
        out_lr = self.lrelu(self.HRconv2(out_lr))
        out = self.conv_last(out_lr)
        return out

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
