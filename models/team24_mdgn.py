import torch
import torch.nn as nn
import torch.nn.functional as F

class MDSA(nn.Module):
    def __init__(self, in_channels):
        super(MDSA, self).__init__()
        self.f1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU(num_parameters=in_channels))
        self.f2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU(num_parameters=in_channels))
        self.f3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU(num_parameters=in_channels))

        self.conv_fuse = nn.Sequential(
                            nn.Conv2d(3 * in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
                            nn.PReLU(num_parameters=in_channels)
                        )
                    
        self.sa = nn.Sequential(
                    nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.Sigmoid()
                )

    def forward(self, x):
        f1 = self.f1(x)
        f2 = self.f2(f1)
        f3 = self.f3(f2)
        f = self.conv_fuse( torch.cat([f1, f2, f3], dim=1) )
        s = self.sa(x)
        return f*s


class MDGN(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=4, out_nc=3, upscale=4):
        super(MDGN, self).__init__()
        self.fea_conv = nn.Conv2d(in_nc, nf, kernel_size=3, stride=1, padding=1, bias=True) 

        B = []
        for i in range(num_modules):
            B += [MDSA(in_channels=nf)]
        self.B = nn.Sequential(*B)

        self.LR_conv = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True) 
        self.upsampler = nn.Sequential(
                            nn.Conv2d(nf, out_nc * (upscale ** 2), kernel_size=3, stride=1, padding=1, bias=True),
                            nn.PixelShuffle(upscale)
                        )

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B = self.B(out_fea)
        out_lr = self.LR_conv(out_B) + out_fea
        return self.upsampler(out_lr)


