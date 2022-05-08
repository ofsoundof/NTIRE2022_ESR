import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=padding, dilation=dilation, groups=groups)

    def forward(self, x):
        return self.conv(x)
    
class ESA(nn.Module):
    def __init__(self, channels, shrink=4, esa_channels=None):
        super().__init__()
        f = channels // shrink
        if esa_channels:
            f = esa_channels
        self.conv1    = nn.Conv2d(channels, f, kernel_size=1, padding=0)
        self.conv_f   = nn.Conv2d(f, f, kernel_size=1, padding=0)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2    = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3    = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3_   = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4    = nn.Conv2d(f, channels, kernel_size=1, padding=0)
        self.sigmoid  = nn.Sigmoid()
        self.relu     = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_     = self.conv1(x)
        c1      = self.conv2(c1_)
        v_max   = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3      = self.relu(self.conv3(v_range))
        c3      = self.conv3_(c3)
        c3      = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf      = self.conv_f(c1_)
        c4      = self.conv4(c3+cf)
        m       = self.sigmoid(c4)
        return x * m
    
class BuildingBlock(nn.Module):
    def __init__(self, channels, esa_channels=None, n_convs=3):
        super().__init__()

        self.n_convs = n_convs
        last_channels = channels * (n_convs+1)
        
        self.esa = nn.ModuleList([ESA(channels, esa_channels=esa_channels) for _ in range(n_convs)])
        self.esa_last = ESA(channels, esa_channels=esa_channels)
                
        self.convs = nn.ModuleList([conv_layer(channels, channels, 3) for _ in range(n_convs)])
        self.conv_last = conv_layer(last_channels, channels, 1)
        self.act = nn.LeakyReLU(0.05, inplace=True)

    def forward(self, x):
        cat_list = [x]

        for i in range(self.n_convs):
            x_conv = self.convs[i](x) + x
            x_act  = self.act(x_conv)
            x      = self.esa[i](x_act)
            cat_list.append(x)

        out = torch.cat(cat_list, dim=1)
        out = self.conv_last(out)
        out = self.esa_last(out)
        return out

def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(conv, pixel_shuffle)
    
class SR_model(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, upscale=4,
                 channels=32, esa_channels=16, n_modules=4, n_convs=3):
        super().__init__()
        
        self.fea_conv  = conv_layer(in_channels, channels, 3)
        self.mods      = nn.ModuleList([BuildingBlock(channels, esa_channels, n_convs) for _ in range(n_modules)])
        self.c         = conv_layer(channels * n_modules, channels, 1)
        self.LR_conv   = conv_layer(channels, channels, 3)
        self.act       = nn.LeakyReLU(0.05, inplace=True)
        self.upsampler = pixelshuffle_block(channels, out_channels, upscale_factor=4)

    def forward(self, x):
        out_fea = self.fea_conv(x)
        
        out_module = []
        out_m = out_fea
        for i in range(len(self.mods)):
            out_m = self.mods[i](out_m)
            out_module.append(out_m)

        out_B  = self.act(self.c(torch.cat(out_module, dim=1)))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)
        return output