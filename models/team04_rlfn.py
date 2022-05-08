import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - stride) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)

def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

class ESA(nn.Module):
    def __init__(self,esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
#         self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
#         self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
#         v_range = self.relu(self.conv_max(v_max))
#         c3 = self.relu(self.conv3(v_range))
#         c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        
        return x * m

class RLFB(nn.Module):
    def __init__(self, in_channels, mid_channel, out_channels=None, esa_channels=16):
        super(RLFB, self).__init__()
        
        if(mid_channel==None):
            mid_channel =  in_channels
        if(out_channels==None):
            out_channels = in_channels
        
        self.c1_r = conv_layer(in_channels, mid_channel, 3)
        self.c2_r = conv_layer(mid_channel, mid_channel, 3)
        self.c3_r = conv_layer(mid_channel, in_channels, 3)
        
        self.c5 = conv_layer(in_channels, out_channels, 1)
        self.esa = ESA(esa_channels, out_channels, nn.Conv2d)
        
        self.act = activation('lrelu', neg_slope=0.05)
        
    def forward(self, input):
        out = (self.c1_r(input))
        out = self.act(out)

        out = (self.c2_r(out))
        out = self.act(out)

        out = (self.c3_r(out))
        out = self.act(out)

        out = out + input
        out = self.esa(self.c5(out)) 

        return out

class RLFN_cut(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=46, mf=48, upscale=4):
        super(RLFN_cut, self).__init__()

        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = RLFB(in_channels=nf, mid_channel = mf)
        self.B2 = RLFB(in_channels=nf, mid_channel = mf)
        self.B3 = RLFB(in_channels=nf, mid_channel = mf)
        self.B4 = RLFB(in_channels=nf, mid_channel = mf)

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0

    def forward(self, input):
        out_fea = self.fea_conv(input)
        
        out_B = self.B1(out_fea)
        out_B = self.B2(out_B)
        out_B = self.B3(out_B)
        out_B = self.B4(out_B)

        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
