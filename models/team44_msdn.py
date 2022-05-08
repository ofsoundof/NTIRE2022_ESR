import math

import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

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


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
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

def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'silu':
        layer = nn.SiLU(inplace)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class VisionAttention(nn.Module):  # Visual Attention Network
    def __init__(self, n_feats, k=21, d=3, shrink=0.25, scale=2):
        super().__init__()
        f = int(n_feats*shrink)
        self.head = nn.Conv2d(n_feats, f, 1)
        # self.proj_2 = nn.Conv2d(f, f, kernel_size=3, stride=2)
        self.activation = nn.GELU()
        self.LKA = nn.Sequential(
            conv_layer(f, f, k//d, dilation=d, groups=f),
            # self.activation,
            conv_layer(f, f, 2*d-1, groups=f),
            # self.activation,
            conv_layer(f, f, kernel_size=1),
            # self.activation,
        )
        self.tail = nn.Conv2d(f, n_feats, 1)
        self.scale = scale

    def forward(self, x):
        c1 = self.head(x)
        # c2 = self.proj_2(c1)
        c2 = F.max_pool2d(c1, kernel_size=self.scale * 2 + 1, stride=self.scale)
        c2 = self.activation(c2)
        c2 = self.LKA(c2)
        c3 = F.interpolate(c2, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        # c3 = self.activation(c3)
        a = self.tail(c3 + c1)
        a = torch.sigmoid(a)
        return x * a


class MSDB(nn.Module):  # Multi-scale Information Distillation Block
    def __init__(self, in_channels, distillation_rate=0.25, act_type='silu', attentionScale=2):
        super(MSDB, self).__init__()
        self.dc = self.distilled_channels = int(in_channels * distillation_rate)
        self.rc = self.remaining_channels = in_channels

        self.c1_d = conv_block(in_channels, self.dc, 1, act_type=act_type)
        self.c1_r = nn.Sequential(
            conv_block(in_channels, 2*self.rc, 1, act_type=act_type),
            conv_block(2*self.rc, self.rc, 3, groups=2, act_type=act_type)
        )

        self.c2_d = conv_block(self.remaining_channels, self.dc, 1, act_type=act_type)
        self.c2_r = conv_block(self.remaining_channels, self.rc, 3, act_type=act_type)

        self.c3 = conv_block(self.remaining_channels, self.dc, 3, dilation=2, act_type=act_type)

        self.c4 = conv_layer(self.dc * 3, in_channels, 1)
        self.attention = VisionAttention(in_channels, k=21, d=3, shrink=0.125, scale=attentionScale)

    def forward(self, input):
        distilled_c1 = self.c1_d(input)
        r_c1 = self.c1_r(input)

        distilled_c2 = self.c2_d(r_c1)
        r_c2 = self.c2_r(r_c1)

        r_c3 = self.c3(r_c2)

        out = torch.cat([distilled_c1, distilled_c2, r_c3], dim=1)
        out_fused = self.attention(self.c4(out))

        return out_fused


class MSDN(nn.Module):
    def __init__(self, in_nc=3, nf=56, dist_rate=0.5, num_modules=3, out_nc=3, upscale=4, act_type='silu'):
        super(MSDN, self).__init__()

        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        self.B = nn.ModuleList(
            [MSDB(in_channels=nf, distillation_rate=dist_rate, act_type=act_type, attentionScale=num_modules - i + 1)
             for i in range(num_modules)]
        )
        self.C = nn.Sequential(
            conv_block(nf * num_modules, nf, kernel_size=1, act_type=act_type),
            conv_layer(nf, nf, kernel_size=3)
        )

        self.upsampler = pixelshuffle_block(nf, out_nc, upscale_factor=upscale)

    def forward(self, input):
        fea = self.fea_conv(input*255)
        fea_out = []
        for i in range(len(self.B)):
            if i == 0:
                fea_out.append(self.B[i](fea))
            else:
                fea_out.append(self.B[i](fea_out[-1]))
        fea_out = torch.cat(fea_out, dim=1)
        fea_out = self.C(fea_out) + fea
        output = self.upsampler(fea_out)/255

        return output
