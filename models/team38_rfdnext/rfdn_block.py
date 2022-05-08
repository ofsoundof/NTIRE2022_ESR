from collections import OrderedDict
import torch.nn as nn

import torch
import torch.nn.functional as F


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


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


def activation(act_type, inplace=True):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace=inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(0.05, inplace=inplace)
    elif act_type == 'gelu':
        layer = nn.GELU()
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

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

class CX(nn.Module):
    def __init__(self, C, act_type="gelu"):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(C, C, 7, 1, 3, groups=C),
            nn.Conv2d(C, 4 * C, 1),
            activation(act_type),
            nn.Conv2d(4 * C, C, 1)
        )

    def forward(self, x):
        return self.conv(x) + x

class RFDB(nn.Module):
    def __init__(self, in_channels, act_type, distill_factor=2):
        super().__init__()
        
        DC = in_channels // distill_factor
        
        self.c1_d = conv_layer(in_channels, DC, 1)

        self.c1_r = conv_layer(in_channels, DC, 3)
        self.c2_d = conv_layer(DC, DC, 1)

        self.c2_r = conv_layer(DC, DC, 3)
        self.c3_d = conv_layer(DC, DC, 1)

        self.c3_r = conv_layer(DC, DC, 3)

        self.c4 = conv_layer(DC, DC, 3)
        
        self.act = activation(act_type)
        
        self.c5 = conv_layer(DC * 4, in_channels, 1)
        self.esa = CX(in_channels, act_type)

    def forward(self, input):

        dc_1 = self.c1_d(input)
        
        rc_1 = self.c1_r(input) + dc_1
        dc_2 = self.c2_d(rc_1)
        
        rc_2 = self.c2_r(rc_1) + rc_1
        dc_3 = self.c3_d(rc_2)

        rc_3 = self.act(self.c3_r(rc_2) + rc_2)
        rc_4 = self.c4(rc_3)

        out = self.act(torch.cat([dc_1, dc_2, dc_3, rc_4], dim=1))
        out = self.c5(out)

        return self.esa(out)


class MRB(nn.Module):
    def __init__(self, in_channels, distill_factor=2, act_type="gelu", **kwargs):
        super().__init__()
        DC = in_channels // distill_factor
        self.init_reduce = conv_layer(in_channels, DC, 1)
        self.c1 = conv_layer(DC, DC, 3)
        self.c2 = conv_layer(DC, DC, 3)
        self.c3 = conv_layer(DC, DC, 3)
        self.end_reduce = conv_layer(4 * DC, in_channels, 1)
        
        self.esa = ESA(in_channels, nn.Conv2d)
        self.act = activation(act_type)

    def forward(self, input):
        r0 = self.act(self.init_reduce(input))
        c1 = self.act(self.c1(r0) + r0)
        c2 = self.act(self.c2(c1) + c1 + r0)
        c3 = self.act(self.c3(c2) + c2 + c1 + r0)

        out = self.end_reduce(torch.cat([r0, c1, c2, c3], dim=1))
        out = self.esa(out)

        return out + input

def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)