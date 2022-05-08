import torch.nn as nn

import math

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

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


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
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


def channel_shuffle(x, groups):
    batch, num_chan, height, width = x.data.size()
    channel_per_group = num_chan // groups
    x = x.view(batch, groups, channel_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, -1, height, width)
    return x


# Enhanced fast spatial attention
class EFSA(nn.Module):
    def __init__(self, n_feats, conv):
        super(EFSA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv_block(f, f, kernel_size=3, dilation=1, act_type='lrelu')  # convert to dilated conv
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv_block(f, f, kernel_size=3, dilation=2, act_type='lrelu')
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = hsigmoid()#nn.Sigmoid()
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.conv_max(v_max)
        c3 = self.conv3(v_max) + v_range
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = c1_
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


def make_model(args, parent=False):
    model = RFDN()
    return model


class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super(Scale, self).__init__()
        self.scale = Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale



# Attention-guided adaptive weighted residual unit
class AAWRU(nn.Module):
    def __init__(self, nf, kernel_size, stride, wn, act=nn.ReLU(True)):
        super(AAWRU, self).__init__()
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)
        self.stride = stride
        
        self.body = nn.Sequential(
            wn(conv_layer(nf, nf, kernel_size=3)),
            act,
            wn(conv_layer(nf, nf, kernel_size=3)),
            EFSA(nf, nn.Conv2d)
        )

    def forward(self, x):
        res1 = self.body(x)
        res = self.res_scale(res1) + self.x_scale(x)
        return res


# Local residual feature fusion block
class LRFFB(nn.Module):
    def __init__(self, nf, wn, act=nn.ReLU(inplace=True)):
        super(LRFFB, self).__init__()
        self.b0 = AAWRU(nf, 3, 1, wn=wn, act=act)
        self.b1 = AAWRU(nf, 3, 1, wn=wn, act=act)
        self.b2 = AAWRU(nf, 3, 1, wn=wn, act=act)
        self.b3 = AAWRU(nf, 3, 2, wn=wn, act=act)
        self.reduction = wn(nn.Conv2d(nf * 2, nf, 1, padding=0))
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.b1(x0) + x0
        x2 = self.b2(x1) + x1
        x3 = self.b3(x2)
        res1 = self.reduction(channel_shuffle(torch.cat([x3, x2], dim=1), 2))
        res2 = self.reduction(channel_shuffle(torch.cat([res1, x1], dim=1), 2))
        res = self.reduction(channel_shuffle(torch.cat([res2, x0], dim=1), 2))

        return self.res_scale(res) + self.x_scale(x)


class RFESR(nn.Module):
    def __init__(self, in_nc=3, nf=32, num_modules=4, out_nc=3, upscale=4):
        super(RFESR, self).__init__()

        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)
        wn = lambda x: nn.utils.weight_norm(x)
        act = nn.LeakyReLU(inplace=True)

        self.B1 = LRFFB(nf, wn=wn, act=act)
        self.B2 = LRFFB(nf, wn=wn, act=act)
        self.B3 = LRFFB(nf, wn=wn, act=act)
        self.B4 = LRFFB(nf, wn=wn, act=act)

        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=4)
        self.scale_idx = 0

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_lr = self.LR_conv(out_B4) + out_fea
        output = self.upsampler(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
