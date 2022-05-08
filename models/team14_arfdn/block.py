import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation, groups=groups)

def norm(norm_type, nc, mpdist=False):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        if mpdist:
            layer = nn.SyncBatchNorm(nc, affine=True)
        else:
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

def conv_block_old(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu', mpdist=False):

    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = spectral_norm(nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=bias, groups=groups))
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc, mpdist) if norm_type else None
    return sequential(p, c, n, a)

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

def activation(act_type, inplace=True, neg_slope=0.1, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr

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

def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

def upconv_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1, act_type='relu'):
    upsample = nn.Upsample(scale_factor=upscale_factor, mode='nearest')
    conv = conv_layer(in_channels, in_channels, kernel_size, stride)
    act = activation(act_type)
    conv_last = conv_layer(in_channels, out_channels, kernel_size, stride)
    return sequential(upsample, conv, act, conv_last)

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
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


class SubpixelConvolutionLayer(nn.Module):
    r"""

    Args:
        channels (int): Number of channels in the input image. (Default: 64)
    """

    def __init__(self, channels: int = 16) -> None:
        super(SubpixelConvolutionLayer, self).__init__()

        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.rfb1 = ReceptiveFieldBlock(channels, channels)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.rfb2 = ReceptiveFieldBlock(channels, channels)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.rfb1(out)
        out = self.leaky_relu1(out)
        out = self.upsample(out)
        out = self.rfb2(out)
        out = self.leaky_relu2(out)
        return out


class ARFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(ARFDB, self).__init__()
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c0_d = conv_layer(in_channels, self.dc, 1)

        self.c1_l1 = nn.Conv2d(in_channels, self.dc, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.c1_l2 = nn.Conv2d(self.dc, self.dc, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.c1_m1 = nn.Conv2d(in_channels, self.dc, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.c1_m2 = nn.Conv2d(self.dc, self.dc, kernel_size=(3, 1), stride=1, padding=(1, 0))
        #self.c1_r = conv_layer(in_channels, self.rc, 3, dilation=2)


        self.c1_d = conv_layer(self.dc, self.dc, 1)


        self.c2_l1 = nn.Conv2d(self.dc, self.dc, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.c2_l2 = nn.Conv2d(self.dc, self.dc, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.c2_m1 = nn.Conv2d(self.dc, self.dc, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.c2_m2 = nn.Conv2d(self.dc, self.dc, kernel_size=(3, 1), stride=1, padding=(1, 0))


        self.c2_d = conv_layer(self.dc, self.dc, 1)

        self.c3_l1 = nn.Conv2d(self.dc, self.dc, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.c3_l2 = nn.Conv2d(self.dc, self.dc, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.c3_m1 = nn.Conv2d(self.dc, self.dc, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.c3_m2 = nn.Conv2d(self.dc, self.dc, kernel_size=(3, 1), stride=1, padding=(1, 0))


        self.c4 = conv_layer(self.dc, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc * 4, self.rc, 1)

        self.mpa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        distilled_c1 = self.act(self.c0_d(input))

        l_c1 = (self.c1_l2(self.act(self.c1_l1(input))))
        m_c1 = (self.c1_m2(self.act(self.c1_m1(input))))
        # r_c1 = (self.c1_r(input))

        r_c1 = self.act(l_c1 + m_c1 + distilled_c1)

        distilled_c2 = self.act(self.c1_d(r_c1))  # 1/2 size


        l_c2 = (self.c2_l2(self.act(self.c2_l1(r_c1))))
        m_c2 = (self.c2_m2(self.act(self.c2_m1(r_c1))))

        r_c2 = self.act(l_c2 + m_c2 + r_c1 + distilled_c2 + distilled_c1)

        distilled_c3 = self.act(self.c2_d(r_c2))  # 1/4 size

        l_c3 = (self.c3_l2(self.act(self.c3_l1(r_c2))))
        m_c3 = (self.c3_m2(self.act(self.c3_m1(r_c2))))
        r_c3 = self.act(l_c3 + m_c3 + r_c2 + distilled_c3 + distilled_c2 + distilled_c1)
        # r_c3 = l_c3 + m_c3 + r_c2 + distilled_c3 + distilled_c2 + distilled_c1

        r_c4 = self.act(self.c4(r_c3))  # 这个relu 是不是可以省略。

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1) #distilled_c1 + distilled_c2 + distilled_c3
        out_fused = self.mpa(self.c5(out))

        return out_fused
