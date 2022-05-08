#import block as B
import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 6, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 6, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BAM(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(BAM, self).__init__()

        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        out1 = self.ca(x)
        out2 = self.sa(x)


        return out1*out2*x



class LAM_Module(nn.Module):
    """ Layer attention module"""
    def __init__(self, in_dim):
        super(LAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out + x
        out = out.view(m_batchsize, -1, height, width)
        return out



class conv_layer1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_layer1, self).__init__()


        self.conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3), padding=(0, 1), groups=1)
        self.conv1_2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1), padding=(1, 0), groups=1)
        self.conv1_3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1), groups=1)

    def forward(self,x):
        out1 = self.conv1_3(x)
        out2 = self.conv1_1(x)
        out3 = self.conv1_2(x)
        return out1+out2+out3




def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    wn = lambda x: torch.nn.utils.weight_norm(x)
    return wn(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups))


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
    if act_type == 'silu':
        layer = nn.SELU(inplace)
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


class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        f = n_feats // 4
        self.conv1 = wn(conv(n_feats, f, kernel_size=1))
        self.conv_f = wn(conv(f, f, kernel_size=1))
        self.conv_max = wn(conv(f, f, kernel_size=3, padding=1))
        self.conv2 = wn(conv(f, f, kernel_size=3, stride=2, padding=0))
        self.conv3 = wn(conv(f, f, kernel_size=3, padding=1))
        self.conv3_ = wn(conv(f, f, kernel_size=3, padding=1))
        self.conv4 = wn(conv(f, n_feats, kernel_size=1))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.SiLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))+v_max
        c3 = self.relu(self.conv3(v_range))+v_range
        c3 = self.conv3_(c3)+c3
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


class RFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.5):
        super(RFDB, self).__init__()
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc,3)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc,3)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer(self.remaining_channels, self.rc,3)
        self.c4 = conv_layer(self.remaining_channels, self.dc,3)
        self.act = activation('silu')
        self.c5 = conv_layer(self.dc * 4, in_channels, 1)
        #self.BAM =BAM(32,16)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))
        #out_fused = self.BAM(self.c5(out))

        return out_fused


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer1(in_channels, out_channels * (upscale_factor ** 2))
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


# def make_model(args, parent=False):
#     model = team38_rfdnext()
#     return model


class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class RLCSR(nn.Module):
    def __init__(self, in_nc=3, nf=32, num_modules=6, out_nc=3, upscale=4):
        super(RLCSR, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=in_nc, out_channels=nf, kernel_size=(1, 3), padding=(0, 1), groups=1,
                                 bias=False)
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.BAM= BAM(64,16)
        self.conv1_2 = nn.Conv2d(in_channels=in_nc, out_channels=nf, kernel_size=(3, 1), padding=(1, 0), groups=1,
                                 bias=False)
        self.conv1_3 = nn.Conv2d(in_channels=in_nc, out_channels=nf, kernel_size=(3, 3), padding=(1, 1), groups=1,
                                 bias=False)
        self.convl11 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=(1, 3), padding=(0, 1), groups=1,
                                 bias=False)
        self.convl22 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=(3, 1), padding=(1, 0), groups=1,
                                 bias=False)
        self.convl33 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=(3, 3), padding=(1, 1), groups=1,
                                 bias=False)
        # self.fea_conv = conv_layer1(in_nc, nf)
        self.convl1 = nn.Conv2d(in_channels=nf * 2, out_channels=nf, kernel_size=(1, 3), padding=(0, 1), groups=1,
                                bias=False)
        self.convl2 = nn.Conv2d(in_channels=nf * 2, out_channels=nf, kernel_size=(3, 1), padding=(1, 0), groups=1,
                                bias=False)
        self.convl3 = nn.Conv2d(in_channels=nf * 2, out_channels=nf, kernel_size=(3, 3), padding=(1, 1), groups=1,
                                bias=False)

        self.B1 = RFDB(in_channels=nf)
        self.B2 = RFDB(in_channels=nf)
        self.B3 = RFDB(in_channels=nf)
        self.B4 = RFDB(in_channels=nf)
        self.B5 = RFDB(in_channels=nf)
        self.B6 = RFDB(in_channels=nf)

        self.reduction1 = wn(nn.Conv2d(nf * 2, nf, 1))
        self.reduction2 = wn(nn.Conv2d(nf * 2, nf, 1))
        self.reduction3 = wn(nn.Conv2d(nf * 2, nf, 1))
        self.reduction4 = wn(nn.Conv2d(nf * 2, nf, 1))
        self.reduction5 = wn(nn.Conv2d(nf * 2, nf, 1))

        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='silu')
        self.la = LAM_Module(nf * num_modules)
        self.LR_conv = conv_layer(nf, nf, kernel_size=3)
        self.last_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=4)
        self.scale_idx = 0
        self.last = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.res_scale = Scale(1)
        self.in_scale = Scale(1)
        self.outconv = nn.Conv2d(6,out_nc,3,1,1)
    def forward(self, input):
        out_fea = self.conv1_2(input) + self.conv1_1(input) + self.conv1_3(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1) + out_B1
        out_B3 = self.B3(out_B2) + out_B2
        out_B4 = self.B4(out_B3) + out_B3
        out_B5 = self.B5(out_B4) + out_B4
        out_B6 = self.B6(out_B5) + out_B5

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_B11 = out_B.unsqueeze(1)
        out2 = self.la(out_B11)
        # print(out2.size(),'out21')
        out2 = self.convl11(out2) + self.convl22(out2) + self.convl33(out2)
        # print(out2.size(),'out22')

        res1= self.reduction1(channel_shuffle(torch.cat([out_B1,out_B2],dim=1),2))
        res2 = self.reduction2(channel_shuffle(torch.cat([res1, out_B3], dim=1), 2))
        res3 = self.reduction3(channel_shuffle(torch.cat([res2, out_B4], dim=1), 2))
        res3 = self.reduction4(channel_shuffle(torch.cat([res3, out_B5], dim=1), 2))
        out_lr = self.reduction5(channel_shuffle(torch.cat([res3, out_B6], dim=1), 2))






        # out_lr = self.LR_conv(out_B)
        # print(out_lr.size(),'out_lr')
        out = torch.cat([out2, out_lr], 1)
        out = self.BAM(out)
        res = self.convl1(out) + self.convl2(out) + self.convl3(out)


        output = self.upsampler(res)


        output = output + F.interpolate(input, scale_factor=4, mode='bicubic', align_corners=False)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
