import torch.nn as nn

import torch
import torch.nn.functional as F


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    #wn = lambda x: torch.nn.utils.weight_norm(x)
    padding = int((kernel_size - 1) / 2) * dilation
    return (nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
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

    c = (nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups))
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

class LapSA(nn.Module):
    def __init__(self, n_feats, conv=nn.Conv2d):
        super(LapSA, self).__init__()
        f = n_feats // 4
        self.squeeze = conv(n_feats, f, kernel_size=1)
        self.fuse = conv(f+n_feats, n_feats, kernel_size=1)

        self.down1 = nn.Sequential(
                        nn.MaxPool2d(2, 2),
                        conv(f, f, 3, 1, 1)
                    )
        self.down2 = nn.Sequential(
                        nn.MaxPool2d(2, 2),
                        conv(f, f, 3, 1, 1)
                    )
        self.down3 = nn.Sequential(
                        nn.MaxPool2d(2, 2),
                        conv(f, f, 3, 1, 1)
                    )
        self.excite = conv(3*f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.act = nn.ReLU(True)


    def forward(self, x):
        s = self.act(self.squeeze(x))
        # Pyra 1
        d1 = self.act(self.down1(s))
        u1 = F.interpolate(d1, (s.size(2), s.size(3)), mode='bilinear', align_corners=False) 
        h1 = s - u1
        #h1 = u1

        # Pyra 2
        d2 = self.act(self.down2(d1))
        u2 = F.interpolate(d2, (d1.size(2), d1.size(3)), mode='bilinear', align_corners=False) 
        h2 = d1 - u2
        #h2 = u2
        h2 = F.interpolate(h2, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 

        #'''
        # Pyra 3
        d3 = self.act(self.down3(d2))
        u3 = F.interpolate(d3, (d2.size(2), d2.size(3)), mode='bilinear', align_corners=False) 
        h3 = d2 - u3
        #h3 = u3
        h3 = F.interpolate(h3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        #'''

        m = self.excite(torch.cat((h1, h2, h3), dim=1))
        m = self.sigmoid(m)
        out = self.fuse(torch.cat((x*m, h1), dim=1))
        
        return out


class UESA(nn.Module):
    def __init__(self, num_feats):
        super(UESA, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.convs = nn.Sequential(
                    (nn.Conv2d(num_feats, num_feats//8, 3, 1, 1)),
                    nn.ReLU(True),
                    (nn.Conv2d(num_feats//8, num_feats, 3, 1, 1)),
                )

    def forward(self, x):
        mask = F.max_pool2d(x, kernel_size=8, stride=4)
        mask = self.convs(mask)
        mask = F.interpolate(mask, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        mask = self.sigmoid(mask+x)
        
        return x * mask

class SCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = torch.std(x, dim=[2,3], keepdim=True)
        y_mu = torch.mean(y, dim=1, keepdim=True)
        y_sigma = torch.std(y, dim=1, keepdim=True)
        y = (y - y_mu) / (y_sigma + 1e-5)
        y = self.conv_du(y)
        return x * y


class CFRB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(CFRB, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        #self.rc = self.remaining_channels = int(in_channels - self.distilled_channels)
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc*4, in_channels, 1)
        #self.esa = ESA(in_channels, nn.Conv2d)
        self.ca = CALayer(in_channels)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.ca(self.c5(out)) 

        return out_fused

#wn = lambda x: torch.nn.utils.weight_norm(x)

class RFDN_v2(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDN_v2, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = (nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=False))
        self.c1_r = (conv_layer(in_channels, self.rc, 3))
        self.c2_d = (nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=False))
        self.c2_r = (conv_layer(self.remaining_channels, self.rc, 3))
        self.c3_d = (nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=False))
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc*4, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out)) 

        return out_fused

class CALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class FDEB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(FDEB, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.exp = 5
        self.c1_d = (nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=False))
        self.c1_r = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels*self.exp, 1, 1, 0),
                    nn.LeakyReLU(0.05),
                    nn.Conv2d(in_channels*self.exp, in_channels, 1, 1, 0),
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                )
        self.c2_d = (nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=False))
        #self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c2_r = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels*self.exp, 1, 1, 0),
                    nn.LeakyReLU(0.05),
                    nn.Conv2d(in_channels*self.exp, in_channels, 1, 1, 0),
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                )
        self.c3_d = (nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=False))
        #self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c3_r = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels*self.exp, 1, 1, 0),
                    nn.LeakyReLU(0.05),
                    nn.Conv2d(in_channels*self.exp, in_channels, 1, 1, 0),
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                )
        self.c4 = conv_layer(self.remaining_channels, self.remaining_channels, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc*3+self.remaining_channels, in_channels, 1)
        #self.sa = ESA(in_channels, nn.Conv2d)
        self.sa = LapSA(in_channels)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r((input)))
        r_c1 = (r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r((r_c1)))
        r_c2 = (r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r((r_c2)))
        r_c3 = (r_c3+r_c2)

        r_c4 = (self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.sa(self.c5(out)) 

        return out_fused

class TSB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(TSB, self).__init__()
        lr_ch = in_channels//4
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = (nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=False))
        self.c1_r = (conv_layer(in_channels, self.rc, 3))
        self.c2_d = (nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=False))
        self.c2_r = (conv_layer(self.remaining_channels, self.rc, 3))
        self.c3_d = (nn.Conv2d(in_channels, self.dc, 1, 1, 0, bias=False))
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc*4, in_channels, 1)

        self.lr_convs = nn.Sequential(
                    nn.Conv2d(lr_ch, lr_ch, 3, 1, 1),
                    nn.LeakyReLU(0.05),
                    nn.Conv2d(lr_ch, lr_ch, 3, 1, 1),
                )

        self.mask_convs = nn.Sequential(
                    nn.Conv2d(lr_ch+in_channels, lr_ch, 1, 1, 0),
                    nn.LeakyReLU(0.05),
                    nn.Conv2d(lr_ch, lr_ch, 3, 1, 1),
                    nn.LeakyReLU(0.05),
                    nn.Conv2d(lr_ch, in_channels, 1, 1, 0),
                    nn.Sigmoid()
                )

    def forward(self, input, input_lr):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = (self.c5(out)) 

        lr_feat = self.lr_convs(input_lr)
        hr_feat = F.max_pool2d(out_fused, kernel_size=4, stride=4)
        mask = torch.cat([lr_feat, hr_feat], dim=1)
        mask = self.mask_convs(mask)
        mask = F.interpolate(mask, (input.size(2), input.size(3)), mode='bilinear', align_corners=False) 
        out_masked = out_fused * mask


        return out_masked, lr_feat


class FDB(nn.Module):
    def __init__(self, num_feat):
        super(FDB, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat*5, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.act = nn.ReLU(True)

    def forward(self, x):
        N,C,H,W = x.size()
        out = self.act(self.conv1(x))
        out, out_d = torch.split(out, (C*4, C), dim=1)
        out = out.view(N, -1, C, H, W)
        out = torch.sum(out, dim=1)
        out = self.conv2(out)


        return out, out_d


class FDN(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(FDN, self).__init__()
        self.esa = ESA(in_channels, nn.Conv2d)
        self.fdb1 = FDB(in_channels)
        self.fdb2 = FDB(in_channels)
        self.fdb3 = FDB(in_channels)
        #self.fdb4 = FDB(in_channels)
        self.conv = nn.Conv2d(in_channels*4, in_channels, 1, 1, 0)

    def forward(self, x):
        f1, d1 = self.fdb1(x)
        f2, d2 = self.fdb2(f1)
        f3, d3 = self.fdb3(f2)
        f = torch.cat([f3, d1, d2, d3], dim=1)
        out = self.esa(self.conv(f))
         
        return out



def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)
