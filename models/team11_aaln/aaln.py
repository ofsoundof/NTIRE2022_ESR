from models.team11_aaln import common

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

### Lattice网络
def make_model(args, parent=False):
    model = AALN()
    return model

# class depthconv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(depthconv, self).__init__()
#         self.d_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch)
#         self.p_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         input = self.d_conv(x)
#         output = self.p_conv(input)
#         return output


class lightsaatt(nn.Module):
    def __init__(self, in_ch):
        super(lightsaatt, self).__init__()
        self.d_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch)
        self.act = nn.PReLU(init=0.05)
        self.p_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = self.act(self.d_conv(x))
        output = self.sigmoid(self.p_conv(input))
        final = output * x
        return final

def mean_channels(x):
    assert(x.dim() == 4)
    spatial_sum = x.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (x.shape[2] * x.shape[3])

def stdv(x):
    assert(x.dim() == 4)
    x_mean = mean_channels(x)
    x_var = (x - x_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (x.shape[2] * x.shape[3])
    return x_var.pow(0.5)

class NCA(nn.Module):
    def __init__(self, n_feats, reduction):
        super(NCA, self).__init__()
        # 上路平均池化
        self.upper_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feats, n_feats // reduction, 1, 1, 0),
            nn.ReLU(inplace=True)
        )
        # 下路标准差
        self.stdv = stdv
        self.lower_branch = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // reduction, 1, 1, 0),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(n_feats // reduction, n_feats, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        upper = self.upper_branch(x)
        lower = self.stdv(x)
        lower = self.lower_branch(lower)

        out = self.fuse(torch.add(upper, lower))

        return out * x

# basic block
class DSAB1(nn.Module):
    def __init__(self, n_feats):
        super(DSAB1, self).__init__()

        self.conv_3 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            nn.PReLU(init=0.05)
        )

        # the convolution kernel of 5x5
        self.conv_5 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            nn.PReLU(init=0.05)
        )
        self.att = NCA(n_feats*2, 12)
        self.conv_1 = nn.Conv2d(n_feats*2, n_feats, 1, padding=0, stride=1)


    def forward(self, x):
        output_3 = self.conv_3(x)
        output_5 = self.conv_5(output_3)
        output_35 = torch.cat([output_3, output_5], 1)
        output = self.att(output_35)
        output = self.conv_1(output)
        fin = output + x
        return fin


# class DSAB2(nn.Module):
#     def __init__(self, n_feats, conv=common.default_conv):
#         super(DSAB2, self).__init__()
#
#         kernel_size_3 = 3
#
#         self.conv_1 = nn.Conv2d(n_feats*2, n_feats, 1, padding=0, stride=1)
#         # the convolution kernel of 3x3
#         self.conv_3 = nn.Sequential(
#             conv(n_feats, n_feats, kernel_size_3),
#             nn.PReLU(init=0.05)
#         )
#
#         # the convolution kernel of 5x5
#         self.conv_5 = nn.Sequential(
#             nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, dilation=1, padding=1),
#             nn.PReLU(init=0.05)
#         )
#
#
#     def forward(self, x):
#         output_3 = self.conv_3(x)
#         output_5 = self.conv_5(output_3)
#
#         output = self.conv_1(torch.cat([output_3, output_5], 1))
#
#         output += x
#
#         return output

#the block of net

class attBlock(nn.Module):
    def __init__(self, n_feats):
        super(attBlock, self).__init__()

        self.conv_block0 = DSAB1(n_feats)
        self.conv_block1 = DSAB1(n_feats)
        self.compress = nn.Conv2d(2 * n_feats, n_feats, kernel_size=1, padding=0, bias=True)
        self.att = lightsaatt(n_feats)

    def forward(self, x):
        # analyse unit
        x_feature_shot = self.conv_block0(x)
        # synthes_unit
        x_feat_long = self.conv_block1(x_feature_shot)
        out = torch.cat((x_feature_shot, x_feat_long), 1)
        out = self.att(self.compress(out)) + x

        return out




class AALN(nn.Module):
    def __init__(self):
        super(AALN, self).__init__()

        n_feats = 54
        scale = 4
        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift()
        self.add_mean = common.MeanShift(sign=1)
        # define head module
        self.input = nn.Sequential(
            nn.Conv2d(3, n_feats, 3, stride=1, padding=1),
            nn.PReLU(init=0.05),
            nn.Conv2d(n_feats, n_feats, 3, stride=1, padding=1),
            nn.PReLU(init=0.05)
        )
        # define body module
        self.B1 = attBlock(n_feats)
        self.B2 = attBlock(n_feats)
        self.B3 = attBlock(n_feats)
        self.B4 = attBlock(n_feats)
        self.tail_conv = nn.Conv2d(n_feats * 4, n_feats, kernel_size=1, stride=1, padding=0)
        # define tail module

        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            nn.PReLU(init=0.05),
            nn.Conv2d(n_feats, 3 * (scale**2), 1, 1, padding=0),
            nn.PixelShuffle(scale)
        )



    def forward(self, x):
        x = self.sub_mean(x)
        x_in = self.input(x)

        res1 = self.B1(x_in)
        res2 = self.B2(res1)
        res3 = self.B3(res2)
        res4 = self.B4(res3)

        out_B = torch.cat([res1, res2, res3, res4], dim=1)
        out_lr = self.tail_conv(out_B) + x_in
        out = self.upsample(out_lr)

        sr_1 = self.add_mean(out)
        inter_res = nn.functional.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)
        sr_out = torch.add(sr_1, inter_res)

        return sr_out

