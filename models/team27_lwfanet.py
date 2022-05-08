import torch
from torch import nn as nn
from torch.nn import functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, in_planes):
        super(SpatialAttention, self).__init__()

        self.sa_conv = nn.Conv2d(in_planes, 1, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        return self.sigmoid(self.sa_conv(x))


class LWFA(nn.Module):

    def __init__(self, num_feat):
        super(LWFA, self).__init__()

        # 3x3
        self.conv1_1 = nn.Conv2d(num_feat, num_feat // 4, 1, 1)
        self.conv1_2 = nn.Conv2d(num_feat // 4, num_feat // 4, 3, 1, 1)

        # 5x5
        self.conv2_1 = nn.Conv2d(num_feat, num_feat // 4, 1, 1)
        self.conv2_2 = nn.Conv2d(num_feat // 4, num_feat // 4, 3, 1, 1)
        self.conv2_3 = nn.Conv2d(num_feat // 4, num_feat // 4, 3, 1, 1)

        # 7x7
        self.conv3_1 = nn.Conv2d(num_feat, num_feat // 4, 1, 1)
        self.conv3_2 = nn.Conv2d(num_feat // 4, num_feat // 4, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(num_feat // 4, num_feat // 4, 3, 1, 1)
        self.conv3_4 = nn.Conv2d(num_feat // 4, num_feat // 4, 3, 1, 1)

        # 9x9
        self.conv4_1 = nn.Conv2d(num_feat, num_feat // 4, 1, 1)
        self.conv4_2 = nn.Conv2d(num_feat // 4, num_feat // 4, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(num_feat // 4, num_feat // 4, 3, 1, 1)
        self.conv4_4 = nn.Conv2d(num_feat // 4, num_feat // 4, 3, 1, 1)
        self.conv4_5 = nn.Conv2d(num_feat // 4, num_feat // 4, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.ca = ChannelAttention(num_feat)
        self.sa1 = SpatialAttention(num_feat)
        self.sa2 = SpatialAttention(num_feat)  

    def forward(self, x):
        branch1 = self.lrelu(self.conv1_2(self.lrelu(self.conv1_1(x))))
        branch2 = self.lrelu(self.conv2_3(self.lrelu(self.conv2_2(self.lrelu(self.conv2_1(x))))))
        branch3 = self.lrelu(self.conv3_4(self.lrelu(self.conv3_3(self.lrelu(self.conv3_2(self.lrelu(self.conv3_1(x))))))))
        branch4 = self.lrelu(self.conv4_5(self.lrelu(self.conv4_4(self.lrelu(self.conv4_3(self.lrelu(self.conv4_2(self.lrelu(self.conv4_1(x))))))))))

        out = torch.cat((branch1, branch2, branch3, branch4), 1)

        out_ca = self.ca(out) * out
        out_sa = self.sa1(out) * out
        x_sa = self.sa2(x) * x
        return out_ca + out_sa + x_sa


def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)



class LWFANet(nn.Module):

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=4):
        super(LWFANet, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(LWFA, num_block, num_feat=num_feat)
        # upsample
        self.conv_body = nn.Conv2d(num_feat,num_feat, 3, 1, 1)
        self.conv_L = nn.Conv2d(num_feat, 64, 1, 1)
        self.conv_up1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.conv_L(feat)
        feat = self.lrelu(
             self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(
             self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out