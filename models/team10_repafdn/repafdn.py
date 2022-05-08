import torch
from torch import nn as nn

from models.team10_repafdn.block import (
    PA,
    FDB,
    FDB_S,
    conv_block,
    conv_layer,
    pixelshuffle_block,
)


class RePAFDN(nn.Module):
    """
    Re-parameterized Pixel Attention Feature Distillation Network
    """

    def __init__(
        self,
        in_nc=3,
        nf=48,
        out_nc=3,
        upscale=4,
        act_type="lrelu",
        ds_rate=0.25,
        rgb_range=255.0,
        dc=24,
    ):
        super(RePAFDN, self).__init__()
        self.rgb_range = rgb_range
        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = FDB_S(in_channels=nf, act_type=act_type, dc=dc)
        self.B2 = FDB_S(in_channels=nf, act_type=act_type, dc=dc)
        self.B3 = FDB_S(in_channels=nf, act_type=act_type, dc=dc)
        self.B4 = FDB(in_channels=nf, act_type=act_type, distillation_rate=ds_rate)
        self.c = conv_block(nf * 4, nf, kernel_size=1, act_type=act_type)

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        self.pa = PA(nf)
        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)

    def forward(self, input):
        out_fea = self.fea_conv(input)

        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B)
        out_lr = self.pa(out_lr)
        out_lr = out_lr + out_fea
        output = self.upsampler(out_lr)
        return output
