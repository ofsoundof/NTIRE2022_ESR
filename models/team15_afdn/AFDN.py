import torch
import torch.nn as nn
import models.team15_afdn.block as B


def make_model(args, parent=False):
    model = AFDN()
    return model


class AFDN(nn.Module):
    def __init__(self, in_nc=3, nf=46, num_modules=4, out_nc=3, upscale=4):
        super(AFDN, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = B.AFDB(in_channels=nf)
        self.B2 = B.AFDB(in_channels=nf)
        self.B3 = B.AFDB(in_channels=nf)
        self.B4 = B.AFDB(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=4)
        self.scale_idx = 0
        self.warm_model()

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

    def warm_model(self):
        self.fea_conv = self.fea_conv.cuda()
        self.B1 = self.B1.cuda()
        self.B2 = self.B2.cuda()
        self.B3 = self.B3.cuda()
        self.B4 = self.B4.cuda()
        self.c = self.c.cuda()
        self.LR_conv = self.LR_conv.cuda()
        self.upsampler = self.upsampler.cuda()
        input = torch.FloatTensor(1, 3, 256, 256).cuda()
        self.forward(input)