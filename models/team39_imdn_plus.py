import torch
import torch.nn as nn
from collections import OrderedDict


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


class ShortcutBlock(nn.Module):
    # Elementwise sum the output of a submodule to its input
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


class IMDB_plus(nn.Module):
    def __init__(self, in_channels, distillation_rate=1 / 6):
        super(IMDB_plus, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)  # 6
        self.c1 = conv_layer(in_channels, in_channels, 3)  # 36 --> 36
        self.c2 = conv_layer(self.distilled_channels * 5, self.distilled_channels * 5, 3)  # 30 --> 30
        self.c3 = conv_layer(self.distilled_channels * 4, self.distilled_channels * 4, 3)  # 24 --> 24
        self.c4 = conv_layer(self.distilled_channels * 3, self.distilled_channels * 3, 3)  # 18 --> 18
        self.c5 = conv_layer(self.distilled_channels * 2, self.distilled_channels * 2, 3)  # 12 --> 12
        self.c6 = conv_layer(self.distilled_channels * 1, self.distilled_channels * 1, 3)  # 6 --> 6
        self.act = nn.SiLU()
        self.c7 = conv_layer(self.distilled_channels * 6, in_channels, 1)

    def forward(self, x):
        out_c1 = self.act(self.c1(x))  # 36 --> 36
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.distilled_channels * 5),
                                                 dim=1)  # 6, 30
        out_c2 = self.act(self.c2(remaining_c1))  # 30 --> 30
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.distilled_channels * 4),
                                                 dim=1)  # 6, 24
        out_c3 = self.act(self.c3(remaining_c2))  # 24 --> 24
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.distilled_channels * 3),
                                                 dim=1)  # 6, 18
        out_c4 = self.act(self.c4(remaining_c3))  # 18 --> 18
        distilled_c4, remaining_c4 = torch.split(out_c4, (self.distilled_channels, self.distilled_channels * 2),
                                                 dim=1)  # 6, 12
        out_c5 = self.act(self.c5(remaining_c4))  # 12 --> 12
        distilled_c5, remaining_c5 = torch.split(out_c5, (self.distilled_channels, self.distilled_channels),
                                                 dim=1)  # 6, 6
        out_c6 = self.act(self.c6(remaining_c5))  # 6 --> 6

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, distilled_c4, distilled_c5, out_c6], dim=1)
        out_fused = self.c7(out) + x
        return out_fused


class IMDN_plus(nn.Module):
    def __init__(self, in_nc, nf, nb, out_nc):
        super(IMDN_plus, self).__init__()

        fea_conv = conv_layer(in_nc, nf, kernel_size=3)
        rb_blocks = [IMDB_plus(nf) for _ in range(nb)]
        LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        upsampler = upsample_block(nf, out_nc, upscale_factor=4)

        self.FEM = sequential(fea_conv, ShortcutBlock(sequential(*rb_blocks, LR_conv)))

        self.RM = sequential(*upsampler)

    def forward(self, x):
        out_CFEM = self.FEM(x)
        out = self.RM(out_CFEM)
        return out

