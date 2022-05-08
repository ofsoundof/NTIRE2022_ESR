# A majority of our code is overhauled from https://github.com/njulj/RFDN
##############################################################################
# MIT License

# Copyright (c) 2021 njulj

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################

import torch.nn as nn
import torch
import models.team38_rfdnext.rfdn_block as B


def get_block(block_type):
    if block_type == "RFDB":
        return B.RFDB
    elif block_type == "MRB":
        return B.MRB


class RFDN(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4, block_type="RFDB", act_type="lrelu", **kwargs):
        super(RFDN, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)
        
        block = get_block(block_type)
        self.B1 = block(in_channels=nf,  act_type=act_type)
        self.B2 = block(in_channels=nf,  act_type=act_type)
        self.B3 = block(in_channels=nf,  act_type=act_type)
        self.B4 = block(in_channels=nf,  act_type=act_type)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type=act_type)

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        self.upsampler = B.pixelshuffle_block(nf, out_nc, upscale_factor=upscale)


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