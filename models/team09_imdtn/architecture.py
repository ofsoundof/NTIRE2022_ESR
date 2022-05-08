import torch.nn as nn
from . import block as B
import torch
# For any upscale factors

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


class IMDTN(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=3, out_nc=3, upscale=4):
        super(IMDTN, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        # IMDBs
        self.IMDTB1 = B.IMDTBlock(in_channels=nf)
        self.IMDTB2 = B.IMDTBlock(in_channels=nf)
        self.IMDTB3 = B.IMDTBlock(in_channels=nf)
        
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)
        
        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)


    def forward(self, input):
        _, _, h_old, w_old = input.size()
        h_pad = (h_old // 6 + 1) * 6 - h_old
        w_pad = (w_old // 6 + 1) * 6 - w_old
        input = torch.cat([input, torch.flip(input, [2])], 2)[:, :, :h_old + h_pad, :]
        input = torch.cat([input, torch.flip(input, [3])], 3)[:, :, :, :w_old + w_pad]
        
        out_fea = self.fea_conv(input)
        out_B1 = self.IMDTB1(out_fea)
        out_B2 = self.IMDTB2(out_B1)
        out_B3 = self.IMDTB3(out_B2)
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        # out_lr = out_B + out_fea
        output = self.upsampler(out_lr)

        return output[..., :h_old *4, :w_old * 4]





