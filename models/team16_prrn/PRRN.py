import torch
import torch.nn as nn
import torch.nn.functional as F
from models.team16_prrn.basicblock import conv_layer, pixelshuffle_block, activation

class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out

class CA_TF(nn.Module):
    '''Channel Attention in the First Stage Attention (FSA)'''
    def __init__(self, nf):
        super(CA_TF, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        y = self.sigmoid(self.conv1(self.avg_pool(x)))
        return x*y

class PA_TF(nn.Module):
    '''Pixel Attention in the First Stage Attention (FSA)'''
    def __init__(self, nf):

        super(PA_TF, self).__init__()
        self.pa = PA(nf)
        self.ca = CA_TF(nf)
        self.conv1 = nn.Conv2d(nf,nf,1)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = conv_layer(nf, nf, kernel_size=3)

    def forward(self, x):

        y_1_1 = self.pa(x)
        y_1_2 = self.ca(x)
        y_1 = self.sigmoid(self.conv1(y_1_1 + y_1_2))

        y_2 = self.conv2(x)
        return y_1 * y_2

class PRRB(nn.Module):
    def __init__(self, nf, reduction = 2):
        super(PRRB, self).__init__()
        group_width = nf // reduction

        self.conv1_1 = nn.Conv2d(nf, group_width, kernel_size = 1, bias = False)
        self.conv1_2 = nn.Conv2d(nf, group_width, kernel_size = 1, bias = False)

        self.pgam_1 = PA_TF(group_width)
        self.sigmoid = nn.Sigmoid()
        self.conv3_1 = conv_layer(in_channels=group_width, out_channels=group_width, kernel_size=3)
        self.conv3_2 = conv_layer(in_channels=group_width, out_channels=group_width, kernel_size=3)
        self.conv3_3 = conv_layer(in_channels=group_width, out_channels=group_width, kernel_size=3)

        self.conv1_end = nn.Conv2d(nf, nf, kernel_size = 1, bias = False)
        self.sca = CA_TF(nf) # Second Channel Attention (SCA)
        self.act = activation('silu',inplace=True)

    def forward(self, x):
        out_a = self.act(self.conv1_1(x))
        out_b = self.act(self.conv1_2(x))

        #upper branch
        attention = self.sigmoid(self.pgam_1(out_a))
        out_a_end = self.act(self.conv3_2(torch.mul(attention, self.conv3_1(out_a))))

        #bottom branch
        out_b_end = self.act(self.conv3_3(out_b))

        out_mid_1 = self.act(self.conv1_end(torch.cat([out_a_end, out_b_end], dim=1)))
        out_mid = self.sca(out_mid_1)
        out = out_mid + x

        return out

'''
***********************************************************************************************************************
'''

class PRRN(nn.Module): 
    '''
    Progressive Representation Re-Calibration Network (team16_prrn) for Lightweight Super-Resolution
    '''

    def __init__(self, in_nc=3, out_nc=3, nf=40, scale=4):
        super(PRRN, self).__init__()

        ### first convolution
        self.conv_first = conv_layer(in_nc, nf, kernel_size=3)

        ### main blocks
        self.scpa_v1 = PRRB(nf)
        self.scpa_v2 = PRRB(nf)
        self.scpa_v3 = PRRB(nf)
        self.scpa_v4 = PRRB(nf)
        self.scpa_v5 = PRRB(nf)
        self.scpa_v6 = PRRB(nf)
        self.scpa_v7 = PRRB(nf)
        self.scpa_v8 = PRRB(nf)
        self.scpa_v9 = PRRB(nf)
        self.scpa_v10 = PRRB(nf)
        self.scpa_v11 = PRRB(nf)
        self.scpa_v12 = PRRB(nf)
        self.scpa_v13 = PRRB(nf)
        self.scpa_v14 = PRRB(nf)
        self.scpa_v15 = PRRB(nf)
        self.scpa_v16 = PRRB(nf)


        self.conv1_mid_1 = conv_layer(in_channels=nf * 2, out_channels=nf, kernel_size=1)
        self.conv1_mid_2 = conv_layer(in_channels=nf * 2, out_channels=nf, kernel_size=1)
        self.conv1_mid_3 = conv_layer(in_channels=nf * 2, out_channels=nf, kernel_size=1)
        self.conv1_mid_4 = conv_layer(in_channels=nf * 2, out_channels=nf, kernel_size=1)
        self.conv1_mid_5 = conv_layer(in_channels=nf * 2, out_channels=nf, kernel_size=1)
        self.conv1_mid_6 = conv_layer(in_channels=nf * 2, out_channels=nf, kernel_size=1)
        self.conv1_mid_7 = conv_layer(in_channels=nf * 2, out_channels=nf, kernel_size=1)
        self.conv1_mid_8 = conv_layer(in_channels=nf * 2, out_channels=nf, kernel_size=1)
        self.conv1_mid_9 = conv_layer(in_channels=nf * 2, out_channels=nf, kernel_size=1)
        self.conv1_mid_10 = conv_layer(in_channels=nf * 2, out_channels=nf, kernel_size=1)
        self.conv1_mid_11 = conv_layer(in_channels=nf * 2, out_channels=nf, kernel_size=1)
        self.conv1_mid_12 = conv_layer(in_channels=nf * 2, out_channels=nf, kernel_size=1)
        self.conv1_mid_13 = conv_layer(in_channels=nf * 2, out_channels=nf, kernel_size=1)
        self.conv1_mid_14 = conv_layer(in_channels=nf * 2, out_channels=nf, kernel_size=1)
        self.conv1_mid_15 = conv_layer(in_channels=nf * 2, out_channels=nf, kernel_size=1)
        self.conv1_mid_16 = conv_layer(in_channels=nf * 2, out_channels=nf, kernel_size=1)

        self.conv3_end = conv_layer(nf, nf, kernel_size=3)
        #### upsampling
        self.upsampler = pixelshuffle_block(nf, out_nc, upscale_factor=scale)

    def forward(self, x):

        input_ = self.conv_first(x)
        fea_v1 = self.scpa_v1(input_)
        fea_stack_v1 = torch.cat([input_, fea_v1], dim=1)
        fea_stack_v1 = self.conv1_mid_1(fea_stack_v1)

        fea_v2 = self.scpa_v2(fea_stack_v1)
        fea_stack_v2 = torch.cat([input_, fea_v2], dim=1)
        fea_stack_v2 = self.conv1_mid_2(fea_stack_v2)

        fea_v3 = self.scpa_v3(fea_stack_v2)
        fea_stack_v3 = torch.cat([input_, fea_v3], dim=1)
        fea_stack_v3 = self.conv1_mid_3(fea_stack_v3)

        fea_v4 = self.scpa_v4(fea_stack_v3)
        fea_stack_v4 = torch.cat([input_, fea_v4], dim=1)
        fea_stack_v4 = self.conv1_mid_4(fea_stack_v4)

        fea_v5 = self.scpa_v5(fea_stack_v4)
        fea_stack_v5 = torch.cat([input_, fea_v5], dim=1)
        fea_stack_v5 = self.conv1_mid_5(fea_stack_v5)

        fea_v6 = self.scpa_v6(fea_stack_v5)
        fea_stack_v6 = torch.cat([input_, fea_v6], dim=1)
        fea_stack_v6 = self.conv1_mid_6(fea_stack_v6)

        fea_v7 = self.scpa_v7(fea_stack_v6)
        fea_stack_v7 = torch.cat([input_, fea_v7], dim=1)
        fea_stack_v7 = self.conv1_mid_7(fea_stack_v7)

        fea_v8 = self.scpa_v8(fea_stack_v7)
        fea_stack_v8 = torch.cat([input_, fea_v8], dim=1)
        fea_stack_v8 = self.conv1_mid_8(fea_stack_v8)

        fea_v9 = self.scpa_v9(fea_stack_v8)
        fea_stack_v9 = torch.cat([input_, fea_v9], dim=1)
        fea_stack_v9 = self.conv1_mid_9(fea_stack_v9)

        fea_v10 = self.scpa_v10(fea_stack_v9)
        fea_stack_v10 = torch.cat([input_, fea_v10], dim=1)
        fea_stack_v10 = self.conv1_mid_10(fea_stack_v10)

        fea_v11 = self.scpa_v11(fea_stack_v10)
        fea_stack_v11 = torch.cat([input_, fea_v11], dim=1)
        fea_stack_v11 = self.conv1_mid_11(fea_stack_v11)

        fea_v12 = self.scpa_v12(fea_stack_v11)
        fea_stack_v12 = torch.cat([input_, fea_v12], dim=1)
        fea_stack_v12 = self.conv1_mid_12(fea_stack_v12)

        fea_v13 = self.scpa_v13(fea_stack_v12)
        fea_stack_v13 = torch.cat([input_, fea_v13], dim=1)
        fea_stack_v13 = self.conv1_mid_13(fea_stack_v13)

        fea_v14 = self.scpa_v14(fea_stack_v13)
        fea_stack_v14 = torch.cat([input_, fea_v14], dim=1)
        fea_stack_v14 = self.conv1_mid_14(fea_stack_v14)

        fea_v15 = self.scpa_v15(fea_stack_v14)
        fea_stack_v15 = torch.cat([input_, fea_v15], dim=1)
        fea_stack_v15= self.conv1_mid_15(fea_stack_v15)

        fea_v16 = self.scpa_v16(fea_stack_v15)
        fea_stack_v16 = torch.cat([input_, fea_v16], dim=1)
        fea_stack_v16 = self.conv1_mid_16(fea_stack_v16)

        out_2 = self.conv3_end(fea_stack_v16) + input_
        out = self.upsampler(out_2)

        return out