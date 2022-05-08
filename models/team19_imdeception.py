import torch.nn as nn
import torch

class GConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,groups=4,dilation=1):
        super(GConv2d,self).__init__()
        self.conv2d_block=nn.ModuleList()
        self.groups=groups
        channels = in_channels//groups
        o_channels = out_channels//groups
        for _ in range(groups-1):
            self.conv2d_block.append(nn.Conv2d(in_channels=channels,out_channels=o_channels,kernel_size=kernel_size,dilation=dilation,padding=dilation*(kernel_size-1)//2))
        self.conv2d_block.append(nn.Conv2d(in_channels=in_channels-(groups-1)*channels,out_channels=out_channels-(groups-1)*o_channels,kernel_size=kernel_size,dilation=dilation,padding=dilation*(kernel_size-1)//2))

    def forward(self,x):
        return torch.cat([filterg(xg) for filterg,xg in zip(self.conv2d_block,torch.chunk(x,self.groups,1))],dim=1)
class Gblock(nn.Module):
    def __init__(self, in_channels, out_channels, groups, negative_slope=0,dilation=1):
        super(Gblock, self).__init__()
        self.conv0 = GConv2d(in_channels,out_channels,kernel_size=3,groups=groups,dilation=dilation)
        if negative_slope==0:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu(x)
        x = self.conv1(x)
        return x
class BlockSelfAttention2(nn.Module):
    def __init__(self,in_ch,local_block_size=8,ch_down=4,effective_area=32):
        super(BlockSelfAttention2,self).__init__()
        self.Space2Depth_local = nn.PixelUnshuffle(local_block_size)
        self.Depth2Space_local = nn.PixelShuffle(local_block_size)
        self.softmax = nn.Softmax(dim=2)
        self.out_ch = (in_ch//ch_down)*local_block_size*local_block_size
        self.conv_phi_theta_g = nn.Conv2d(in_ch,3*(in_ch//ch_down), kernel_size=1)

        self.conv_out = nn.Conv2d(in_ch  // ch_down, in_ch, kernel_size=1)
        self.block_size = effective_area // local_block_size
        self.Space2Depth_global = nn.PixelUnshuffle(self.block_size)
        self.Depth2Space_global = nn.PixelShuffle(self.block_size)

    def forward(self,x):

        x_n_n_t_g = self.conv_phi_theta_g(x)

        _, _, H8, W8 = x_n_n_t_g.shape
        H8div = (H8 // self.block_size**2 + 1) * self.block_size**2
        W8div = (W8 // self.block_size**2 + 1) * self.block_size**2
        x_n_n_t_g = torch.nn.functional.pad(x_n_n_t_g,(0,W8div-W8,0,H8div-H8))

        x_n_n_t_g = self.Space2Depth_local(x_n_n_t_g)
        N, C3, H, W = x_n_n_t_g.shape
        C = C3//3

        x_n_n_t_g_block = self.Space2Depth_global(x_n_n_t_g).permute(0,2,3,1).contiguous().view(-1,C3,self.block_size,self.block_size)

        Nb, _, Hb, Wb = x_n_n_t_g_block.shape

        x_n_n_t_g_block = x_n_n_t_g_block.permute((0, 2, 3, 1)).contiguous()
        x_n_n_t_g_block = x_n_n_t_g_block.view(Nb,-1,C3)

        x_org_n, x_org_n_t, x_g = torch.split(x_n_n_t_g_block,(self.out_ch,self.out_ch,self.out_ch),2)
        x_org_n_t = torch.transpose(x_org_n_t,2,1)

        SA = torch.bmm(x_org_n,x_org_n_t)
        SA = self.softmax(SA)

        x_org_SA = torch.bmm(SA,x_g)

        x_org_SA = x_org_SA.view(Nb,Hb,Wb,C)
        x_org_SA = x_org_SA.permute((0,3,1,2))

        Hb = H // self.block_size
        Wb = W // self.block_size
        x_org_SA_unblock = self.Depth2Space_global(x_org_SA.contiguous().view(-1,Hb,Wb,C*self.block_size**2).permute(0,3,1,2))

        x_org_SA_unblock = self.Depth2Space_local(x_org_SA_unblock)

        x_org_SA_unblock = x_org_SA_unblock[:,:,0:H8,0:W8]

        x_org_SA_out = self.conv_out(x_org_SA_unblock)
        y = x_org_SA_out+x
        return y
class GIDB(nn.Module):
    def __init__(self,in_ch,out_ch,shal_ch,deep_ch):
        super(GIDB, self).__init__()
        groups = 4
        self.conv0 = Gblock(in_ch,shal_ch+deep_ch,groups)
        self.shal_ch = shal_ch
        self.deep_ch = deep_ch

        self.conv1 = Gblock(deep_ch,shal_ch+deep_ch,groups)

        self.conv2 = Gblock(deep_ch,shal_ch+deep_ch,groups)

        self.conv3_shal = Gblock(deep_ch,shal_ch,groups)

        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)

        self.conv_fuse0 = nn.Conv2d(4*shal_ch+in_ch,out_ch,kernel_size=1,padding=0)

    def forward(self,x):

        x_sa = x

        x_shal0, x_deep0 = torch.split(self.act(self.conv0(x_sa)),(self.shal_ch, self.deep_ch),1)

        x_shal1, x_deep1 = torch.split(self.act(self.conv1(x_deep0)),(self.shal_ch, self.deep_ch),1)

        x_shal2, x_deep2 = torch.split(self.act(self.conv2(x_deep1)),(self.shal_ch, self.deep_ch),1)

        x_shal3 = self.act(self.conv3_shal(x_deep2))

        x_deep = torch.cat([x_shal0,x_shal1,x_shal2, x_shal3,x_sa],1)

        x_deep = self.conv_fuse0(x_deep)

        return x_deep
class IMDeception(nn.Module):
    def __init__(self,in_ch,scale,core=16,out_ch=3):
        super(IMDeception, self).__init__()
        core_ch=core
        self.core_ch=core_ch
        self.feat_conv0 = nn.Conv2d(in_ch,4 * core_ch,kernel_size=3,padding=1)

        self.self_attention1 = BlockSelfAttention2(3 * core_ch, local_block_size=4,ch_down=4)

        self.self_attention2 = BlockSelfAttention2(3 * core_ch, local_block_size=4, ch_down=4)

        self.act = nn.LeakyReLU(negative_slope=0.05,inplace=True)

        self.block1 = GIDB(4 * core_ch, 4 * core_ch, core_ch, 3 * core_ch)

        self.block2 = GIDB(3 * core_ch, 4 * core_ch, core_ch, 3 * core_ch)

        self.block3 = GIDB(3 * core_ch, 4 * core_ch, core_ch, 3 * core_ch)

        self.block4 = GIDB(3 * core_ch, 4 * core_ch, core_ch, 3 * core_ch)

        self.block5 = GIDB(3 * core_ch, 4 * core_ch, core_ch, 3 * core_ch)

        self.block6_shal = GIDB(3 * core_ch, core_ch, core_ch, 2 * core_ch)

        self.conv_fuse0 = nn.Conv2d(6*core_ch,4*core_ch,kernel_size=1,padding=0)
        self.conv_fuse1 = nn.Conv2d(4 * core_ch, 4 * core_ch, kernel_size=3, padding=1)

        self.conv_out = nn.Conv2d(4*core_ch, scale * scale * out_ch, 3, padding=1)
        self.Depth2Space = nn.PixelShuffle(scale)


    def forward(self,x):
        x_out = self.feat_conv0(x)

        x1_, x1 = torch.split(self.block1(x_out),(self.core_ch,3*self.core_ch),1)

        x2_, x2 = torch.split(self.block2(x1),(self.core_ch,3*self.core_ch),1)

        x2 = self.self_attention1(x2)

        x3_, x3 = torch.split(self.block3(x2), (self.core_ch, 3 * self.core_ch), 1)

        x4_, x4 = torch.split(self.block4(x3), (self.core_ch, 3 * self.core_ch), 1)

        x4 = self.self_attention2(x4)

        x5_, x5 = torch.split(self.block5(x4), (self.core_ch, 3 * self.core_ch), 1)

        x6_ = self.block6_shal(x5)

        x_cat = self.act(self.conv_fuse0(torch.cat([x1_,x2_,x3_,x4_,x5_,x6_],1)))
        x_cat = self.act(self.conv_fuse1(x_cat))
        x_cat = x_cat + x_out
        y = self.conv_out(x_cat)
        y = self.Depth2Space(y)
        return y