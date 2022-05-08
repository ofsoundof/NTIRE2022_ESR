import torch
import torch.nn as nn
import torch.nn.functional as F


lrelu_value = 0.1
act = nn.LeakyReLU(lrelu_value)


class RepConv(nn.Module):
    def __init__(self, n_feats):
        super(RepConv, self).__init__()
        self.rep_conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

    def forward(self, x):
        out = self.rep_conv(x)

        return out


class BasicBlock(nn.Module):
    """ Basic block for building HFAN

    Args:
        n_feats (int): The number of feature maps.

    Diagram:
        --RepConv--LeakyReLU--RepConv--
        
    """

    def __init__(self, n_feats):
        super(BasicBlock, self).__init__()
        self.conv1 = RepConv(n_feats)
        self.conv2 = RepConv(n_feats)

    def forward(self, x):
        res = self.conv1(x)
        res = act(res)
        res = self.conv2(res)

        return res


class HFAB(nn.Module):
    """ High-Frequency Attention Block

    args:
        n_feats (int): The number of input feature maps.
        up_blocks (int): The number of RepConv in this HFAB.
        mid_feats (int): Input feature map numbers of RepConv.

    Diagram:
        --Reduce_dimension--[RepConv]*up_blocks--Expand_dimension--Sigmoid--

    """

    def __init__(self, n_feats, up_blocks, mid_feats):
        super(HFAB, self).__init__()
        self.squeeze = nn.Conv2d(n_feats, mid_feats, 3, 1, 1)
        convs = [BasicBlock(mid_feats) for _ in range(up_blocks)]
        self.convs = nn.Sequential(*convs)
        self.excitate = nn.Conv2d(mid_feats, n_feats, 3, 1, 1)
    
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = act(self.squeeze(x))
        out = act(self.convs(out))
        out = self.excitate(out)
        out = self.sigmoid(out)
        out *= x

        return out


class FMEN(nn.Module):
    """ Fast and Memory-Efficient Network

    Diagram:
        --Conv--Conv-HFAB-[BasicBlock-HFAB]*down_blocks-Conv-+-Upsample--
               |_____________________________________________|

    """

    def __init__(self):
        super(FMEN, self).__init__()

        self.down_blocks = 4

        up_blocks = [2, 1, 1, 1, 1]
        mid_feats = 16
        n_feats = 50
        n_colors = 3
        scale = 4

        # define head module
        self.head = nn.Conv2d(n_colors, n_feats, 3, 1, 1)

        # warm up
        self.warmup = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            HFAB(n_feats, up_blocks[0], mid_feats-4)
        )

        # define body module
        basic_blocks = [BasicBlock(n_feats) for _ in range(self.down_blocks)]
        hfabs = [HFAB(n_feats, up_blocks[i+1], mid_feats) for i in range(self.down_blocks)]

        self.basic_blocks = nn.ModuleList(basic_blocks)
        self.hfabs = nn.ModuleList(hfabs)

        self.lr_conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

        # define tail module
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_colors*(scale**2), 3, 1, 1),
            nn.PixelShuffle(scale)
        )


    def forward(self, x):
        x = self.head(x)

        h = self.warmup(x)
        for i in range(self.down_blocks):
            h = self.basic_blocks[i](h)
            h = self.hfabs[i](h)
        h = self.lr_conv(h)

        h += x
        x = self.tail(h)

        return x
