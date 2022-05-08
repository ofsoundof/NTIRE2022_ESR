import functools
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def make_model(level):
    return ESAN(level = level)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3_1 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_2 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        c1 = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.relu(self.conv3_1(c1))
        c3 = self.relu(self.conv3_2(c3))
        c3 = self.conv3_3(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        c4 = self.conv4(c3 + c1_)
        m = self.sigmoid(c4)
        return x * m

class ResidualBlock_ESA(nn.Module):
    '''
    ---Conv-ReLU-Conv-ESA +-
    '''
    def __init__(self, nf=32):
        super(ResidualBlock_ESA, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.ESA = ESA(nf, nn.Conv2d)
        # initialize_weights([self.conv1, self.conv2, self.ESA], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        out = self.ESA(out)
        return identity + out

class ESAN(nn.Module):
    ''' ELECTRIC'''

    def __init__(self, in_nc=3, out_nc=3, nf = 32, level = 0, upscale=4):
        super(ESAN, self).__init__()
        self.upscale = upscale
        self.level = level
        conv_first = []
        recon_trunk = []
        upconv = []
        self.upconv0 = nn.Conv2d(in_nc, out_nc * 4 * 4, 3, 1, 1, bias=True)
        basic_block = functools.partial(ResidualBlock_ESA, nf=nf)

        power = 16 #18
        for i in range(level):
            recon_trunk.append(make_layer(basic_block, power))
            # power *= 2
            conv_first.append(nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True))
            # upsampling
            upconv.append(nn.Conv2d(nf, out_nc * 4 * 4, 3, 1, 1, bias=True))
        self.pixel_shuffle = nn.PixelShuffle(4)

        self.recon_trunk = nn.ModuleList()
        for i in range(self.level):
            self.recon_trunk.append(
                nn.Sequential(*recon_trunk[i])
            )
        self.conv_first = nn.ModuleList()
        for i in range(self.level):
            self.conv_first.append(
                conv_first[i]
            )
        self.upconv = nn.ModuleList()
        for i in range(self.level):
            self.upconv.append(
                upconv[i]
            )

    def forward(self, x):
        result = self.pixel_shuffle(self.upconv0(x))

        for i in range(self.level):
            fea = self.conv_first[i](x)
            out = self.recon_trunk[i](fea)
            out = self.pixel_shuffle(self.upconv[i](out))
            result += out
        return result

