import os.path as osp
import glob
import cv2
import numpy as np
import torch
import time
import os
import argparse
import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


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


class InvertedResidualBN(nn.Module):
    def __init__(self, nf=64, expand_ratio=6):
        super(InvertedResidualBN, self).__init__()
        stride = 1
        inp = nf
        oup = nf
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ResidualBlockBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlockBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(nf)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(nf)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return identity + out


class ResidualBlockLeakyBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlockLeakyBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(nf)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(nf)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.lrelu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return identity + out


class NASNetBN(nn.Module):
    """modified SRResNet"""

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4, arch_list=None):
        super(NASNetBN, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

        layers = []
        for index in arch_list[:nb]:
            if index == 0:
                layers.append(InvertedResidualBN(nf, expand_ratio=3))
            elif index == 1:
                layers.append(InvertedResidualBN(nf, expand_ratio=6))
            elif index == 2:
                layers.append(ResidualBlockBN(nf))
            elif index == 3:
                layers.append(ResidualBlockLeakyBN(nf))
        self.recon_trunk = nn.Sequential(*layers)

        # upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last], 0.1)
        initialize_weights(self.upconv2, 0.1)

        self.nb = nb

    def set_arch(self, arch):
        for index, block in zip(arch, self.recon_trunk):
            block.set_arch(index)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)

        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model_nas_base_nf32_nb16.pth', help='Path to model.')
    parser.add_argument('--test_img_folder', type=str, default='test_images', help='Path to test image folder.')
    parser.add_argument('--result_path', type=str, default='results', help='Path to result folder.')

    args = parser.parse_args()

    logging.basicConfig(format='[%(process)d] %(asctime)s: %(message)s', level=logging.INFO)

    model_path = args.model_path
    test_img_folder = args.test_img_folder + '/*'

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.current_device()
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

    # Load model
    model = NASNetBN(in_nc=3, out_nc=3, nf=32, nb=16, upscale=4,
                     arch_list=[3, 1, 2, 3, 3, 0, 1, 2, 0, 0, 0, 0, 2, 3, 3, 1])

    if model_path is not None and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=True)
    else:
        print(model_path, 'checkpoint not exist')
    # print(model)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logging.info('Params number: {}'.format(number_parameters))

    # record PSNR, runtime
    test_results = OrderedDict()
    test_results['runtime'] = []
    idx = 0

    if device == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

    param_size = sum(np.prod(v.size()) for v in model.parameters()) / 1e6
    print('Total parameter size {} M'.format(param_size))
    print('Model path {:s}. \nTesting...'.format(model_path))

    result_path = args.result_path
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    idx = 0
    for path in glob.glob(test_img_folder):
        idx += 1
        base = osp.splitext(osp.basename(path))[0]
        print(idx, base)
        # read images
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        # record time
        now = time.time()
        if device == 'cuda':
            start.record()
        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

        if device == 'cuda':
            end.record()
            torch.cuda.synchronize()
            test_results['runtime'].append(start.elapsed_time(end))  # milliseconds
        else:
            test_results['runtime'].append(time.time() - now)  # milliseconds
        print('time used', time.time() - now)

        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        cv2.imwrite(os.path.join(result_path, '{:s}.png'.format(base)), output)

    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    logging.info('------> Average runtime is : {:.6f} seconds'.format(ave_runtime))

    print('Testing End')
