import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'silu':
        layer = nn.SiLU(inplace=True)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

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

def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


# def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
#     conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
#     pixel_shuffle = nn.PixelShuffle(upscale_factor)
#     return sequential(conv, pixel_shuffle)
def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

#team38_rfdnext
class ESA(nn.Module):
    def __init__(self, n_feats, conv): # 50 conv2
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


class RFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDB, self).__init__()
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        # self.act = activation('lrelu', neg_slope=0.05)
        self.act = activation('silu')
        self.c5 = conv_layer(self.dc * 4, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)
        # self.gct = GCT(in_channels)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))
        # out_fused = self.gct(self.c5(out))

        return out_fused

class LinearBlock(nn.Module):
    def __init__(self, in_filters, out_filters, mid_filters, act_type='prelu',with_idt=False):
        super(LinearBlock, self).__init__()
        self.inp_planes = in_filters
        self.out_planes = out_filters
        self.mid_planes = mid_filters
        self.act_type = act_type

        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        self.conv3x3 = nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=3, padding=1)
        self.conv1x1 = nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=1, padding=0)

        if self.act_type == 'prelu':
            self.act = nn.PReLU()
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'silu':
            self.act  = nn.SiLU(inplace=True)
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        h = self.conv3x3(x)
        h = self.conv1x1(h)
        if self.with_idt:
            h += x
        if self.act_type != 'linear':
            h = self.act(h)
        return h

    def rep_params(self):
        device = self.conv3x3.weight.get_device()
        if device < 0:
            device = None

        kernel_size = 3
        delta = torch.eye(self.inp_planes, device=device).unsqueeze(2).unsqueeze(2)
        delta = F.pad(delta, (kernel_size - 1, kernel_size - 1, kernel_size - 1, kernel_size - 1))
        RK = F.conv2d(delta, self.conv3x3.weight)
        RK = F.conv2d(RK, self.conv1x1.weight)
        RK = torch.flip(RK, [2, 3]).permute(1, 0, 2, 3).contiguous()

        RB = self.conv3x3.bias.view(1, -1, 1, 1)
        RB = F.conv2d(input=RB, weight=self.conv1x1.weight).view(-1, ) + self.conv1x1.bias

        if self.with_idt:
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt

        return RK, RB



class RFDB_LinearBlock(nn.Module):
    def __init__(self, in_channels, mid_filters = 256):
        super(RFDB_LinearBlock, self).__init__()
        self.dc = in_channels // 2
        self.rc = in_channels
        
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        # self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c1_r  = LinearBlock(in_channels, self.rc, mid_filters, act_type='silu', with_idt=True)

        self.c2_d = conv_layer(self.rc, self.dc, 1)
        # self.c2_r = conv_layer(self.rc, self.rc, 3)
        self.c2_r  = LinearBlock(self.rc, self.rc, mid_filters, act_type='silu', with_idt=True)

        self.c3_d = conv_layer(self.rc, self.dc, 1)
        # self.c3_r = conv_layer(self.rc, self.rc, 3)
        self.c3_r  = LinearBlock(self.rc, self.rc, mid_filters, act_type='silu', with_idt=True)

        self.c4  = LinearBlock(self.rc, self.dc, mid_filters, act_type='silu', with_idt=True)
        # self.c4 = conv_layer(self.rc, self.dc, 3)
        self.act = activation('silu')
        self.c5 = conv_layer(self.dc * 4, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)
        # self.gct = GCT(in_channels)


    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        # r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        # r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        # r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))
        # out_fused = self.gct(self.c5(out))

        return out_fused



class Conv3X3(nn.Module):
    def __init__(self, inp_planes, out_planes, act_type='prelu'):
        super(Conv3X3, self).__init__()

        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type

        self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.act  = None

        if self.act_type == 'prelu':
            self.act = nn.PReLU()
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)    
        elif self.act_type == 'silu':
            self.act  = nn.SiLU(inplace=True)
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        y = self.conv3x3(x)
        if self.act_type != 'linear':
            y = self.act(y)
        return y

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

class RFDN(nn.Module):
    def __init__(self, in_nc=3, nf=32, num_modules=4, out_nc=3, upscale=4):
        super(RFDN, self).__init__()

        # self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)
        self.fea_conv = LinearBlock(in_nc, nf, 256, 'linear')

        self.B1 = RFDB_LinearBlock(in_channels=nf)
        self.B2 = RFDB_LinearBlock(in_channels=nf)
        self.B3 = RFDB_LinearBlock(in_channels=nf)
        self.B4 = RFDB_LinearBlock(in_channels=nf)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='prelu') #lrelu

        # self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)


        #### upsampling

    def forward(self, input):
        # input = input/ 255.0
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        # out_lr = self.LR_conv(out_B) + out_fea
        out_lr = out_B + out_fea

        output = self.upsampler(out_lr)


        # output = output.permute(0, 2, 3, 1).contiguous()  #NCHW=>NHWC
        # output  = output * 255
        # output = torch.clamp(output, 0.0, 255.0).to(torch.uint8)
        # inter_res = nn.functional.interpolate(input, scale_factor=4, mode='bicubic', align_corners=False)
        # output = torch.clamp(output+inter_res, 0.0, 255.0)#.to(torch.uint8)

        return output

  

class RFDB_Conv3X3(nn.Module):
    def __init__(self, in_channels, mid_filters = 256):
        super(RFDB_Conv3X3, self).__init__()
        self.dc = in_channels // 2
        self.rc = in_channels
        
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        # self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c1_r  = Conv3X3(in_channels, self.rc, act_type='silu')
        

        self.c2_d = conv_layer(self.rc, self.dc, 1)
        # self.c2_r = conv_layer(self.rc, self.rc, 3)
        self.c2_r  = Conv3X3(self.rc, self.rc, act_type='silu')

        self.c3_d = conv_layer(self.rc, self.dc, 1)
        # self.c3_r = conv_layer(self.rc, self.rc, 3)
        self.c3_r  = Conv3X3(self.rc, self.rc, act_type='silu')

        self.c4  = Conv3X3(self.rc, self.dc, act_type='silu')
        # self.c4 = conv_layer(self.rc, self.dc, 3)
        self.act = activation('silu')
        self.c5 = conv_layer(self.dc * 4, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)
        # self.gct = GCT(in_channels)


    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        # r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        # r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        # r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))
        # out_fused = self.gct(self.c5(out))

        return out_fused
  
class RFDN_Conv3X3(nn.Module):
    def __init__(self, in_nc=3, nf=32, num_modules=4, out_nc=3, upscale=4):
        super(RFDN_Conv3X3, self).__init__()

        # self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)
        self.fea_conv = Conv3X3(in_nc, nf, 'linear')

        self.B1 = RFDB_Conv3X3(in_channels=nf)
        self.B2 = RFDB_Conv3X3(in_channels=nf)
        self.B3 = RFDB_Conv3X3(in_channels=nf)
        self.B4 = RFDB_Conv3X3(in_channels=nf)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='prelu') #lrelu

        # self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)


    def forward(self, input):
        # input = input/ 255.0
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        # out_lr = self.LR_conv(out_B) + out_fea
        out_lr = out_B + out_fea

        output = self.upsampler(out_lr)
      
        # output = output.permute(0, 2, 3, 1).contiguous()  #NCHW=>NHWC
        # output  = output * 255
        # output = torch.clamp(output, 0.0, 255.0).to(torch.uint8)
        # inter_res = nn.functional.interpolate(input, scale_factor=4, mode='bicubic', align_corners=False)
        # output = torch.clamp(output+inter_res, 0.0, 255.0)#.to(torch.uint8)

        return output

  


def demoDIV2K_4x(model):

    model.eval()

    avg_psnr, avg_ssim = 0, 0
    

    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    print('Params number: {}'.format(number_parameters))


    target_dir = '/mnt/ch/dataset/DIV2K/DIV2K_valid_HR/'
    test_dir = '/mnt/ch/dataset/DIV2K/DIV2K_valid_LR_bicubic/X4/'

    all_time = 0.0
    test_count = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # record PSNR, runtime
    test_results = OrderedDict()
    test_results['runtime'] = []

    with torch.no_grad():
        image_filenames = []
        target_filenames = []
        for x in os.listdir(target_dir):
            tmp = x.split('.')[0] + 'x4.png'
            image_filenames.append(test_dir+tmp)
            target_filenames.append(target_dir + x)
        image_filenames.sort()
        target_filenames.sort()

        image_num = len(image_filenames)

        PSNR, SSIM = 0, 0
        for i in range(image_num):
            # print(image_filenames[i])
            # data_in = Image.open(image_filenames[i]).convert('RGB')
            # data_in = modcrop(data_in, 2)

            # data_in = np.asarray(data_in.clone())
            data_in=cv2.imread(image_filenames[i], cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]]
            data_in = torch.from_numpy(data_in.transpose((2, 0, 1))).float().unsqueeze(0).cuda()


            target=cv2.imread(target_filenames[i], cv2.IMREAD_UNCHANGED)

            # cv2.imwrite("1.png", target, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            # target = Image.open(target_filenames[i]).convert('RGB')
            # target = modcrop(target, 2)
            # target = np.asarray(target)[:, :, ::-1]

            # target = torch.from_numpy(target.transpose((2, 0, 1))).float()
            # target = target.permute(1, 2, 0).numpy()[:, :, ::-1]
            # t0 = time.time()
            start.record()

            prediction = model(data_in)
            end.record()
            # t1 = time.time()
            
            test_count += 1
            # all_time+=t1-t0
            torch.cuda.synchronize()
            test_results['runtime'].append(start.elapsed_time(end))  # milliseconds

            prediction = prediction.squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :,::-1]  # [T,H,W,C] -> tensor -> numpy, rgb -> bgr

            # prediction = data_in.resize((data_in.width * 2, data_in.height * 2), Image.BICUBIC)
            # prediction = np.asarray(prediction)[:, :, ::-1]

            # imgname = 'out/'+image_filenames[i].split('/')[-1]
            # cv2.imwrite(imgname, prediction)
            # cv2.imwrite(imgname, prediction, [cv2.IMWRITE_PNG_COMPRESSION, 0])


            # prediction_Y = bgr2ycbcr(prediction)
            # target_Y = bgr2ycbcr(target)

            # prediction_Y = prediction_Y * 255
            # target_Y = target_Y * 255
            pp = calculate_psnr(prediction, target)
            # ss = calculate_ssim(prediction, target)
            PSNR += pp
            # SSIM += ss

        PSNR = PSNR / image_num
        # SSIM = SSIM / image_num
        print(' ==> Average PSNR = {:.4f}, Average SSIM = {:.4f}.'.format(PSNR,PSNR))
        ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
        print('------> Average runtime is : {:.6f} seconds'.format(ave_runtime))
        # print("===> Timer: %.4f sec." % (all_time / test_count))


if __name__ == '__main__':
    import os
    import time
    from collections import OrderedDict
    from FLOPs.profile import profile
    import onnxruntime as rt
    import onnx
    from onnxsim import simplify
    from PIL import Image
    import numpy as np
    import cv2
    from util.utils import *

    width = 256
    height = 256

    model_plain = RFDN_Conv3X3(upscale=4)
    model=RFDN(upscale=4)

    # flops, params = profile(model, input_size=(1, 3, height, width))
    # print('{} x {}, flops: {:.10f} GFLOPs, params: {}'.format(height, width, flops / (1e9), params))

    # from FLOPs.esr_eval import get_model_activation, get_model_flops
    # input_dim = (3, height, width)
    # activations, num_conv2d = get_model_activation(model, input_dim)
    # print('{:>16s} : {:<.4f} [M]'.format('#Activations', activations / 10 ** 6))
    # print('{:>16s} : {:<d}'.format('#Conv2d', num_conv2d))

    # flops = get_model_flops(model, input_dim, False)
    # print('{:>16s} : {:<.4f} [G]'.format('FLOPs', flops / 10 ** 9))

    # num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    # print('{:>16s} : {:<.4f} [M]'.format('#Params', num_parameters / 10 ** 6))



    pretrained = './result/rfdn-0324-kd/epoch_3413.pth'
    checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)
    new_state_dcit = OrderedDict()
    for k, v in checkpoint.items():
        if k.startswith("module."):
            name = k[7:]
            print(name)
            new_state_dcit[name] = v
    model.load_state_dict(new_state_dcit)


#fea_conv
    RK, RB = model.fea_conv.rep_params()
    model_plain.fea_conv.conv3x3.weight.data = RK
    model_plain.fea_conv.conv3x3.bias.data = RB
#B1.c1
    model_plain.B1.c1_d.weight.data = model.B1.c1_d.weight.data
    model_plain.B1.c1_d.bias.data = model.B1.c1_d.bias.data
    RK, RB = model.B1.c1_r.rep_params()
    model_plain.B1.c1_r.conv3x3.weight.data = RK
    model_plain.B1.c1_r.conv3x3.bias.data = RB
#B1.c2
    model_plain.B1.c2_d.weight.data = model.B1.c2_d.weight.data
    model_plain.B1.c2_d.bias.data = model.B1.c2_d.bias.data
    RK, RB = model.B1.c2_r.rep_params()
    model_plain.B1.c2_r.conv3x3.weight.data = RK
    model_plain.B1.c2_r.conv3x3.bias.data = RB
#B1.c3
    model_plain.B1.c3_d.weight.data = model.B1.c3_d.weight.data
    model_plain.B1.c3_d.bias.data = model.B1.c3_d.bias.data
    RK, RB = model.B1.c3_r.rep_params()
    model_plain.B1.c3_r.conv3x3.weight.data = RK
    model_plain.B1.c3_r.conv3x3.bias.data = RB
#B1.c4
    RK, RB = model.B1.c4.rep_params()
    model_plain.B1.c4.conv3x3.weight.data = RK
    model_plain.B1.c4.conv3x3.bias.data = RB
#B1.c5
    model_plain.B1.c5.weight.data = model.B1.c5.weight.data
    model_plain.B1.c5.bias.data = model.B1.c5.bias.data
#B1.esa
    model_plain.B1.esa.conv1.weight.data = model.B1.esa.conv1.weight.data
    model_plain.B1.esa.conv1.bias.data = model.B1.esa.conv1.bias.data
    model_plain.B1.esa.conv_f.weight.data = model.B1.esa.conv_f.weight.data
    model_plain.B1.esa.conv_f.bias.data = model.B1.esa.conv_f.bias.data
    model_plain.B1.esa.conv_max.weight.data = model.B1.esa.conv_max.weight.data
    model_plain.B1.esa.conv_max.bias.data = model.B1.esa.conv_max.bias.data
    model_plain.B1.esa.conv2.weight.data = model.B1.esa.conv2.weight.data
    model_plain.B1.esa.conv2.bias.data = model.B1.esa.conv2.bias.data
    model_plain.B1.esa.conv3.weight.data = model.B1.esa.conv3.weight.data
    model_plain.B1.esa.conv3.bias.data = model.B1.esa.conv3.bias.data
    model_plain.B1.esa.conv3_.weight.data = model.B1.esa.conv3_.weight.data
    model_plain.B1.esa.conv3_.bias.data = model.B1.esa.conv3_.bias.data
    model_plain.B1.esa.conv4.weight.data = model.B1.esa.conv4.weight.data
    model_plain.B1.esa.conv4.bias.data = model.B1.esa.conv4.bias.data

#B2.c1
    model_plain.B2.c1_d.weight.data = model.B2.c1_d.weight.data
    model_plain.B2.c1_d.bias.data = model.B2.c1_d.bias.data
    RK, RB = model.B2.c1_r.rep_params()
    model_plain.B2.c1_r.conv3x3.weight.data = RK
    model_plain.B2.c1_r.conv3x3.bias.data = RB
#B2.c2
    model_plain.B2.c2_d.weight.data = model.B2.c2_d.weight.data
    model_plain.B2.c2_d.bias.data = model.B2.c2_d.bias.data
    RK, RB = model.B2.c2_r.rep_params()
    model_plain.B2.c2_r.conv3x3.weight.data = RK
    model_plain.B2.c2_r.conv3x3.bias.data = RB
#B2.c3
    model_plain.B2.c3_d.weight.data = model.B2.c3_d.weight.data
    model_plain.B2.c3_d.bias.data = model.B2.c3_d.bias.data
    RK, RB = model.B2.c3_r.rep_params()
    model_plain.B2.c3_r.conv3x3.weight.data = RK
    model_plain.B2.c3_r.conv3x3.bias.data = RB
#B2.c4
    RK, RB = model.B2.c4.rep_params()
    model_plain.B2.c4.conv3x3.weight.data = RK
    model_plain.B2.c4.conv3x3.bias.data = RB
#B2.c5
    model_plain.B2.c5.weight.data = model.B2.c5.weight.data
    model_plain.B2.c5.bias.data = model.B2.c5.bias.data
#B2.esa
    model_plain.B2.esa.conv1.weight.data = model.B2.esa.conv1.weight.data
    model_plain.B2.esa.conv1.bias.data = model.B2.esa.conv1.bias.data
    model_plain.B2.esa.conv_f.weight.data = model.B2.esa.conv_f.weight.data
    model_plain.B2.esa.conv_f.bias.data = model.B2.esa.conv_f.bias.data
    model_plain.B2.esa.conv_max.weight.data = model.B2.esa.conv_max.weight.data
    model_plain.B2.esa.conv_max.bias.data = model.B2.esa.conv_max.bias.data
    model_plain.B2.esa.conv2.weight.data = model.B2.esa.conv2.weight.data
    model_plain.B2.esa.conv2.bias.data = model.B2.esa.conv2.bias.data
    model_plain.B2.esa.conv3.weight.data = model.B2.esa.conv3.weight.data
    model_plain.B2.esa.conv3.bias.data = model.B2.esa.conv3.bias.data
    model_plain.B2.esa.conv3_.weight.data = model.B2.esa.conv3_.weight.data
    model_plain.B2.esa.conv3_.bias.data = model.B2.esa.conv3_.bias.data
    model_plain.B2.esa.conv4.weight.data = model.B2.esa.conv4.weight.data
    model_plain.B2.esa.conv4.bias.data = model.B2.esa.conv4.bias.data

#B3.c1
    model_plain.B3.c1_d.weight.data = model.B3.c1_d.weight.data
    model_plain.B3.c1_d.bias.data = model.B3.c1_d.bias.data
    RK, RB = model.B3.c1_r.rep_params()
    model_plain.B3.c1_r.conv3x3.weight.data = RK
    model_plain.B3.c1_r.conv3x3.bias.data = RB
#B3.c2
    model_plain.B3.c2_d.weight.data = model.B3.c2_d.weight.data
    model_plain.B3.c2_d.bias.data = model.B3.c2_d.bias.data
    RK, RB = model.B3.c2_r.rep_params()
    model_plain.B3.c2_r.conv3x3.weight.data = RK
    model_plain.B3.c2_r.conv3x3.bias.data = RB
#B3.c3
    model_plain.B3.c3_d.weight.data = model.B3.c3_d.weight.data
    model_plain.B3.c3_d.bias.data = model.B3.c3_d.bias.data
    RK, RB = model.B3.c3_r.rep_params()
    model_plain.B3.c3_r.conv3x3.weight.data = RK
    model_plain.B3.c3_r.conv3x3.bias.data = RB
#B3.c4
    RK, RB = model.B3.c4.rep_params()
    model_plain.B3.c4.conv3x3.weight.data = RK
    model_plain.B3.c4.conv3x3.bias.data = RB
#B3.c5
    model_plain.B3.c5.weight.data = model.B3.c5.weight.data
    model_plain.B3.c5.bias.data = model.B3.c5.bias.data
#B3.esa
    model_plain.B3.esa.conv1.weight.data = model.B3.esa.conv1.weight.data
    model_plain.B3.esa.conv1.bias.data = model.B3.esa.conv1.bias.data
    model_plain.B3.esa.conv_f.weight.data = model.B3.esa.conv_f.weight.data
    model_plain.B3.esa.conv_f.bias.data = model.B3.esa.conv_f.bias.data
    model_plain.B3.esa.conv_max.weight.data = model.B3.esa.conv_max.weight.data
    model_plain.B3.esa.conv_max.bias.data = model.B3.esa.conv_max.bias.data
    model_plain.B3.esa.conv2.weight.data = model.B3.esa.conv2.weight.data
    model_plain.B3.esa.conv2.bias.data = model.B3.esa.conv2.bias.data
    model_plain.B3.esa.conv3.weight.data = model.B3.esa.conv3.weight.data
    model_plain.B3.esa.conv3.bias.data = model.B3.esa.conv3.bias.data
    model_plain.B3.esa.conv3_.weight.data = model.B3.esa.conv3_.weight.data
    model_plain.B3.esa.conv3_.bias.data = model.B3.esa.conv3_.bias.data
    model_plain.B3.esa.conv4.weight.data = model.B3.esa.conv4.weight.data
    model_plain.B3.esa.conv4.bias.data = model.B3.esa.conv4.bias.data

#B4.c1
    model_plain.B4.c1_d.weight.data = model.B4.c1_d.weight.data
    model_plain.B4.c1_d.bias.data = model.B4.c1_d.bias.data
    RK, RB = model.B4.c1_r.rep_params()
    model_plain.B4.c1_r.conv3x3.weight.data = RK
    model_plain.B4.c1_r.conv3x3.bias.data = RB
#B4.c2
    model_plain.B4.c2_d.weight.data = model.B4.c2_d.weight.data
    model_plain.B4.c2_d.bias.data = model.B4.c2_d.bias.data
    RK, RB = model.B4.c2_r.rep_params()
    model_plain.B4.c2_r.conv3x3.weight.data = RK
    model_plain.B4.c2_r.conv3x3.bias.data = RB
#B4.c3
    model_plain.B4.c3_d.weight.data = model.B4.c3_d.weight.data
    model_plain.B4.c3_d.bias.data = model.B4.c3_d.bias.data
    RK, RB = model.B4.c3_r.rep_params()
    model_plain.B4.c3_r.conv3x3.weight.data = RK
    model_plain.B4.c3_r.conv3x3.bias.data = RB
#B4.c4
    RK, RB = model.B4.c4.rep_params()
    model_plain.B4.c4.conv3x3.weight.data = RK
    model_plain.B4.c4.conv3x3.bias.data = RB
#B4.c5
    model_plain.B4.c5.weight.data = model.B4.c5.weight.data
    model_plain.B4.c5.bias.data = model.B4.c5.bias.data
#B4.esa
    model_plain.B4.esa.conv1.weight.data = model.B4.esa.conv1.weight.data
    model_plain.B4.esa.conv1.bias.data = model.B4.esa.conv1.bias.data
    model_plain.B4.esa.conv_f.weight.data = model.B4.esa.conv_f.weight.data
    model_plain.B4.esa.conv_f.bias.data = model.B4.esa.conv_f.bias.data
    model_plain.B4.esa.conv_max.weight.data = model.B4.esa.conv_max.weight.data
    model_plain.B4.esa.conv_max.bias.data = model.B4.esa.conv_max.bias.data
    model_plain.B4.esa.conv2.weight.data = model.B4.esa.conv2.weight.data
    model_plain.B4.esa.conv2.bias.data = model.B4.esa.conv2.bias.data
    model_plain.B4.esa.conv3.weight.data = model.B4.esa.conv3.weight.data
    model_plain.B4.esa.conv3.bias.data = model.B4.esa.conv3.bias.data
    model_plain.B4.esa.conv3_.weight.data = model.B4.esa.conv3_.weight.data
    model_plain.B4.esa.conv3_.bias.data = model.B4.esa.conv3_.bias.data
    model_plain.B4.esa.conv4.weight.data = model.B4.esa.conv4.weight.data
    model_plain.B4.esa.conv4.bias.data = model.B4.esa.conv4.bias.data

#upsample

    model_plain.c[0].weight.data = model.c[0].weight.data
    model_plain.c[0].bias.data = model.c[0].bias.data   

    model_plain.c[1].weight.data = model.c[1].weight.data

    model_plain.upsampler[0].weight.data = model.upsampler[0].weight.data
    model_plain.upsampler[0].bias.data = model.upsampler[0].bias.data



    demoDIV2K_4x(model_plain.cuda())
    demoDIV2K_4x(model.cuda())


    def save_checkpoint(epoch):
        model_out_path =  "model_plain_epoch_{}.pth".format(epoch)
        torch.save(model_plain.state_dict(), model_out_path)
        print("===> Checkpoint saved to {}".format(model_out_path))
    save_checkpoint(100)

    print("pass")

    