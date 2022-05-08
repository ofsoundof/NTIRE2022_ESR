
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
   
class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range=1,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ESA(nn.Module):
    def __init__(self, n_feats, conv=nn.Conv2d):
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
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)       
        return x * m



class ResDB(nn.Module):
    def __init__(self, n_feats,n_dist=16):
        super(ResDB, self).__init__()
        self.n_feats=n_feats
        self.n_dist=n_dist

        self.expansion1=nn.Sequential(
            nn.PReLU(n_feats),
            nn.Conv2d(n_feats,n_feats+3*n_dist,1),
        )
        self.expansion2=nn.Sequential(
            nn.PReLU(n_feats+n_dist),
            nn.Conv2d(n_feats+n_dist,n_feats+2*n_dist,1),
        )
        self.expansion3=nn.Sequential(
            nn.PReLU(n_feats+2*n_dist),
            nn.Conv2d(n_feats+2*n_dist,n_feats+n_dist,1),
        )

        self.compression1=nn.Sequential(
            nn.PReLU(n_feats),
            nn.Conv2d(n_feats,n_feats,3,1,1)
        )
        self.compression2=nn.Sequential(
            nn.PReLU(n_feats),
            nn.Conv2d(n_feats,n_feats,3,1,1)
        )
        self.compression3=nn.Sequential(
            nn.PReLU(n_feats),
            nn.Conv2d(n_feats,n_feats,3,1,1)
        )

        self.conv_tail=nn.Sequential(
            nn.PReLU(n_feats+3*n_dist,),
            nn.Conv2d(n_feats+3*n_dist,n_feats,1)
        )

        self.attention=ESA(n_feats)


        

    def forward(self, x):
        input=x
        
        res=self.expansion1(x)
        res,fea_d11,fea_d12,fea_d13=torch.split(res,(self.n_feats,self.n_dist,self.n_dist,self.n_dist),dim=1)
        res=self.compression1(res)
        x=x+res

        res=self.expansion2(torch.cat((x,fea_d11),dim=1))
        res,fea_d21,fea_d22=torch.split(res,(self.n_feats,self.n_dist,self.n_dist),dim=1)
        res=self.compression2(res) 
        x=x+res

        res=self.expansion3(torch.cat((x,fea_d12,fea_d21),dim=1))
        res,fea_d31=torch.split(res,(self.n_feats,self.n_dist),dim=1)
        res=self.compression3(res) 
        x=x+res

        res=self.conv_tail(torch.cat((x,fea_d13,fea_d22,fea_d31),dim=1)) 
        res=self.attention(res)  

        return res+input



class ResDN(nn.Module):
    def __init__(self, upscale_factor=4, in_channels=3, n_feats=48, out_channels=3):
        super(ResDN, self).__init__()

        self.sub_mean = MeanShift()
        self.add_mean = MeanShift(sign=1)

        # head part
        self.fea_conv = nn.Conv2d(in_channels, n_feats, 3, 1, 1)

        # trunk part
        self.body_unit1 = ResDB(n_feats)
        self.body_unit2 = ResDB(n_feats)
        self.body_unit3 = ResDB(n_feats)
        self.body_unit4 = ResDB(n_feats)

        self.T_tdm1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, padding=0, bias=True),
            nn.ReLU())
        self.L_tdm1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, padding=0, bias=True),
            nn.ReLU())

        self.T_tdm2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, padding=0, bias=True),
            nn.ReLU())
        self.L_tdm2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, padding=0, bias=True),
            nn.ReLU())

        self.T_tdm3 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, padding=0, bias=True),
            nn.ReLU())
        self.L_tdm3 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, padding=0, bias=True),
            nn.ReLU())

        # tail part
        modules_tail = [nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True),
                        nn.Conv2d(n_feats, 3 * (upscale_factor ** 2), kernel_size=3, padding=1, bias=True),
                        nn.PixelShuffle(upscale_factor)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.fea_conv(x)
        res1 = self.body_unit1(x)
        res2 = self.body_unit2(res1)
        res3 = self.body_unit3(res2)
        res4 = self.body_unit4(res3)

        T_tdm1 = self.T_tdm1(res4)
        L_tdm1 = self.L_tdm1(res3)
        out_TDM1 = torch.cat((T_tdm1, L_tdm1), 1)

        T_tdm2 = self.T_tdm2(out_TDM1)
        L_tdm2 = self.L_tdm2(res2)
        out_TDM2 = torch.cat((T_tdm2, L_tdm2), 1)

        T_tdm3 = self.T_tdm3(out_TDM2)
        L_tdm3 = self.L_tdm3(res1)
        out_TDM3 = torch.cat((T_tdm3, L_tdm3), 1)

        res = out_TDM3 + x
        out = self.tail(res)
        x = self.add_mean(out)

        return x
