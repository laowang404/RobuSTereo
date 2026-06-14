# @Time    : 2023/8/29 09:16
# @Author  : zhangchenming
import torch
import torch.nn as nn
import torch.nn.functional as F

from stereo.modeling.common.basic_block_2d import BasicConv2d, BasicDeconv2d


class Conv2xUp(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, concat=True):
        super(Conv2xUp, self).__init__()
        self.concat = concat
        self.conv1 = BasicDeconv2d(in_channels, out_channels,
                                   norm_layer=norm_layer, act_layer=nn.LeakyReLU,
                                   kernel_size=4, stride=2, padding=1)

        self.conv2 = BasicConv2d(out_channels * 2, out_channels * 2,
                                 norm_layer=norm_layer, act_layer=nn.LeakyReLU,
                                 kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem

        x = self.conv2(x)
        return x


class FeatureAtt(nn.Module):
    def __init__(self, cv_chan, feat_chan):
        super(FeatureAtt, self).__init__()

        self.feat_att = nn.Sequential(
            BasicConv2d(feat_chan, feat_chan // 2,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU,
                        kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan // 2, cv_chan, 1))

    def forward(self, cv, feat):
        feat_att = self.feat_att(feat).unsqueeze(2)
        cv = torch.sigmoid(feat_att) * cv
        return cv


def context_upsample(disp_low, up_weights, scale_factor=4):
    ###
    # disp_low (b,1,h,w)
    # up_weights (b,9,4*h,4*w)
    ###
    b, c, h, w = disp_low.shape

    disp_unfold = F.unfold(disp_low.reshape(b, c, h, w), 3, 1, 1).reshape(b, -1, h, w)
    disp_unfold = F.interpolate(disp_unfold, (h * scale_factor, w * scale_factor), mode='nearest').reshape(b, 9, h * scale_factor, w * scale_factor)

    disp = (disp_unfold * up_weights).sum(1)

    return disp
