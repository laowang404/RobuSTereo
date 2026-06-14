# @Time    : 2023/8/29 03:01
# @Author  : zhangchenming
import math
import torch
import torch.nn as nn
import timm
from torch.hub import load_state_dict_from_url
from stereo.modeling.common.basic_block_2d import BasicConv2d
from .igev_blocks import Conv2xUp


class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Feature(SubModule):
    def __init__(self):
        super(Feature, self).__init__()
        model = timm.create_model('mobilenetv2_100', pretrained=True, features_only=True)
        layers = [1, 2, 3, 5, 6]
        chans = [16, 24, 32, 96, 160]
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

        self.deconv32_16 = Conv2xUp(chans[4], chans[3], norm_layer=nn.InstanceNorm2d, concat=True)
        self.deconv16_8 = Conv2xUp(chans[3] * 2, chans[2],  norm_layer=nn.InstanceNorm2d,concat=True)
        self.deconv8_4 = Conv2xUp(chans[2] * 2, chans[1],  norm_layer=nn.InstanceNorm2d,concat=True)

        self.conv4 = BasicConv2d(chans[1] * 2, chans[1] * 2,
                                 norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU,
                                 kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        x2 = self.block0(x)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)

        x16 = self.deconv32_16(x32, x16)
        x8 = self.deconv16_8(x16, x8)
        x4 = self.deconv8_4(x8, x4)
        x4 = self.conv4(x4)
        return [x4, x8, x16, x32]
