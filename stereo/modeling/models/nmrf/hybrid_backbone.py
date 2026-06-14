# @Time    : 2024/10/14 20:06
# @Author  : zhangchenming
import torch
import torch.nn as nn
import torch.nn.functional as F
# from stereo.modeling.backbones.dinov2 import DINOv2

import xformers.ops


class MemoryEfficientCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, qkv_bias=False):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=qkv_bias)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op = None

    def forward(self, x, context=None):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class FPNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, high_dim):
        super(FPNLayer, self).__init__()
        self.deconv = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU())

        self.conv = nn.Sequential(nn.Conv2d(out_channels + high_dim, out_channels, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU())

    def forward(self, low, high):
        low = self.deconv(low)
        if low.shape != high.shape:
            low = F.interpolate(low, size=(high.shape[-2], high.shape[-1]), mode='nearest')
        feat = torch.cat([high, low], 1)
        feat = self.conv(feat)
        return feat


# class FCUBlock(nn.Module):
#     def __init__(self, in_channels, scale, out_channels):
#         super(FCUBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, 9 * scale * scale, kernel_size=1)
#         self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         weight = self.conv1(x)
#
#         x = self.conv2(x)
#
#         unfold = F.unfold(x, kernel_size=3, dilation=1, padding=1)
#         unfold = unfold.reshape(b, -1, h, w)  # [bz, 9*out_channels, h, w]
#
#         return x


class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(
            self,
            features,
            activation,
            deconv=False,
            bn=False,
            expand=False,
            align_corners=True,
            size=None
    ):
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand:
            out_features = features // 2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = output + res

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        if size is not None:
            output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)

        output = self.out_conv(output)

        return output


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class Hybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit_backbone = DINOv2(model_name='vits')
        state_dict = torch.hub.load_state_dict_from_url('dinov2_vits14_pretrain.pth', map_location="cpu")
        self.vit_backbone.load_state_dict(state_dict, strict=True)

        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=384, out_channels=128, kernel_size=1) for _ in range(4)
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)])

        self.refines = nn.ModuleList([
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1) for _ in range(2)
        ])

        # self.up1 = FPNLayer(in_channels=768, out_channels=128, high_dim=64)
        # self.up2 = FPNLayer(in_channels=128, out_channels=128, high_dim=48)

        self.output_dim = 128

        self.refinenet4 = _make_fusion_block(128, True)
        self.refinenet3 = _make_fusion_block(128, True)
        self.refinenet2 = _make_fusion_block(128, True)
        self.refinenet1 = _make_fusion_block(128, True)

        # self.out_conv = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU(True)
        # )

        # self.attn1 = MemoryEfficientCrossAttention(query_dim=128, context_dim=128)
        # self.attn2 = MemoryEfficientCrossAttention(query_dim=128, context_dim=128)

    def forward(self, images):
        _, _, orig_h, orig_w = images.shape
        features = self.vit_backbone.get_intermediate_layers(images, [2, 5, 8, 11], reshape=True)  # [bz, 768, H/14, W/14]
        out = []
        for i, x in enumerate(features):
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            if i < 2:
                x = F.interpolate(x, size=(int(orig_h / (4*2**i)), int(orig_w / (4*2**i))), mode='bilinear')
                x = self.refines[i](x)
            out.append(x)

        path_4 = self.refinenet4(out[-1], size=out[-2].shape[2:])  # H/14
        path_3 = self.refinenet3(path_4, out[-2], size=out[-3].shape[2:])  # H/8
        path_2 = self.refinenet3(path_3, out[-3], size=out[-4].shape[2:])  # H/4
        path_1 = self.refinenet1(path_2, out[-4])  # H/4

        # s3 = self.up1(c4, c3)  # [bz, 128, H/8, W/8]  # left right cross attention

        # left_feature, right_feature = torch.chunk(s3, 2, dim=0)
        # B, _, w, h = left_feature.shape
        # x = left_feature.flatten(2).transpose(1, 2)
        # context = right_feature.flatten(2).transpose(1, 2)
        # x = self.attn1(x, context=context) + x
        # left_feature = x.reshape(B, w, h, -1).permute(0, 3, 1, 2).contiguous()
        # s3 = torch.concatenate([left_feature, right_feature], dim=0)

        # s2 = self.up2(s3, c2)  # [bz, 128, H/4, W/4] # left right cross attention

        # left_feature, right_feature = torch.chunk(s2, 2, dim=0)
        # B, _, w, h = left_feature.shape
        # x = left_feature.flatten(2).transpose(1, 2)
        # context = right_feature.flatten(2).transpose(1, 2)
        # x = self.attn2(x, context=context) + x
        # left_feature = x.reshape(B, w, h, -1).permute(0, 3, 1, 2).contiguous()
        # s2 = torch.concatenate([left_feature, right_feature], dim=0)

        return path_1, path_3
