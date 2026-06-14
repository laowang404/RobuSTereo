# @Time: 2024-11-17
# @Author: Wang Yuran
from typing import Optional, Union
import torch
from torch import device
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import gc
from .transformer import vit_large, vit_base
from .image_encoder import ImageEncoderViT

def get_autocast_params(device=None, enabled=False, dtype=None):
    if device is None:
        autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        #strip :X from device
        autocast_device = str(device).split(":")[0]
    if 'cuda' in str(device):
        out_dtype = dtype
        enabled = True
    else:
        out_dtype = torch.bfloat16
        enabled = False
        # mps is not supported
        autocast_device = "cpu"
    return autocast_device, enabled, out_dtype

class ResNet50(nn.Module):
    def __init__(self, pretrained=False, high_res = False, weights = None, 
                 dilation = None, freeze_bn = True, anti_aliased = False, early_exit = False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        if dilation is None:
            dilation = [False,False,False]
        if anti_aliased:
            pass
        else:
            if weights is not None:
                self.net = tvm.resnet50(weights = weights,replace_stride_with_dilation=dilation)
            else:
                self.net = tvm.resnet50(pretrained=pretrained,replace_stride_with_dilation=dilation)
            
        self.high_res = high_res
        self.freeze_bn = freeze_bn
        self.early_exit = early_exit
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            net = self.net
            feats = {1:x}
            x = net.conv1(x)
            x = net.bn1(x)
            x = net.relu(x)
            feats[2] = x 
            x = net.maxpool(x)
            x = net.layer1(x)
            feats[4] = x 
            x = net.layer2(x)
            feats[8] = x
            if self.early_exit:
                return feats
            x = net.layer3(x)
            feats[16] = x
            x = net.layer4(x)
            feats[32] = x
            return feats

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                pass

class VGG19(nn.Module):
    def __init__(self, pretrained=False, amp = False, amp_dtype = torch.float16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(device_type=autocast_device, enabled=autocast_enabled, dtype = autocast_dtype):
            feats = {}
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x
                    scale = scale*2
                x = layer(x)
            return feats

class CNNandDinov2(nn.Module):
    def __init__(self, cnn_kwargs = None, amp = False, CNN_type = "ResNet50", dinov2_weights = None, amp_dtype = torch.float16):
        super().__init__()
        if dinov2_weights is None:
            dinov2_weights = torch.load("pretrain/imgnet_distilled.pth", map_location="cpu")['model']
        else:
            dinov2_weights = torch.load(dinov2_weights, map_location="cpu")['model']
        
        vit_kwargs = dict(img_size= 518,
            patch_size= 14,
            init_values = 1.0,
            ffn_layer = "mlp",
            block_chunks = 0,
        )
        # dvt_weights2load = dinov2_weights

        dvt_weights2load = {}
        for k, v in dinov2_weights.items():
            dvt_weights2load[k.replace("model.", "")] = v

        dinov2_vitl14 = vit_base(**vit_kwargs).eval()        
        dinov2_vitl14.load_state_dict(dvt_weights2load, strict=False)
        cnn_kwargs = cnn_kwargs if cnn_kwargs is not None else {}
        if CNN_type == "ResNet50":
            self.cnn = ResNet50(**cnn_kwargs)
        elif CNN_type == "VGG19":
            self.cnn = VGG19(**cnn_kwargs)
        elif CNN_type == "ResNet101":
            self.cnn = tvm.resnet101(**cnn_kwargs)
        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            dinov2_vitl14 = dinov2_vitl14.to(self.amp_dtype)
        self.dinov2_vitl14 = [dinov2_vitl14] # ugly hack to not show parameters to DDP
    
    
    def train(self, mode: bool = True):
        return self.cnn.train(mode)
    
    def forward(self, x, upsample = False):
        x = F.avg_pool2d(x, (2, 2))

        B,C,H,W = x.shape

        feature_pyramid = self.cnn(x)
        
        if upsample:
            with torch.no_grad():
                if self.dinov2_vitl14[0].device != x.device:
                    self.dinov2_vitl14[0] = self.dinov2_vitl14[0].to(x.device).to(self.amp_dtype)
                dinov2_features_16 = self.dinov2_vitl14[0].forward_features(x.to(self.amp_dtype))
                features_16 = dinov2_features_16['x_norm_patchtokens'].permute(0,2,1).reshape(B,768,H//14, W//14)
                del dinov2_features_16
                feature_pyramid[16] = features_16
        return feature_pyramid