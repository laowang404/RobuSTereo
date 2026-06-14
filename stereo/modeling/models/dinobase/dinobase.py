# @Time: 2024-11-17
# @Author: Wang Yuran
import torch
import torch.nn as nn
import torch.nn.functional as F
from stereo.modeling.common.basic_block_2d import BasicConv2d
from stereo.modeling.cost_volume.cost_volume import build_sub_volume, InterlacedVolume, build_concat_volume, build_gwc_volume
from stereo.modeling.disp_pred.disp_regression import disparity_regression
import torchvision.models as tvm

from .hourglass import Hourglass
from .igev_blocks import Conv2xUp, context_upsample
from .gru_blocks import MultiBasicEncoder, CombinedGeoEncodingVolume, BasicMultiUpdateBlock
from .encoder import CNNandDinov2
import numpy as np

class DinoBase(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        self.max_disp = cfgs.MAX_DISP
        self.num_groups = cfgs.get('NUM_GROUPS', 8)
        self.use_concat_volume = cfgs.get('USE_CONCAT_VOLUME', False)
        self.use_gwc_volume = cfgs.get('USE_GWC_VOLUME', True)
        self.use_sub_volume = cfgs.get('USE_SUB_VOLUME', False)
        self.use_interlaced_volume = cfgs.get('USE_INTERLACED_VOLUME', False)
        self.concat_feature_channel = cfgs.get('CONCAT_CHANNELS', 12)
        self.interlaced_feature_channel = cfgs.get('INTERLACED_CHANNELS', 8)
        self.mono_loss_factor = cfgs.get('MONO_LOSS_FACTOR', 0.5)
        self.mono_loss_stop_epoch = cfgs.get('MONO_LOSS_STOP_EPOCH', 0)
        self.encoder_cfg = cfgs.get('ENCODER_CFG', {})
        self.cnn_kwargs = self.encoder_cfg.get('CNN_KWARGS', {})
        self.CNN_type = self.encoder_cfg.get('CNN_type', "ResNet50")
        # self.use_vgg = self.encoder_cfg.get('use_vgg', False)
        self.amp = self.encoder_cfg.get('AMP', False)
        self.dino = cfgs.get("DINO", True)

        self.SSI_loss = cfgs.get('SSI_LOSS', 0)
        self.SSI_alpha = cfgs.get('SSI_ALPHA', 0.5)
        self.SSI_Grad_loss = cfgs.get('SSI_GRAD_LOSS', 0)
        self.SSI_scales = cfgs.get('SSI_SCALES', 4)

        self.SSI_filter_ratio = cfgs.get('SSI_FILTER_RATIO', 0.0)

        # self.ssi = ScaleAndShiftInvariantLoss(alpha=self.SSI_alpha, scales=self.SSI_scales) if self.SSI_loss else None
        self.ssi = ScaleAndShiftInvariantLoss_filter(alpha=self.SSI_alpha, scales=self.SSI_scales) if self.SSI_loss else None

        self.ssi_grad = ScaleAndShiftInvariantGradientLoss() if self.SSI_Grad_loss else None


        self.n_gru_layers = cfgs.N_GRU_LAYERS
        self.corr_radius = cfgs.CORR_RADIUS
        self.corr_levels = cfgs.CORR_LEVELS
        self.slow_fast_gru = cfgs.SLOW_FAST_GRU
        context_dims = cfgs.HIDDEN_DIMS


        volume_channel = 0
        if self.use_gwc_volume:
            volume_channel += self.num_groups
        if self.use_concat_volume:
            volume_channel += self.concat_feature_channel * 2
        if self.use_sub_volume:
            volume_channel += 1
        if self.use_interlaced_volume:
            volume_channel += self.interlaced_feature_channel

        self.cnet = MultiBasicEncoder(output_dim=[context_dims, context_dims],
                                      norm_fn="batch",
                                      downsample=cfgs.N_DOWNSAMPLE)
        self.update_block = BasicMultiUpdateBlock(n_gru_layers=self.n_gru_layers,
                                                  corr_levels=self.corr_levels,
                                                  corr_radius=self.corr_radius,
                                                  volume_channel=volume_channel,
                                                  hidden_dims=context_dims)
        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(context_dims[i], context_dims[i] * 3, 3, padding=3 // 2) for i in
             range(self.n_gru_layers)])
        self.spx_2_gru = Conv2xUp(32, 32, norm_layer=nn.BatchNorm2d)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )

        self.dino_encoder = CNNandDinov2(cnn_kwargs=self.cnn_kwargs, amp=self.amp, CNN_type=self.CNN_type)

        # backbone
        backbone_channels = [48, 64, 192, 160]
        backbone_channels[0] = backbone_channels[0] + 48

        # get image feature
        self.stem_2 = nn.Sequential(
            BasicConv2d(3, 32,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU,
                        kernel_size=3, stride=2, padding=1),
            BasicConv2d(32, 32,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU,
                        kernel_size=3, stride=1, padding=1)
        )
        self.stem_4 = nn.Sequential(
            BasicConv2d(32, 48,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU,
                        kernel_size=3, stride=2, padding=1),
            BasicConv2d(48, 48,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU,
                        kernel_size=3, stride=1, padding=1),
        )

        # disp refine
        self.spx = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )
        self.spx_2 = Conv2xUp(24, 32, norm_layer=nn.InstanceNorm2d, concat=True)
        self.spx_4 = nn.Sequential(
            BasicConv2d(backbone_channels[0], 24,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU,
                        kernel_size=3, stride=1, padding=1),
            BasicConv2d(24, 24,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU,
                        kernel_size=3, stride=1, padding=1))

        # conv for gwc volume
        self.conv = BasicConv2d(backbone_channels[0], backbone_channels[0],
                                norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU,
                                kernel_size=3, stride=1, padding=1)
        self.desc = nn.Conv2d(backbone_channels[0], backbone_channels[0], kernel_size=1, padding=0, stride=1)

        # aggregation
        self.cost_agg = Hourglass(volume_channel, backbone_channels)

        # cost
        self.classifier = nn.Conv3d(volume_channel, 1, 3, 1, 1, bias=False)

        if self.use_concat_volume:
            self.concat_conv = nn.Sequential(BasicConv2d(backbone_channels[0], 32,
                                                         norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU,
                                                         kernel_size=3, stride=1, padding=1),
                                             nn.Conv2d(32, self.concat_feature_channel,
                                                       kernel_size=1, padding=0, stride=1, bias=False))
        if self.use_interlaced_volume:
            self.build_interlaced_volume = InterlacedVolume(self.interlaced_feature_channel)

        

        self.encoder_linear = nn.ModuleList([
            nn.Conv2d(128, 48, kernel_size=1),
            nn.Conv2d(256, 64, kernel_size=1),
            nn.Conv2d(512, 192, kernel_size=1)
        ]) if self.CNN_type=="VGG19" else nn.ModuleList([
            nn.Conv2d(64, 48, kernel_size=1),
            nn.Conv2d(256, 64, kernel_size=1),
            nn.Conv2d(512, 192, kernel_size=1)
        ])

        if self.dino:
            self.encoder_linear.append(nn.Conv2d(768, 160, kernel_size=1))
        else:
            self.encoder_linear.append(nn.Conv2d(1024, 160, kernel_size=1))


    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        spx_pred = self.spx_gru(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample(disp * 4., spx_pred).unsqueeze(1)
        return up_disp

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()  # [bz, 3, H, W]

        B, C, H, W = image1.shape
        assert H % 32 == 0 and W % 32 == 0, [H, W]

        dino_feature_1 = self.dino_encoder(image1, self.dino)
        dino_feature_2 = self.dino_encoder(image2, self.dino)

        # list: [bz, 48, H/4, W/4] [bz, 64, H/8, W/8] [bz, 192, H/16, W/16] [bz, 160, H/32, W/32]

        features_left_d = [dino_feature_1[2], dino_feature_1[4], dino_feature_1[8], dino_feature_1[16]]
        features_right_d = [dino_feature_2[2], dino_feature_2[4], dino_feature_2[8], dino_feature_2[16]]


        for i in range(4):
            features_left_d[i] = self.encoder_linear[i](features_left_d[i])
            features_right_d[i] = self.encoder_linear[i](features_right_d[i])
        
        if self.dino:
            features_left_d[-1] = F.interpolate(features_left_d[-1], (H//32, W//32), mode='bilinear', align_corners=False)
            features_right_d[-1] = F.interpolate(features_right_d[-1], (H//32, W//32), mode='bilinear', align_corners=False)

        
        features_left = features_left_d
        features_right = features_right_d
        

        stem_2x = self.stem_2(image1)  # [bz, 32, H/2, W/2]
        stem_4x = self.stem_4(stem_2x)  # [bz, 48, H/4, W/4]
        stem_2y = self.stem_2(image2)  # [bz, 32, H/2, W/2]
        stem_4y = self.stem_4(stem_2y)  # [bz, 48, H/4, W/4]
        features_left[0] = torch.cat((features_left[0], stem_4x), 1)  # [bz, 96, H/4, W/4]
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)


        match_left = self.desc(self.conv(features_left[0]))  # [bz, 96, H/4, W/4]
        match_right = self.desc(self.conv(features_right[0]))  # [bz, 96, H/4, W/4]

        all_volume = []
        if self.use_gwc_volume:
            # [bz, num_group, max_disp/4, H/4, W/4]
            gwc_volume = build_gwc_volume(match_left, match_right, self.max_disp // 4, self.num_groups)  # [bz, num_group, max_disp/4, H/4, W/4]
            all_volume.append(gwc_volume)

        if self.use_concat_volume:
            concat_feature_left = self.concat_conv(match_left)
            concat_feature_right = self.concat_conv(match_right)
            concat_volume = build_concat_volume(concat_feature_left, concat_feature_right, self.max_disp // 4)  # [bz, concat_c * 2, max_disp/4, H/4, W/4]
            all_volume.append(concat_volume)

        if self.use_sub_volume:
            sub_volume = build_sub_volume(match_left, match_right, self.max_disp // 4)  # [bz, max_disp/4, H/4, W/4]
            sub_volume = torch.unsqueeze(sub_volume, 1)
            all_volume.append(sub_volume)

        if self.use_interlaced_volume:
            interlaced_volume = self.build_interlaced_volume(match_left, match_right, self.max_disp // 4)
            all_volume.append(interlaced_volume)

        cost_volume = torch.cat(all_volume, dim=1)
        geo_encoding_volume = self.cost_agg(cost_volume, features_left)  # [bz, channel, max_disp/4, H/4, W/4]

        prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)  # [bz, max_disp/4, H/4, W/4]
        init_disp = disparity_regression(prob, self.max_disp // 4)  # [bz, 1, H/4, W/4]

        # gru
        cnet_list = self.cnet(image1, num_layers=self.n_gru_layers)
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in
                    zip(inp_list, self.context_zqr_convs)]
        geo_fn = CombinedGeoEncodingVolume(match_left.float(),
                                           match_right.float(),
                                           geo_encoding_volume.float(),
                                           radius=self.corr_radius,
                                           num_levels=self.corr_levels)
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1, 1, w, 1).repeat(b, h, 1, 1)  # [1, 1, W/4, 1] -> [bz, H/4, W/4, 1]
        disp = init_disp
        disp_preds = []

        iters = self.cfgs.TRAIN_ITERS if self.training else self.cfgs.EVAL_ITERS
        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)  # [bz, (channel+1)*(2r+1)*corr_levels, H/4, W/4]
            if self.n_gru_layers == 3 and self.slow_fast_gru:  # Update low-res ConvGRU
                net_list = self.update_block(net_list, inp_list,
                                             iter16=True,
                                             iter08=False,
                                             iter04=False,
                                             update=False)
            if self.n_gru_layers >= 2 and self.slow_fast_gru:  # Update low-res ConvGRU and mid-res ConvGRU
                net_list = self.update_block(net_list, inp_list,
                                             iter16=self.n_gru_layers == 3,
                                             iter08=True,
                                             iter04=False,
                                             update=False)
            net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp,
                                                                  iter16=self.n_gru_layers == 3,
                                                                  iter08=self.n_gru_layers >= 2)
            disp = disp + delta_disp
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_preds.append(disp_up)

        xspx = self.spx_4(features_left[0])  # [bz, 24, H/4, W/4]
        xspx = self.spx_2(xspx, stem_2x)  # [bz, 24, H/2, W/2]
        spx_pred = self.spx(xspx)  # [bz, 9, H, W]
        spx_pred = F.softmax(spx_pred, 1)  # [bz, 9, H, W]
        init_disp = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)  # [bz, 1, H, W]

        return {'init_disp': init_disp,
                'disp_preds': disp_preds,
                'disp_pred': disp_preds[-1]}

    def mono_loss(self, model_pred, input_data):
        # import pdb; pdb.set_trace()
        current_epoch = input_data['current_epoch']
        disp_gt = input_data["disp"]  # [bz, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, h, w]
        valid = mask.float()  # [bz, h, w]

        disp_gt = disp_gt.unsqueeze(1)  # [bz, 1, h, w]
        mag = torch.sum(disp_gt ** 2, dim=1).sqrt()  # [bz, h, w]
        valid = ((valid >= 0.5) & (mag < self.max_disp)).unsqueeze(1)  # [bz, 1, h, w]
        assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
        assert not torch.isinf(disp_gt[valid.bool()]).any()

        disp_init_pred = model_pred['init_disp']
        disp_loss = 0.0
        if self.ssi is not None:
            mask_ = torch.ones_like(input_data["mono"])
            ssi_loss = self.ssi(disp_init_pred.squeeze(1), input_data["mono"], mask_)
            # ssi_loss = self.ssi(input_data["mono"], disp_init_pred.squeeze(1), mask_)
            disp_loss += ssi_loss * self.SSI_loss
        if self.ssi_grad is not None:
            mask_ = torch.ones_like(input_data["mono"])
            ssi_grad_loss = self.ssi_grad(disp_init_pred.squeeze(1), input_data["mono"], mask_)
            disp_loss += ssi_grad_loss * self.SSI_Grad_loss
            # print(ssi_grad_loss * self.SSI_Grad_loss)
        loss_gamma = 0.9
        disp_preds = model_pred['disp_preds']
        n_predictions = len(disp_preds)
        assert n_predictions >= 1
        for i in range(n_predictions):
            adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
            # i_loss = (disp_preds[i] - disp_gt).abs()
            # assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
            if self.ssi is not None:
                mask_ = torch.ones_like(input_data["mono"])
                ssi_loss = self.ssi(disp_preds[i].squeeze(1), input_data["mono"], mask_)
                # ssi_loss = self.ssi(input_data["mono"], disp_preds[i].squeeze(1), mask_)
                disp_loss += i_weight * ssi_loss * self.SSI_loss
            if self.ssi_grad is not None:
                mask_ = torch.ones_like(input_data["mono"])
                ssi_grad_loss = self.ssi_grad(disp_preds[i].squeeze(1), input_data["mono"], mask_)
                disp_loss += i_weight * ssi_grad_loss * self.SSI_Grad_loss
                # print(i_weight * ssi_grad_loss * self.SSI_Grad_loss)

        tb_info = {'scalar/train/mono_loss_disp': disp_loss.item()}


        return disp_loss, tb_info

    def relative_loss(self, model_pred, input_data):
        pass


    def get_loss(self, model_pred, input_data):
        # import pdb; pdb.set_trace()
        current_epoch = input_data['current_epoch']
        disp_gt = input_data["disp"]  # [bz, h, w]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)  # [bz, h, w]
        valid = mask.float()  # [bz, h, w]

        disp_gt = disp_gt.unsqueeze(1)  # [bz, 1, h, w]
        mag = torch.sum(disp_gt ** 2, dim=1).sqrt()  # [bz, h, w]
        valid = ((valid >= 0.5) & (mag < self.max_disp)).unsqueeze(1)  # [bz, 1, h, w]
        
        assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
        assert not torch.isinf(disp_gt[valid.bool()]).any()

        disp_init_pred = model_pred['init_disp']
        disp_loss = 1.0 * F.smooth_l1_loss(disp_init_pred[valid.bool()], disp_gt[valid.bool()], reduction='mean')

        loss_gamma = 0.9
        disp_preds = model_pred['disp_preds']
        n_predictions = len(disp_preds)
        assert n_predictions >= 1
        for i in range(n_predictions):
            adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
            i_loss = (disp_preds[i] - disp_gt).abs()
            assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
            disp_loss += i_weight * i_loss[valid.bool()].mean()

        tb_info = {'scalar/train/loss_disp': disp_loss.item()}

        # import pdb; pdb.set_trace()

        if current_epoch < self.mono_loss_stop_epoch and self.SSI_loss != 0:
            if torch.sum(valid) < 10:
                disp_loss = 0.0
            mono_loss, mono_tb_info = self.mono_loss(model_pred, input_data)
            disp_loss += mono_loss * self.mono_loss_factor
            tb_info.update(mono_tb_info)
            # import pdb; pdb.set_trace()
            # print(mono_loss)

        return disp_loss, tb_info



def grad_loss_kernel(out, target, cuda=True):
    import pdb; pdb.set_trace()
    out = out.unsqueeze(1)
    target = target.unsqueeze(1)
    x_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    weights_x = torch.from_numpy(x_filter).float().unsqueeze(0).unsqueeze(0)
    weights_y = torch.from_numpy(y_filter).float().unsqueeze(0).unsqueeze(0)

    if cuda:
        weights_x = weights_x.cuda()
        weights_y = weights_y.cuda()

    g1_x = F.conv2d(out,weights_x,padding=1)
    g2_x = F.conv2d(target,weights_x,padding=1)
    g1_y = F.conv2d(out,weights_y,padding=1)
    g2_y = F.conv2d(target,weights_y,padding=1)

    g_1 = torch.sqrt(torch.pow(g1_x, 2) + torch.pow(g1_y, 2))
    g_2 = torch.sqrt(torch.pow(g2_x, 2) + torch.pow(g2_y, 2))

    return F.mse_loss(g_1, g_2)

import torch
import torch.nn as nn


def compute_scale_and_shift(prediction, target, mask):
    # import pdb; pdb.set_trace()
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        # import pdb; pdb.set_trace()

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)

class ScaleAndShiftInvariantLoss_filter(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based', filter_ratio = 0.1):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha
        self.filter_ratio = filter_ratio

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        # import pdb; pdb.set_trace()

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        # scale, shift = scale.detach(), shift.detach()
        shifted_pred = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        error = (shifted_pred - target) ** 2

        # import pdb; pdb.set_trace()
        quantiles = torch.quantile(error.view(error.size(0), -1), 1 - self.filter_ratio, dim=1)

        filter = error < quantiles[:, None, None]
        filter = filter.detach()

        f_scale, f_shift = compute_scale_and_shift(prediction, target, filter)
        f_scale, f_shift = f_scale.detach(), f_shift.detach()
        shifted_pred_filtered = f_scale.view(-1, 1, 1) * prediction + f_shift.view(-1, 1, 1)
        shifted_pred_filtered = torch.clip(shifted_pred_filtered, 0, 200)
        total = self.__data_loss(shifted_pred_filtered, target, mask)

        
        # if self.__alpha > 0:
        #     total += self.__alpha * self.__regularization_loss(shifted_pred_filtered, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)

class ScaleAndShiftInvariantGradientLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.__grad_loss = grad_loss_kernel

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__grad_loss(self.__prediction_ssi, target)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)