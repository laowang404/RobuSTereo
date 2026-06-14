# @Time    : 2024/10/8 03:20
# @Author  : zhangchenming
import torch


def for_compatibility(model):
    return model


def build_optimizer(params, base_lr):
    model = params.module

    base_lr = base_lr
    backbone_lr_decay = 0.1
    backbone_weight_decay = 1e-05
    weight_decay_norm = 1e-05
    norm_module_types = (
        torch.nn.BatchNorm2d,
        torch.nn.InstanceNorm2d,
        torch.nn.LayerNorm,
    )
    params = []
    params_norm = []
    param_backbone_relative_position_bias_table_norm = []
    param_relative_position_enc_table_norm = []
    params_backbone = []
    params_offset = []
    memo = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            if f"{module_name}.{module_param_name}".startswith("image_encoder.backbone"):
                if "relative_position_bias_table" in f"{module_param_name}":
                    param_backbone_relative_position_bias_table_norm.append(value)
                else:
                    params_backbone.append(value)
            elif "sampling_offsets" in f"{module_name}":
                params_offset.append(value)
            elif "relative_position_enc_table" in f"{module_param_name}":
                param_relative_position_enc_table_norm.append(value)
            elif isinstance(module, norm_module_types) and weight_decay_norm is not None:
                params_norm.append(value)
            else:
                params.append(value)
    ret = []
    if len(params) > 0:
        ret.append({"params": params, "lr": base_lr})
    if len(params_offset) > 0:
        ret.append({"params": params_offset, "lr": base_lr * 0.1})
    if len(params_norm) > 0:
        ret.append({"params": params_norm, "lr": base_lr, "weight_decay": weight_decay_norm})
    if len(params_backbone) > 0:
        ret.append(
            {"params": params_backbone, "lr": base_lr * backbone_lr_decay, "weight_decay": backbone_weight_decay})
    if len(param_backbone_relative_position_bias_table_norm) > 0:
        ret.append({"params": param_backbone_relative_position_bias_table_norm, "lr": base_lr * backbone_lr_decay,
                    "weight_decay": 0.})
    if len(param_relative_position_enc_table_norm) > 0:
        ret.append({"params": param_relative_position_enc_table_norm, "lr": base_lr, "weight_decay": 0.})
    adamw_args = {
        "params": ret,
        "weight_decay": 1e-05
    }
    return torch.optim.AdamW(**adamw_args)
