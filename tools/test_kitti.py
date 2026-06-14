# @Time    : 2023/10/8 15:01
# @Author  : zhangchenming
import argparse
import os
import sys
import torch
import datetime
import numpy as np
import glob
from torch.utils.data import DataLoader
from PIL import Image
from easydict import EasyDict
from pathlib import Path

sys.path.insert(0, './')
from stereo.utils import common_utils
from stereo.datasets.dataset_template import DatasetTemplate
from stereo.modeling import build_trainer
from stereo.utils.common_utils import load_params_from_file


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='data loader pin memory')
    parser.add_argument('--dist_mode', action='store_true', default=False, help='pretrained_model')
    parser.add_argument('--run_mode', default='eval', help='pretrained_model')
    parser.add_argument('--output_dir', default=None, help='output dir')
    parser.add_argument('--cfg_file', type=str, default=None, help='cfg file path')

    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--data_cfg_file', type=str, default='cfgs/kitti_eval_test.yaml')

    args = parser.parse_args()
    # import pdb; pdb.set_trace()
    args.output_dir = str(Path(args.pretrained_model).parent.parent) if args.output_dir == "-" else args.output_dir
    args.kitti_result_dir = os.path.join(args.output_dir, str(Path(args.data_cfg_file).stem))
    if not os.path.exists(args.kitti_result_dir):
        os.makedirs(args.kitti_result_dir)
    yaml_files = glob.glob(os.path.join(args.output_dir, '*.yaml'), recursive=False)
    args.cfg_file = yaml_files[0] if args.cfg_file == "-" else args.cfg_file
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)

    return args, cfgs


class KittiTestDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode='testing'):
        super().__init__(data_info, data_cfg, mode)
        self.force_crop = self.data_info.get('FORCE_CROP', False)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path = full_paths[:2]
        left_img = np.array(Image.open(left_img_path).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(right_img_path).convert('RGB'), dtype=np.float32)
        # import pdb; pdb.set_trace()

        if isinstance(self.force_crop, list):
            # crop
            h, w, _ = left_img.shape
            th, bh, tw, bw = self.force_crop
            left_img = left_img[th:h-bh, tw:w-bw]
            right_img = right_img[th:h-bh, tw:w-bw]
        # disp
        sample = {
            'left': left_img,
            'right': right_img,
            'name': left_img_path.split('/')[-1],
        }
        sample = self.transform(sample)
        return sample


@torch.no_grad()
def main():
    args, cfgs = parse_config()
    local_rank = 0
    global_rank = 0
    torch.cuda.set_device(local_rank)

    # logger
    log_file = os.path.join(args.output_dir, 'testkitti_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=local_rank)

    # log args and cfgs
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_configs(cfgs, logger=logger)

    data_yaml_config = common_utils.config_loader(args.data_cfg_file)
    data_cfgs = EasyDict(data_yaml_config)
    logger.info('')
    logger.info('~~~~~~~~~~~~~~~~~~~~ EVAL DATASET INFO ~~~~~~~~~~~~~~~~~~~~')
    common_utils.log_configs(data_cfgs.DATA_CONFIG, logger=logger)
    data_info = data_cfgs.DATA_CONFIG.DATA_INFOS[0]
    kitti_test_dataset = KittiTestDataset(data_info, data_cfgs.DATA_CONFIG)
    kitti_test_loader = DataLoader(dataset=kitti_test_dataset,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=args.workers,
                                   pin_memory=args.pin_memory)
    logger.info('Total samples for eval dataset: %d' % (len(kitti_test_dataset)))

    # model
    model = build_trainer(args, cfgs, local_rank, global_rank, logger, None).model.cuda()

    # load pretrained model
    if args.pretrained_model is not None:
        if not os.path.isfile(args.pretrained_model):
            raise FileNotFoundError
        logger.info('Loading parameters from checkpoint %s' % args.pretrained_model)
        load_params_from_file(model, args.pretrained_model, device='cuda:%d' % local_rank,
                              dist_mode=False, logger=logger, strict=False)

    model.eval()
    for i, data in enumerate(kitti_test_loader):
        for k, v in data.items():
            data[k] = v.to(local_rank) if torch.is_tensor(v) else v

        with torch.cuda.amp.autocast(enabled=cfgs.OPTIMIZATION.AMP):
            model_pred = model(data)

        # import pdb; pdb.set_trace()
        # infer
        disp_pred = model_pred['disp_pred'].squeeze(1)
        # when use dividablepad
        if 'pad' in data:
            pad_top, pad_right, _, _ = data['pad']
            disp_pred = disp_pred[:, pad_top:, :-pad_right] if pad_right != 0 else disp_pred[:, pad_top:, :]
        elif 'original_shape' in data:
            _, h_now, w_now = disp_pred.shape
            h, w = data['original_shape']
            disp_pred = torch.nn.functional.interpolate(disp_pred.unsqueeze(0), (h, w), mode='bilinear', align_corners=False)
            disp_pred = disp_pred.squeeze(0)
            disp_pred = disp_pred * w.item() / w_now

        # save to file
        img = disp_pred.squeeze(0).cpu().numpy()
        img = (img * 256).astype('uint16')
        img = Image.fromarray(img)
        name = data['name'][0]
        name = name.replace('jpg', 'png')
        # import pdb; pdb.set_trace()
        img.save(os.path.join(args.kitti_result_dir, name))

        message = 'Iter:{:>4d}/{}'.format(i, len(kitti_test_loader))
        logger.info(message)

    logger.info(args.kitti_result_dir)


if __name__ == '__main__':
    main()