# @Time    : 2023/8/28 22:18
# @Author  : zhangchenming
import sys
import os
import argparse
import datetime
import tqdm
from easydict import EasyDict
import wandb

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, './')
from stereo.utils import common_utils
from stereo.modeling import build_trainer
from cfgs.data_basic import DATA_PATH_DICT
import hashlib

def sha256_file(file_path):
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # 读取文件并更新哈希对象
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        return "文件未找到，请检查文件路径。"
    except Exception as e:
        return f"发生错误: {e}"
    
def sha256_string(string):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(string.encode())
    return sha256_hash.hexdigest()

    
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # mode
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default=None, required=True, help='specify the config for training')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    # save path
    parser.add_argument('--save_root_dir', type=str, default='./output', help='save root dir for this experiment')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    # dataloader
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='data loader pin memory')

    args = parser.parse_args()
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)
    exp_id = sha256_file(args.cfg_file)

    dataset_names = [x.DATASET for x in cfgs.DATA_CONFIG.DATA_INFOS]
    unique_dataset_names = dataset_names
    # print(unique_dataset_names)
    # if len(unique_dataset_names) == 1:
    #     exp_dataset_dir = unique_dataset_names[0]
    # else:
    #     exp_dataset_dir = 'MultiDataset'
    args.data_set = unique_dataset_names[0]
    exp_dataset_dir = unique_dataset_names[0]
    args.exp_group_path = os.path.join(exp_dataset_dir, cfgs.MODEL.NAME)
    args.tag = os.path.basename(args.cfg_file)[:-5]

    for each in cfgs.DATA_CONFIG.DATA_INFOS:
        dataset_name = each.DATASET
        # if dataset_name == 'KittiDataset':
        #     dataset_name = 'KittiDataset15' if 'kitti15' in each.DATA_SPLIT.EVALUATING else 'KittiDataset12'
        # each.DATA_PATH = DATA_PATH_DICT[dataset_name]

    args.run_mode = 'train'
    return args, cfgs, exp_id


def main():
    args, cfgs, exp_id = parse_config()
    if args.dist_mode:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        group_rank = int(os.environ["GROUP_RANK"])
    else:
        local_rank = 0
        global_rank = 0
        group_rank = 0

    # env
    torch.cuda.set_device(local_rank)
    if args.fix_random_seed:
        seed = 0 if not args.dist_mode else dist.get_rank()
        common_utils.set_random_seed(seed=seed)

    # savedir
    args.output_dir = str(os.path.join(args.save_root_dir, args.exp_group_path, args.tag, args.extra_tag))
    if os.path.exists(args.output_dir) and args.extra_tag != 'debug' and cfgs.MODEL.CKPT == -2:
        print(cfgs)
        print(args.output_dir)
        raise Exception('There is already an exp with this name')
    if args.dist_mode:
        dist.barrier()
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpt')
    if not os.path.exists(args.ckpt_dir) and local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
    if global_rank == 0:
        common_utils.backup_source_code(os.path.join(args.output_dir, 'code'))
    if args.dist_mode:
        dist.barrier()
    # print(exp_id)
    # exit()
    # logger
    log_file = os.path.join(args.output_dir, 'train_{}_{}.log'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), group_rank))
    logger = common_utils.create_logger(log_file, rank=local_rank)
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard')) if global_rank == 0 else None
    # import pdb; pdb.set_trace()
    wdb_writer = wandb.init(project='OpenStereo', name=args.extra_tag, config=cfgs, dir=args.output_dir, group=cfgs.MODEL.NAME, resume='allow', mode='disabled' if args.extra_tag != 'debug' else 'disabled', id=sha256_string(args.extra_tag)) if global_rank == 0 else None

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_configs(cfgs, logger=logger)
    if global_rank == 0:
        os.system('cp %s %s' % (args.cfg_file, args.output_dir))

    # trainer
    model_trainer = build_trainer(args, cfgs, local_rank, global_rank, logger, tb_writer, wdb_writer)

    tbar = tqdm.trange(model_trainer.last_epoch + 1, model_trainer.total_epochs,
                       desc='epochs', dynamic_ncols=True, disable=(local_rank != 0),
                       bar_format='{l_bar}{bar}{r_bar}\n')
    # train loop
    for current_epoch in tbar:
        model_trainer.train(current_epoch, tbar)
        model_trainer.save_ckpt(current_epoch)
        if current_epoch % cfgs.TRAINER.EVAL_INTERVAL == 0 or current_epoch == model_trainer.total_epochs - 1:
            model_trainer.evaluate(current_epoch)
        # exit()


if __name__ == '__main__':
    main()
