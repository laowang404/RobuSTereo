# @Time    : 2024/1/20 03:13
# @Author  : zhangchenming
import os
import time
import glob
import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision.utils import save_image
import numpy as np
from thop import profile
import cv2
from PIL import Image


from functools import partial
from stereo.datasets import build_dataloader
from stereo.utils import common_utils
from stereo.utils.common_utils import color_map_tensorboard, write_tensorboard, write_wandb
from stereo.utils.warmup import LinearWarmup
from stereo.utils.clip_grad import ClipGrad
from stereo.utils.lamb import Lamb
from stereo.evaluation.metric_per_image import epe_metric, d1_metric, threshold_metric


class TrainerTemplate:
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer, model, **kwargs):
        self.args = args
        self.cfgs = cfgs
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.logger = logger
        self.tb_writer = tb_writer
        self.wdb_write = kwargs.get('wdb_writer', None)
        # import pdb; pdb.set_trace()

        self.model = self.build_model(model)

        # input1 = torch.randn(4, 3, 224, 224).cuda().float()
        # with torch.cuda.amp.autocast(enabled=self.cfgs.OPTIMIZATION.AMP):
        #     flops, params = profile(self.model, inputs=({"left":input1, "right":input1},))
        #     self.logger.info('FLOPs = ' + str(flops/1000**3) + 'G')
        #     self.logger.info('Params = ' + str(params/1000**2) + 'M')

                            
            
        # Total_params = 0
        # Trainable_params = 0
        # NonTrainable_params = 0

        # for param in self.model.parameters():
        #     mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        #     Total_params += mulValue  # 总参数量
        #     if param.requires_grad:
        #         Trainable_params += mulValue  # 可训练参数量
        #     else:
        #         NonTrainable_params += mulValue  # 非可训练参数量

        # self.logger.info(f'Total params: {Total_params / 1e6}M')
        # self.logger.info(f'Trainable params: {Trainable_params/ 1e6}M')
        # self.logger.info(f'Non-trainable params: {NonTrainable_params/ 1e6}M')


        if self.args.run_mode in ['train', 'eval']:
            self.eval_set, self.eval_loader, self.eval_sampler = self.build_eval_loader()

        if self.args.run_mode == 'train':
            self.train_set, self.train_loader, self.train_sampler = self.build_train_loader()

            self.total_epochs = cfgs.OPTIMIZATION.NUM_EPOCHS
            self.last_epoch = -1

            self.optimizer, self.scheduler = self.build_optimizer_and_scheduler()
            self.scaler = torch.cuda.amp.GradScaler(enabled=cfgs.OPTIMIZATION.AMP)

            try:
                if self.cfgs.MODEL.CKPT >= -1:
                    self.resume_ckpt()
            except Exception as e:
                self.logger.error('Resume checkpoint failed, please check the checkpoint file.')
                self.logger.error(e)
                pass

            self.warmup_scheduler = self.build_warmup()
            self.clip_gard = self.build_clip_grad()

    def build_train_loader(self):
        train_set, train_loader, train_sampler = build_dataloader(
            data_cfg=self.cfgs.DATA_CONFIG,
            batch_size=self.cfgs.OPTIMIZATION.BATCH_SIZE_PER_GPU,
            is_dist=self.args.dist_mode,
            workers=self.args.workers,
            pin_memory=self.args.pin_memory,
            mode='training')
        self.logger.info('Total samples for train dataset: %d' % (len(train_set)))
        return train_set, train_loader, train_sampler

    def build_eval_loader(self):
        eval_set, eval_loader, eval_sampler = build_dataloader(
            data_cfg=self.cfgs.DATA_CONFIG,
            batch_size=self.cfgs.EVALUATOR.BATCH_SIZE_PER_GPU,
            is_dist=self.args.dist_mode,
            workers=self.args.workers,
            pin_memory=self.args.pin_memory,
            mode='evaluating')
        self.logger.info('Total samples for eval dataset: %d' % (len(eval_set)))
        return eval_set, eval_loader, eval_sampler

    def build_model(self, model):
        if self.cfgs.OPTIMIZATION.get('FREEZE_BN', False):
            model = common_utils.freeze_bn(model)
            self.logger.info('Freeze the batch normalization layers')

        if self.cfgs.OPTIMIZATION.SYNC_BN and self.args.dist_mode:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.logger.info('Convert batch norm to sync batch norm')
        model = model.to(self.local_rank)

        if self.args.dist_mode:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank], output_device=self.local_rank,
                find_unused_parameters=self.cfgs.MODEL.FIND_UNUSED_PARAMETERS)

        # load pretrained model
        if self.cfgs.MODEL.PRETRAINED_MODEL:
            self.logger.info('Loading parameters from checkpoint %s' % self.cfgs.MODEL.PRETRAINED_MODEL)
            if not os.path.isfile(self.cfgs.MODEL.PRETRAINED_MODEL):
                raise FileNotFoundError
            common_utils.load_params_from_file(
                model, self.cfgs.MODEL.PRETRAINED_MODEL, device='cuda:%d' % self.local_rank,
                dist_mode=self.args.dist_mode, logger=self.logger, strict=False)
        return model

    def build_optimizer_and_scheduler(self):
        if self.cfgs.OPTIMIZATION.OPTIMIZER.NAME == 'Lamb':
            optimizer_cls = Lamb
        else:
            optimizer_cls = getattr(torch.optim, self.cfgs.OPTIMIZATION.OPTIMIZER.NAME)
        valid_arg = common_utils.get_valid_args(optimizer_cls, self.cfgs.OPTIMIZATION.OPTIMIZER, ['name'])
        optimizer = optimizer_cls(params=[p for p in self.model.parameters() if p.requires_grad], **valid_arg)

        self.cfgs.OPTIMIZATION.SCHEDULER.TOTAL_STEPS = self.total_epochs * len(self.train_loader)
        scheduler_cls = getattr(torch.optim.lr_scheduler, self.cfgs.OPTIMIZATION.SCHEDULER.NAME)
        valid_arg = common_utils.get_valid_args(scheduler_cls, self.cfgs.OPTIMIZATION.SCHEDULER, ['name', 'on_epoch'])
        scheduler = scheduler_cls(optimizer, **valid_arg)

        return optimizer, scheduler

    def resume_ckpt(self):
        self.logger.info('Resume from ckpt:%d' % self.cfgs.MODEL.CKPT)
        if self.cfgs.MODEL.CKPT >= 0:
            ckpt_path = str(os.path.join(self.args.ckpt_dir, 'checkpoint_epoch_%d.pth' % self.cfgs.MODEL.CKPT))
        else:
            ckpt_path = str(os.path.join(self.args.ckpt_dir, 'checkpoint_latest.pth')) 
        checkpoint = torch.load(ckpt_path, map_location='cuda:%d' % self.local_rank)
        self.last_epoch = checkpoint['epoch']
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        if self.args.dist_mode:
            self.model.module.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint['model_state'])

    def build_warmup(self):
        last_step = (self.last_epoch + 1) * len(self.train_loader) - 1
        if 'WARMUP' in self.cfgs.OPTIMIZATION.SCHEDULER:
            warmup_steps = self.cfgs.OPTIMIZATION.SCHEDULER.WARMUP.get('WARM_STEPS', 1)
            warmup_scheduler = LinearWarmup(
                self.optimizer,
                warmup_period=warmup_steps,
                last_step=last_step)
        else:
            warmup_scheduler = LinearWarmup(
                self.optimizer,
                warmup_period=1,
                last_step=last_step)

        return warmup_scheduler

    def build_clip_grad(self):
        clip_gard = None
        if 'CLIP_GRAD' in self.cfgs.OPTIMIZATION:
            clip_type = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('TYPE', None)
            clip_value = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('CLIP_VALUE', 0.1)
            max_norm = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('MAX_NORM', 35)
            norm_type = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('NORM_TYPE', 2)
            clip_gard = ClipGrad(clip_type, clip_value, max_norm, norm_type)
        return clip_gard

    def train(self, current_epoch, tbar):
        self.model.train()
        if self.cfgs.OPTIMIZATION.get('FREEZE_BN', False):
            self.model = common_utils.freeze_bn(self.model)
        if self.args.dist_mode:
            self.train_sampler.set_epoch(current_epoch)
        self.train_one_epoch(current_epoch=current_epoch, tbar=tbar)
        if self.args.dist_mode:
            dist.barrier()
        if self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH:
            self.scheduler.step()
            self.warmup_scheduler.lrs = [group['lr'] for group in self.optimizer.param_groups]

    def evaluate(self, current_epoch):
        self.model.eval()
        self.eval_one_epoch(current_epoch=current_epoch)
        if self.args.dist_mode:
            dist.barrier()

    def save_ckpt(self, current_epoch, step = -1):
        if step == -1:
            if (current_epoch % self.cfgs.TRAINER.CKPT_SAVE_INTERVAL == 0 or current_epoch == self.total_epochs - 1) and self.global_rank == 0:
                ckpt_list = glob.glob(os.path.join(self.args.ckpt_dir, 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)
                if len(ckpt_list) >= self.cfgs.TRAINER.MAX_CKPT_SAVE_NUM:
                    for cur_file_idx in range(0, len(ckpt_list) - self.cfgs.TRAINER.MAX_CKPT_SAVE_NUM + 1):
                        os.remove(ckpt_list[cur_file_idx])
                
                ckpt_name = os.path.join(self.args.ckpt_dir, 'checkpoint_epoch_%d.pth' % current_epoch)
                common_utils.save_checkpoint(self.model, self.optimizer, self.scheduler, self.scaler,
                                            self.args.dist_mode, current_epoch, filename=ckpt_name)
            if self.global_rank == 0:
                ckpt_name_latest = os.path.join(self.args.ckpt_dir, 'checkpoint_latest.pth')
                common_utils.save_checkpoint(self.model, self.optimizer, self.scheduler, self.scaler,
                                            self.args.dist_mode, current_epoch, filename=ckpt_name_latest)
        else:
            if self.global_rank == 0:
                ckpt_name = os.path.join(self.args.ckpt_dir, f'checkpoint_epoch_{current_epoch}_{step}.pth')
                common_utils.save_checkpoint(self.model, self.optimizer, self.scheduler, self.scaler,
                                                self.args.dist_mode, current_epoch, filename=ckpt_name)
        if self.args.dist_mode:
            dist.barrier()

    def train_one_epoch(self, current_epoch, tbar):
        start_epoch = self.last_epoch + 1
        logger_iter_interval = self.cfgs.TRAINER.LOGGER_ITER_INTERVAL
        total_loss = 0.0
        loss_func = self.model.module.get_loss if self.args.dist_mode else self.model.get_loss

        train_loader_iter = iter(self.train_loader)
        for i in range(0, len(self.train_loader)):
            # import pdb; pdb.set_trace()
            self.optimizer.zero_grad()
            lr = self.optimizer.param_groups[0]['lr']

            start_timer = time.time()
            data = next(train_loader_iter)
            for k, v in data.items():
                data[k] = v.to(self.local_rank) if torch.is_tensor(v) else v
            data_timer = time.time()

            with torch.cuda.amp.autocast(enabled=self.cfgs.OPTIMIZATION.AMP):
                # import pdb; pdb.set_trace()
                model_pred = self.model(data)
                infer_timer = time.time()
                data['current_epoch'] = current_epoch
                loss, tb_info = loss_func(model_pred, data)

            # import pdb; pdb.set_trace()
            # for name, parms in self.model.named_parameters():	
            #     print('-->name:', name)
            #     print('-->para:', parms)
            #     print('-->grad_requirs:',parms.requires_grad)
            #     print('-->grad_value:',parms.grad)
            #     print("===")    
            #     break        
            # pdb.set_trace()

            total_iter = current_epoch * len(self.train_loader) + i

            backup_interval = 2000
            # pdb.set_trace()
            if False and loss.isnan():    
                exit()
                # loss.backward()
                # loss = torch.zeros_like(loss)
                # nearest_iter = total_iter // backup_interval * backup_interval
                # self.logger.error(f'Loss is NaN, reload ckpt from epoch {current_epoch} iteration {nearest_iter}.')
                # ckpt_name = os.path.join(self.args.ckpt_dir, f'checkpoint_epoch_{current_epoch}_{nearest_iter}.pth')
                # # ckpt_name =  os.path.join(self.args.ckpt_dir, f'error.pth')
                # self.logger.error(f'Load ckpt from {ckpt_name}')
                # torch.cuda.empty_cache()
                # checkpoint = torch.load(ckpt_name, map_location='cuda:%d' % self.local_rank)

                # # self.scheduler.load_state_dict(checkpoint['scheduler_state'])
                # self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                # # self.scaler.load_state_dict(checkpoint['scaler_state'])
                
                # # 加载模型的状态
                # if self.args.dist_mode:
                #     self.model.module.load_state_dict(checkpoint['model_state'])
                # else:
                #     self.model.load_state_dict(checkpoint['model_state'])
                # torch.cuda.empty_cache()
                # continue
            else:
                # if total_iter % backup_interval == 0:
                #     self.save_ckpt(current_epoch, total_iter)
                #     self.logger.info(f'Save checkpoint at epoch {current_epoch} iteration {total_iter}.')

                # import pdb; pdb.set_trace()
                # 缩放损失并执行反向传播
                scaled_loss = self.scaler.scale(loss)
                scaled_loss.backward()
                
                # 取消梯度缩放，准备进行梯度剪裁和优化
                self.scaler.unscale_(self.optimizer)
                
                # 如果设置了梯度剪裁，执行梯度剪裁
                if self.clip_gard is not None:
                    self.clip_gard(self.model)
                
                # 使用优化器更新参数，确保只在 unscaled 梯度上操作
                self.scaler.step(self.optimizer)
                
                # 清空优化器的梯度以准备下一轮
                self.optimizer.zero_grad()
                
                # 更新 scaler 的缩放因子，为下一轮训练调整精度范围
                self.scaler.update()


            # warmup_scheduler period>1 和 batch_scheduler 不要同时使用
            with self.warmup_scheduler.dampening():
                if not self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH:
                    self.scheduler.step()

            total_loss += loss.item()
            total_iter = current_epoch * len(self.train_loader) + i
            trained_time_past_all = tbar.format_dict['elapsed']
            single_iter_second = trained_time_past_all / (total_iter + 1 - start_epoch * len(self.train_loader))
            remaining_second_all = single_iter_second * (self.total_epochs * len(self.train_loader) - total_iter - 1)
            if total_iter % logger_iter_interval == 0:
                message = ('Training Epoch:{:>2d}/{} Iter:{:>4d}/{} '
                           'Loss:{:#.6g}({:#.6g}) LR:{:.4e} '
                           'DataTime:{:.2f} InferTime:{:.2f}ms '
                           'Time cost: {}/{}'
                           ).format(current_epoch, self.total_epochs, i, len(self.train_loader),
                                    loss.item(), total_loss / (i + 1), lr,
                                    data_timer - start_timer, (infer_timer - data_timer) * 1000,
                                    tbar.format_interval(trained_time_past_all),
                                    tbar.format_interval(remaining_second_all))
                self.logger.info(message)

            if self.cfgs.TRAINER.TRAIN_VISUALIZATION:
                # data_left = data['left'][0]
                # mean_data = data_left.mean(dim=1, keepdim=True)
                # std_data = data_left.std(dim=1, keepdim=True)
                # if mean
                tb_info['image/train/image'] = torch.cat([data['left'][0], data['right'][0]], dim=1) / 256
                tb_info['image/train/disp'] = color_map_tensorboard(data['disp'][0], model_pred['disp_pred'].squeeze(1)[0])

            tb_info.update({'scalar/train/lr': lr})
            if total_iter % logger_iter_interval == 0 and self.local_rank == 0 and self.tb_writer is not None:
                write_tensorboard(self.tb_writer, tb_info, total_iter)
                write_wandb(self.wdb_write, tb_info, total_iter)

    @torch.no_grad()
    def eval_one_epoch(self, current_epoch):

        metric_func_dict = {
            'epe': epe_metric,
            'd1_all': d1_metric,
            'thres_1': partial(threshold_metric, threshold=1),
            'thres_2': partial(threshold_metric, threshold=2),
            'thres_3': partial(threshold_metric, threshold=3),
        }

        evaluator_cfgs = self.cfgs.EVALUATOR
        local_rank = self.local_rank

        epoch_metrics = {}
        for k in evaluator_cfgs.METRIC:
            epoch_metrics[k] = {'indexes': [], 'values': []}

        for i, data in enumerate(self.eval_loader):
            for k, v in data.items():
                data[k] = v.to(local_rank) if torch.is_tensor(v) else v

            # import pdb; pdb.set_trace()
            
            with torch.cuda.amp.autocast(enabled=self.cfgs.OPTIMIZATION.AMP):
                infer_start = time.time()
                model_pred = self.model(data)
                infer_time = time.time() - infer_start
            
            # import pdb; pdb.set_trace()
            # debug for new script
            # img_name = os.path.basename(data['name'][0])
            # disp_path = os.path.join("/data1/wangyuran/DiffStereo/DiffStereo/OpenStereo/dino-output", img_name)
            # disp_path = os.path.join("/data1/wangyuran/DiffStereo/DiffStereo/OpenStereo/dino-output-GT", img_name)
            
            # disp = disp_pred.data.cpu().numpy().squeeze().squeeze()
            # vis = (data["disp"].data.cpu().numpy().squeeze() * 256).astype(np.uint16)
            # img = Image.fromarray(vis)
            # img.save(disp_path)
            # continue
            # disp_pred = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256
            # disp_pred = torch.from_numpy(disp_pred).unsqueeze(0).unsqueeze(0).cuda().float()

            disp_pred = model_pred['disp_pred']
            disp_gt = data["disp"]
            mask = (disp_gt < evaluator_cfgs.MAX_DISP) & (disp_gt > 0)
            if 'occ_mask' in data and evaluator_cfgs.get('APPLY_OCC_MASK', False):
                mask = mask & ~data['occ_mask'].to(torch.bool)

            if 'original_shape' in data:
                # import pdb; pdb.set_trace()
                _,_, h_now, w_now = disp_pred.shape
                h, w = data['original_shape']
                disp_pred = torch.nn.functional.interpolate(disp_pred, (h, w), mode='bilinear', align_corners=False)
                disp_pred = disp_pred * w.item() / w_now

                pad_h = h_now - h
                pad_w = w_now - w
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left

                disp_gt = disp_gt[:, pad_top:-pad_bottom, pad_left:-pad_right]
                mask = (disp_gt < evaluator_cfgs.MAX_DISP) & (disp_gt > 0)


            # import pdb; pdb.set_trace()
            for m in evaluator_cfgs.METRIC:
                if m not in metric_func_dict:
                    raise ValueError("Unknown metric: {}".format(m))
                metric_func = metric_func_dict[m]
                res = metric_func(disp_pred.squeeze(1), disp_gt, mask)
                epoch_metrics[m]['indexes'].extend(data['index'].tolist())
                epoch_metrics[m]['values'].extend(res.tolist())

            if i % self.cfgs.TRAINER.LOGGER_ITER_INTERVAL == 0:
                message = ('Evaluating Epoch:{:>2d} Iter:{:>4d}/{} InferTime: {:.2f}ms'
                           ).format(current_epoch, i, len(self.eval_loader), infer_time * 1000)
                self.logger.info(message)

                if False and self.cfgs.TRAINER.EVAL_VISUALIZATION and self.tb_writer is not None:
                    tb_info = {
                        'image/eval/image': torch.cat([data['left'][0], data['right'][0]], dim=1) / 256,
                        'image/eval/disp': color_map_tensorboard(data['disp'][0], model_pred['disp_pred'].squeeze(1)[0])
                    }
                    write_tensorboard(self.tb_writer, tb_info, current_epoch * len(self.eval_loader) + i)
                    write_wandb(self.wdb_write, tb_info, current_epoch * len(self.eval_loader) + i)

        # gather from all gpus
        if self.args.dist_mode:
            dist.barrier()
            self.logger.info("Start reduce metrics.")
            for k in epoch_metrics.keys():
                indexes = torch.tensor(epoch_metrics[k]["indexes"]).to(local_rank)
                values = torch.tensor(epoch_metrics[k]["values"]).to(local_rank)
                gathered_indexes = [torch.zeros_like(indexes) for _ in range(dist.get_world_size())]
                gathered_values = [torch.zeros_like(values) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_indexes, indexes)
                dist.all_gather(gathered_values, values)
                unique_dict = {}
                for key, value in zip(torch.cat(gathered_indexes, dim=0).tolist(),
                                      torch.cat(gathered_values, dim=0).tolist()):
                    if key not in unique_dict:
                        unique_dict[key] = value
                epoch_metrics[k]["indexes"] = list(unique_dict.keys())
                epoch_metrics[k]["values"] = list(unique_dict.values())

        # import pdb; pdb.set_trace()
        results = {}
        for k in epoch_metrics.keys():
            results[k] = torch.tensor(epoch_metrics[k]["values"]).mean()

        if local_rank == 0 and self.tb_writer is not None:
            tb_info = {}
            for k, v in results.items():
                tb_info[f'scalar/val/{k}'] = v.item()

            write_tensorboard(self.tb_writer, tb_info, current_epoch)
            write_wandb(self.wdb_write, tb_info, current_epoch)

        self.logger.info(f"Epoch {current_epoch} metrics: {results}")
