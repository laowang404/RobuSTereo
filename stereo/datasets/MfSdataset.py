# @Time    : 2023/8/26 14:23
# @Author  : zhangchenming
import os
import numpy as np
from PIL import Image
from stereo.datasets.dataset_utils.readpfm import readpfm
from .dataset_template import DatasetTemplate


from .MfSdatasets import ADE20KDataset as _ADE20KDataset
from .MfSdatasets import DIWDataset as _DIWDataset
from .MfSdatasets import DiodeDataset as _DiodeDataset
from .MfSdatasets import MSCOCODataset as _MSCOCODataset
from .MfSdatasets import MapillaryDataset as _MapillaryDataset
from .kitti_dataset import KittiDataset
from torch.utils.data.dataset import ConcatDataset
class MfSDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)

        if mode == 'training':
            ADEdataset = _ADE20KDataset(data_path=data_info.DATA_PATH, filenames='train_exist.txt', feed_height=384, feed_width=768, max_disparity=192, is_train=True, disable_normalisation=False, keep_aspect_ratio=True, disable_sharpening=False, monodepth_model='midas',return_path = True)
            DIWdataset = _DIWDataset(data_path=data_info.DATA_PATH, filenames='train_exist.txt', feed_height=384, feed_width=768, max_disparity=192, is_train=True, disable_normalisation=False, keep_aspect_ratio=True, disable_sharpening=False, monodepth_model='midas',return_path = True)
            Diodedataset = _DiodeDataset(data_path=data_info.DATA_PATH, filenames='train_exist.txt', feed_height=384, feed_width=768, max_disparity=192, is_train=True, disable_normalisation=False, keep_aspect_ratio=True, disable_sharpening=False, monodepth_model='midas',return_path = True)
            MSCOCOdataset = _MSCOCODataset(data_path=data_info.DATA_PATH, filenames='train_exist.txt', feed_height=384, feed_width=768, max_disparity=192, is_train=True, disable_normalisation=False, keep_aspect_ratio=True, disable_sharpening=False, monodepth_model='midas',return_path = True)
            Mapillarydataset = _MapillaryDataset(data_path=data_info.DATA_PATH, filenames='train_exist.txt', feed_height=384, feed_width=768, max_disparity=192, is_train=True, disable_normalisation=False, keep_aspect_ratio=True, disable_sharpening=False, monodepth_model='midas',return_path = True)
            self.All_set = ConcatDataset([ADEdataset, DIWdataset, Diodedataset, MSCOCOdataset, Mapillarydataset])

        else: 
            self.All_set = KittiDataset(data_info.EVAL_DATASET, data_cfg, mode)
    
        self.data_num = self.data_info.get("DATA_NUM", len(self.All_set)) if mode == 'training' else len(self.data_list)

    def __len__(self):
        return self.data_num
    
    # def __len__(self):
    #     return len(self.All_set)
    
    def __getitem__(self, idx):
        return self.my_get_item(len(self.All_set) - (idx + np.random.randint(0, len(self.All_set) // self.data_num) * self.data_num) - 1)

    def my_get_item(self, idx):
        batch = self.All_set[idx]
        sample = {
            'left': np.array(batch['left_image']),  # [H, W, 3]
            'right': np.array(batch['right_image']),  # [H, W, 3]
            'disp': batch['disparity'],  # [H, W]
        }
        # sample["left"] = np.array(sample["left"]).transpose(1,2,0)
        # sample["right"] = np.array(sample["right"]).transpose(1,2,0)
        # sample["disp"] = np.array(sample["disp"])
        sample = self.transform(sample)
        sample['index'] = idx
        sample['name'] = batch['img_path'][0]
        return sample

