import os
import numpy as np
from PIL import Image
from stereo.datasets.dataset_utils.readpfm import readpfm
from .dataset_template import DatasetTemplate
from torch.utils.data import DataLoader, ConcatDataset
from .datasets.ade20k_dataset import ADE20KDataset
from .datasets.mscoco_dataset import MSCOCODataset
from .datasets.diode_dataset import DiodeDataset
from .datasets.diw_dataset import DIWDataset
from .datasets.mapillary_dataset import MapillaryDataset
import yaml


class DIODEDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        self.height = self.data_info.HEIGHT
        self.width = self.data_info.WIDTH
        self.max_disparity = self.data_info.MAX_DISPARITY
        self.mono_type = self.data_info.MONO_TYPE
        self.dataset = DiodeDataset(self.root,
                                    self.data_list, self.height,
                                    self.width, is_train=True,
                                    disable_normalisation=True,
                                    max_disparity=self.max_disparity,
                                    keep_aspect_ratio=True,
                                    disable_synthetic_augmentation=True,
                                    disable_sharpening=False,
                                    monodepth_model=self.mono_type,
                                    disable_background=False
                                    )

    def __getitem__(self, idx):
        inputs = self.dataset[idx]
        
        sample = {
            'left': inputs['image'],
            'right': inputs['stereo_image'],
            'disp': inputs['disparity']
        }
        sample["left"] = np.array(sample["left"]).transpose(1,2,0)
        sample["right"] = np.array(sample["right"]).transpose(1,2,0)
        sample["disp"] = np.array(sample["disp"])
        sample = self.transform(sample)
        sample['index'] = idx
        sample['name'] = " ".join(self.data_list[idx])
        return sample
