from .dataset_template import DatasetTemplate
import numpy as np
from .datasets.mscoco_dataset import MSCOCODataset as _MSCOCODataset


class MSCOCODataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        self.height = self.data_info.HEIGHT
        self.width = self.data_info.WIDTH
        self.max_disparity = self.data_info.MAX_DISPARITY
        self.mono_type = self.data_info.MONO_TYPE
        self.dataset = _MSCOCODataset(self.root,
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
