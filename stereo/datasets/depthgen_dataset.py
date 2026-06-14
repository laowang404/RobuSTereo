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

dataset_lookup = {'mscoco': MSCOCODataset,
                  'ADE20K': ADE20KDataset,
                #   'sceneflow': SceneFlowDataset,
                  'diw': DIWDataset,
                  'diode': DiodeDataset,
                  'mapillary': MapillaryDataset,
                #   'kitti2015': KITTIStereoDataset
                  }


def load_config(config):
    with open(config, 'r') as fh:
        config = yaml.safe_load(fh)
    return config

def readlines(filename):
    """ read lines of a text file """
    with open(filename, 'r') as file_handler:
        lines = file_handler.read().splitlines()
    return lines

class DepthGenDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        path_info = load_config(self.opt.config_path)
        config_training_datasets = []

        self.split_file = self.data_info.DATA_SPLIT[self.mode.upper()]

        train_datasets = []
        val_datasets = []

        is_train = mode == "training"

        for dataset_type in config_training_datasets:
            dataset_path = path_info[dataset_type]
            train_filenames = readlines(self.split_file)

            val_filenames = 'val_files_all.txt' if dataset_type != 'sceneflow' else 'test_files.txt'
            val_filenames = readlines(os.path.join('splits', dataset_type, val_filenames))
            dataset_class = dataset_lookup[dataset_type]

            # subsample data optionally
            if self.opt.data_sampling != 1.0:
                sampling = self.opt.data_sampling
                assert sampling > 0
                assert sampling < 1.0
                train_filenames = list(np.random.choice(np.array(train_filenames),
                                       int(sampling * len(train_filenames)),
                                       replace=False))

            train_dataset = dataset_class(dataset_path,
                                          train_filenames, self.opt.height,
                                          self.opt.width, is_train=True,
                                          disable_normalisation=self.opt.disable_normalisation,
                                          max_disparity=self.opt.max_disparity,
                                          keep_aspect_ratio=True,
                                          disable_synthetic_augmentation=
                                          self.opt.disable_synthetic_augmentation,
                                          disable_sharpening=self.opt.disable_sharpening,
                                          monodepth_model=self.opt.monodepth_model,
                                          disable_background=self.opt.disable_background
                                          )
            val_dataset = dataset_class(dataset_path, val_filenames,
                                        self.opt.height,
                                        self.opt.width, is_train=False,
                                        disable_normalisation=self.opt.disable_normalisation,
                                        max_disparity=self.opt.max_disparity,
                                        keep_aspect_ratio=True,
                                        disable_synthetic_augmentation=
                                        self.opt.disable_synthetic_augmentation,
                                        disable_sharpening=self.opt.disable_sharpening,
                                        monodepth_model=self.opt.monodepth_model,
                                        disable_background=self.opt.disable_background
                                        )
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)

        super().__init__(data_info, data_cfg, mode)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path, disp_img_path = full_paths
        left_img = Image.open(left_img_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)
        right_img = Image.open(right_img_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)
        disp_img = readpfm(disp_img_path)[0].astype(np.float32)
        disp_img[disp_img == np.inf] = 0

        occ_mask = Image.open(disp_img_path.replace('disp0GT.pfm', 'mask0nocc.png'))
        occ_mask = np.array(occ_mask, dtype=np.float32)
        occ_mask = occ_mask != 255.0

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
            'occ_mask': occ_mask
        }
        sample = self.transform(sample)
        sample['index'] = idx
        sample['name'] = left_img_path
        return sample
