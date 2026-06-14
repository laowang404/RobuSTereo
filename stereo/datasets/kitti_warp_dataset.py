import os
import torch.utils.data as torch_data
import numpy as np
from PIL import Image
from .dataset_template import DatasetTemplate


class KittiWarpDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        # self.return_right_disp = self.data_info.RETURN_RIGHT_DISP
        self.use_noc = self.data_info.get('USE_NOC', False)
        # self.data_num = 10000

    # def __len__(self):
    #     return self.data_num

    def __getitem__(self, idx):

        # idx = idx * np.random.randint(1, 10000) % len(self.data_list)

        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path, disp_img_path = full_paths
        if self.use_noc:
            disp_img_path = disp_img_path.replace('disp_occ', 'disp_noc')
        # image
        left_img = np.array(Image.open(left_img_path).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(right_img_path).convert('RGB'), dtype=np.float32)
        # disp
        disp_img = np.load(disp_img_path)
        # mono_img = mono_img / mono_img.max()
        
        # disp_img = (np.array(Image.open(disp_img_path), dtype=np.float32) / 256.0) if not disp_img_path.endswith("no_GT") else (np.zeros_like(mono_img) - 1)
        # print(disp_img[0,0])
        
        # print("left_img_shape: ", left_img.shape)
        # print("right_img_shape: ", right_img.shape)
        # print("disp_img_shape: ", disp_img.shape)
        # print("mono_img_shape: ", mono_img.shape)
        # exit()
        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
        }


        # if self.return_right_disp:
        #     disp_img_right_path = disp_img_path.replace('c_0', 'c_1')
        #     disp_img_right = np.array(Image.open(disp_img_right_path), dtype=np.float32) / 256.0
        #     sample['disp_right'] = disp_img_right
        
        # print("B_left_img_shape: ", sample['left'].shape)
        # print("B_right_img_shape: ", sample['right'].shape)
        # print("B_disp_img_shape: ", sample['disp'].shape)
        # print("B_mono_img_shape: ", sample['mono'].shape)

        sample = self.transform(sample)
        sample['index'] = idx
        sample['name'] = left_img_path

        # print("left_img_shape: ", sample['left'].shape)
        # print("right_img_shape: ", sample['right'].shape)
        # print("disp_img_shape: ", sample['disp'].shape)
        # print("mono_img_shape: ", sample['mono'].shape)

        return sample
