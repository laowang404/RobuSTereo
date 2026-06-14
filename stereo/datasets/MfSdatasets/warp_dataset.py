# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Stereo-from-mono licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import random
import numpy as np
from PIL import Image  # using pillow-simd for increased speed
import time

import torch
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F

from skimage.filters import gaussian, sobel,sobel_v
from skimage.color import rgb2gray

from scipy.interpolate import griddata
import cv2
cv2.setNumThreads(0)

from .base_dataset import BaseDataset
from .utils import transfer_color


class WarpDataset(BaseDataset):

    def __init__(self,
                 data_path,
                 filenames,
                 feed_height,
                 feed_width,
                 max_disparity,
                 is_train=True,
                 disable_normalisation=False,
                 keep_aspect_ratio=True,
                 **kwargs):

        super(WarpDataset, self).__init__(data_path, filenames, feed_height, feed_width,
                                          is_train=is_train, has_gt=True,
                                          disable_normalisation=disable_normalisation,
                                          keep_aspect_ratio=keep_aspect_ratio)
        
        self.return_path = kwargs['return_path']
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.stereo_brightness = (0.8, 1.2)
            self.stereo_contrast = (0.8, 1.2)
            self.stereo_saturation = (0.8, 1.2)
            self.stereo_hue = (-0.01, 0.01)
            transforms.ColorJitter.get_params(
                self.stereo_brightness, self.stereo_contrast, self.stereo_saturation,
                self.stereo_hue)
        except TypeError:
            self.stereo_brightness = 0.2
            self.stereo_contrast = 0.2
            self.stereo_saturation = 0.2
            self.stereo_hue = 0.01

        self.silly_svsm = False

    def load_images(self, idx, do_flip=False):
        raise NotImplementedError

    def load_disparity(self, idx, do_flip=False):
        raise NotImplementedError

    def __getitem__(self, idx):
        # import pdb;pdb.set_trace()

        inputs = {}

        do_flip = False
        # if self.is_train and random.random() > 0.5:
        #     do_flip = True

        # load from disk
        if self.return_path:
            left_image, right_image, image_path = self.load_images(idx, do_flip=do_flip)
        else:
            left_image, background_image = self.load_images(idx, do_flip=do_flip)
        loaded_disparity = self.load_disparity(idx, do_flip=do_flip)

        inputs['left_image'] = left_image
        inputs['right_image'] = right_image

        inputs['right_image'] = transfer_color(np.array(inputs['right_image']),
                                              np.array(inputs['left_image']))

        if self.return_path:
            inputs = {'left_image': inputs['left_image'],
                    'right_image': inputs['right_image'],
                    'disparity': loaded_disparity.astype(float),
                    'img_path':[image_path]
                    }
        else:
            inputs = {'left_image': inputs['left_image'],
                    'right_image': inputs['right_image'],
                    'disparity': loaded_disparity.astype(float),
                    }

        # self.preprocess(inputs)
        return inputs
