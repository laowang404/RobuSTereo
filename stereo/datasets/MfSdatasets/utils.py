from pathlib import Path
import numpy as np

PATH_CONFIG = {# training datasets
"ADE20K": "/data/wangyuran/DepthGen/Datasets/Training_set/ADE20K/",
"diode": "/data/wangyuran/DepthGen/Datasets/Training_set/diode/",
"diw": "/data/wangyuran/DepthGen/Datasets/Training_set/diw/",
"mapillary": "/data/wangyuran/DepthGen/Datasets/Training_set/mapillary",
"mscoco": "/data/wangyuran/DepthGen/Datasets/Training_set/coco17",
"sceneflow": "/datasets/sceneflow/",
# testing datasets,
"eth3d": "/data/wangyuran/DepthGen/Datasets/Testing_set/ETH3D",
"flicker": "/datasets/flicker1024",
"kitti2015": "/data/wangyuran/DepthGen/Datasets/KITTI15",
"kitti2012": "/data/wangyuran/DepthGen/Datasets/Testing_set/KITTI12",
"middlebury": "/data/wangyuran/DepthGen/Datasets/Testing_set/Middlebury/"
}

def get_left_view(path):
    path_sp = path.split('/')
    for j, segment in enumerate(path_sp):
        if segment == 'Training_set':
            break
    j += 2
    path_sp[j] = path_sp[j] + '_left'
    _path = '/'.join(path_sp)
    path_obj = Path(_path)  # 转换为Path对象
    path_left = path_obj  # 在转换之后立即更新paths列表

    return path_left

def get_right_view(path):
    path_sp = path.split('/')
    for j, segment in enumerate(path_sp):
        if segment == 'Training_set':
            break
    j += 2
    path_sp[j] = path_sp[j] + '_inpainting'
    _path = '/'.join(path_sp)
    path_obj = Path(_path)  # 转换为Path对象
    path_right = path_obj  # 在转换之后立即更新paths列表

    return path_right

def get_disp_view(path):
    path_sp = path.split('/')
    for j, segment in enumerate(path_sp):
        if segment == 'Training_set':
            break
    j += 2
    path_sp[j] = path_sp[j] + '_disp'
    _path = '/'.join(path_sp)
    path_obj = Path(_path)  # 转换为Path对象
    path_right = path_obj  # 在转换之后立即更新paths列表

    return path_right

def transfer_color(target, source):
    target = target.astype(float) / 255
    source = source.astype(float) / 255

    target_means = target.mean(0).mean(0)
    target_stds = target.std(0).std(0)

    source_means = source.mean(0).mean(0)
    source_stds = source.std(0).std(0)

    target -= target_means
    target /= target_stds / source_stds
    target += source_means

    target = np.clip(target, 0, 1)
    target = (target * 255).astype(np.uint8)

    return target

def readlines(filename):
    """ read lines of a text file """
    with open(filename, 'r') as file_handler:
        lines = file_handler.read().splitlines()
    return lines