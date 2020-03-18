import os
import re
import pandas
import matplotlib.pyplot as plt
import numpy as np

cfg = {
    'filenames_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/bottom_edge/images/filenames.txt',
    'filename_re': '(?P<scene>[0-9]*_[0-9]*)_(?P<object>[a-zA-Z]*_[a-zA-Z0-9]*)_(?P<value>[0-9\.\-]*)_(?P<size>[0-9\-]*).png',
    'monodepth_filenames_path': '/media/tom/seagate-8tb-ext4/models/monodepth/monodepth/utils/filenames/kitti_stereo_2015_test_files.txt',
    'monodepth_re': '.*/(?P<scene>[0-9]*_[0-9]*).jpg .*',
    'monodepth_disp_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/monodepth_disparities_kitti/disparities_pp.npy',
    'input_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/bottom_edge/output/dist.csv',
    'output_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/bottom_edge/output/dist_with_empty.csv',
}

# Read scenes used by bottom edge experiment
scenes = set()
with open(cfg['filenames_path'], 'rt') as f:
    for line in f:
        m = re.match(cfg['filename_re'], line)
        scenes.add(m.group('scene'))

# Calculate
disp = np.load(cfg['monodepth_disp_path'], mmap_mode='r')
disp_fn = 0