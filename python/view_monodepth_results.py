import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from numpy import load, abs
from skimage.io import imread
from skimage.transform import resize

cfg = {
    'image_path': '/media/tom/seagate-8tb-ext4/data/kitti/data_scene_flow/training/image_2',
    'monodepth_disp_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/monodepth_disparities_kitti',
    'disp_path': '/media/tom/seagate-8tb-ext4/data/kitti/data_scene_flow/training/disp_noc_0',
}

img_filenames = sorted([os.path.basename(n) for n in glob.glob(os.path.join(cfg['image_path'], '*_10.png'))])

disp = load(os.path.join(cfg['monodepth_disp_path'], 'disparities_pp.npy'))

plt.figure()
ax_img = plt.subplot(4, 1, 1)
img_img = None
ax_mono = plt.subplot(4, 1, 2)
img_mono = None
ax_true = plt.subplot(4, 1, 3)
img_true = None
ax_err = plt.subplot(4, 1, 4)
img_err = None
for index, filename in enumerate(img_filenames):
    filename = os.path.basename(filename).strip('\n')
    img = imread(os.path.join(cfg['image_path'], filename))
    true_disp = imread(os.path.join(cfg['disp_path'], filename)) / 256.0
    meas_disp = disp[index]
    meas_disp = resize(meas_disp, true_disp.shape) * true_disp.shape[1]
    err = np.abs(meas_disp - true_disp)
    err[true_disp == 0.0] = 0.0
    if not img_img:
        plt.sca(ax_img)
        img_img = plt.imshow(img)
    else:
        img_img.set_data(img)
    if not img_mono:
        plt.sca(ax_mono)
        img_mono = plt.imshow(meas_disp)
    else:
        img_mono.set_data(meas_disp)
        img_mono.autoscale()
    if not img_true:
        plt.sca(ax_true)
        img_true = plt.imshow(true_disp)
    else:
        img_true.set_data(true_disp)
        img_true.autoscale()
    if not img_err:
        plt.sca(ax_err)
        img_err = plt.imshow(err)
    else:
        img_err.set_data(err)
        img_err.autoscale()
    plt.sca(ax_img)
    plt.title(filename)
    plt.draw()
    plt.ginput()
