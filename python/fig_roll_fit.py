import numpy as np
import matplotlib.pyplot as plt
import csv
from skimage import img_as_float
from skimage.transform import hough_line, hough_line_peaks, resize
from skimage.io import imshow, imsave

cfg = {
    'disp_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/monodepth_disparities_kitti/disparities_pp.npy',
    'output_path': '/home/tom/Dropbox/PhD/Experiments/ICCV2019_depth/matlab/roll_fit.csv',
    'output_img': '/home/tom/Dropbox/PhD/Experiments/ICCV2019_depth/matlab/roll_fit.png',
    'index': 0,
    'test_disp': (0.03, 0.031),
    'roll_range': np.linspace(-30.0, 30.0, 100),
    'image_shape': (375, 1242),
}

disp = np.load(cfg['disp_path'], mmap_mode='r')
disp = disp[cfg['index']]
disp = resize(disp, cfg['image_shape'])

slice = (cfg['test_disp'][0] < disp) & (disp < cfg['test_disp'][1])
h, theta, d = hough_line(slice, theta=np.linspace(np.deg2rad(90 - 30), np.deg2rad(90 + 30), 250))
_, angles, dists = hough_line_peaks(h, theta, d, num_peaks=1)
roll = 90 - np.rad2deg(angles[0])

y0 = dists[0] / np.sin(angles[0])
y1 = (dists[0] - slice.shape[1] * np.cos(angles[0])) / np.sin(angles[0])

plt.imshow(img_as_float(slice))
plt.plot([0, slice.shape[1]], [y0, y1])

with open(cfg['output_path'], 'wt') as f:
    w = csv.writer(f)
    w.writerow([y0, y1])

imsave(cfg['output_img'], img_as_float(slice))