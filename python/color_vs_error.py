import os
import glob
import matplotlib.pyplot as plt

import pandas
import numpy as np
from numpy import load, average
from skimage import img_as_float
from skimage.io import imread
from skimage.transform import resize

cfg = {
    'image_path': '/media/tom/seagate-8tb-ext4/data/kitti/data_scene_flow/training/image_2',
    'disp_path': '/media/tom/seagate-8tb-ext4/data/kitti/data_scene_flow/training/disp_noc_0',
    'obj_path': '/media/tom/seagate-8tb-ext4/data/kitti/data_scene_flow/training/obj_map',
    'monodepth_disp_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/monodepth_disparities_kitti',
    'csv_path': '/home/tom/Dropbox/PhD/Experiments/ICCV2019_depth/car_colors.csv',
}

colors = pandas.read_csv(cfg['csv_path'])
colors['color'].value_counts().plot.pie()

img_filenames = sorted([os.path.basename(n) for n in glob.glob(os.path.join(cfg['image_path'], '*_10.png'))])

disp = load(os.path.join(cfg['monodepth_disp_path'], 'disparities_pp.npy'), mmap_mode='r')

# plt.figure()
# ax1 = plt.subplot(4, 1, 1)
# img1 = plt.imshow([[0]])
# ax2 = plt.subplot(4, 1, 2)
# img2 = plt.imshow([[0]])
# ax3 = plt.subplot(4, 1, 3)
# img3 = plt.imshow([[0]])
# ax4 = plt.subplot(4, 1, 4)
# img4 = plt.imshow([[0]])
data = []
for index, row in colors.iterrows():
    print(row['image'])
    img = imread(os.path.join(cfg['image_path'], row['image']))
    true_disp = imread(os.path.join(cfg['disp_path'], row['image'])) / 256.0
    mono_disp = resize(disp[img_filenames.index(row['image'])], true_disp.shape) * true_disp.shape[1]
    obj_map = imread(os.path.join(cfg['obj_path'], row['image']))
    mask = img_as_float((obj_map == row['obj_index']) & (true_disp != 0.0))

    # plt.sca(ax1)
    # img1.remove()
    # img1 = plt.imshow(img)
    # plt.title(row['image'])
    # plt.sca(ax2)
    # img2.remove()
    # img2 = plt.imshow(true_disp)
    # plt.sca(ax3)
    # img3.remove()
    # img3 = plt.imshow(mono_disp)
    # plt.sca(ax4)
    # img4.remove()
    # img4 = plt.imshow(obj_map)
    # plt.draw()
    # plt.pause(0.01)

    disp_error = mono_disp - true_disp
    disp_error[true_disp == 0.0] = 0.0
    abs_error = np.abs(disp_error)
    avg_true_disp = average(true_disp, weights=mask)
    avg_abs_error = average(abs_error, weights=mask)

    data.append({'image': row['image'], 'obj_index': row['obj_index'], 'color': row['color'],
                 'avg_true_disp': avg_true_disp, 'avg_abs_error': avg_abs_error})

df = pandas.DataFrame(data)

plt.figure()
colors = set(df['color'])
boxplot_data = []
boxplot_labels = []
for index, color in enumerate(colors):
    color_data = df[df['color'] == color]
    boxplot_data.append(color_data['avg_abs_error'])
    boxplot_labels.append('{}\n(N={})'.format(color, len(color_data)))
plt.boxplot(boxplot_data, labels=boxplot_labels, showfliers=False)
plt.ylabel('Avg. abs error per car [-]')