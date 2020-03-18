import os
import glob
import csv
import re

import pandas
import matplotlib.pyplot as plt
from skimage.io import imread
from numpy import sum, load, mean, zeros_like, average, array, std
from skimage.transform import resize

cfg = {
    'data_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/pitch',
    'csv_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/true_vs_mono.csv',
    'filename_re': '(?P<image>[0-9]+_[0-9]+)_(?P<y>[+|-]?[0-9]+).png',
    'output_path': '/home/tom/Dropbox/PhD/Experiments/ICCV2019_depth/matlab/pitch_vs_disp.csv',
}
cfg['filenames_path'] = os.path.join(cfg['data_path'], 'filenames.txt')
cfg['disp_path'] = os.path.join(cfg['data_path'], 'output', 'disparities_pp.npy')
cfg['sem_path'] = os.path.join(cfg['data_path'], 'obj_map')

object_table = pandas.read_csv(cfg['csv_path'])

disp = load(cfg['disp_path'], mmap_mode='r')

data = []
with open(cfg['filenames_path'], 'rt') as f:
    for index, filename in enumerate(f):
        filename = filename.strip('\n')
        print(filename)
        m = re.match(cfg['filename_re'], filename)
        image = m.group('image')
        y = int(m.group('y'))
        sem = imread(os.path.join(cfg['sem_path'], filename))
        obj_indices = object_table[object_table.image == (image + '.png')].object_index.values
        for i in obj_indices:
            mask = zeros_like(sem, dtype=float)
            disp_r = resize(disp[index], mask.shape) * mask.shape[1]
            mask[sem == i] = 1.0
            mask[disp_r == 0] = 0.0
            pixel_count = sum(mask)
            avg_disp = average(disp_r, weights=mask)
            data.append({'filename': filename, 'image': image, 'y': y, 'obj_index': i, 'pixel_count': pixel_count,
                         'avg_disp': avg_disp})
df = pandas.DataFrame(data)

X = []
Y = []
for image in sorted(set(df.image)):
    for obj_index in sorted(set(df[df.image == image].obj_index)):
        print("Image: {}, obj_index: {}".format(image, obj_index))
        curve = df[(df.image == image) & (df.obj_index == obj_index)].sort_values('y')
        y = curve.y.values
        pixel_count = curve.pixel_count.values
        avg_disp = curve.avg_disp.values
        if not all(pixel_count == pixel_count[0]):
            continue
        X.append(y)
        Y.append(avg_disp - avg_disp[y == 0])
X = array(X)
Y = array(Y)

plt.figure()
plt.plot(X.flatten(), Y.flatten(), '.', alpha=0.05)
plt.xlabel('Horizon shift [px]')
plt.ylabel('Disparity shift [px]')

X2 = X[0]
Ymean = mean(Y, axis=0)
Ystd = std(Y, axis=0)
plt.figure()
plt.fill_between(X2, Ymean - Ystd, Ymean + Ystd, alpha=0.5)
plt.plot(X2, Ymean)
plt.xlabel('Horizon shift [px]')
plt.ylabel('Disparity shift [px]')

with open(cfg['output_path'], 'wt') as f:
    w = csv.writer(f)
    w.writerow(X2)
    w.writerows(Y)