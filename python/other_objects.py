import os
import pandas
import re
import matplotlib.pyplot as plt
import numpy as np

from numpy import load, average, newaxis, mean, std, vstack, tile
from numpy.matlib import repmat
from skimage import img_as_float, img_as_ubyte
from skimage.color import gray2rgb
from skimage.io import imread, imsave
from skimage.transform import resize

cfg = {
    'data_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/other_objects',
    'filename_re': '(?P<image>[0-9]+_[0-9]+)_(?P<object>[a-z_]+)_(?P<distance>[0-9.]+).png',
    'print_filenames': '0000.0_10_[a-z_]+_1.0.png'
}
cfg['image_path'] = os.path.join(cfg['data_path'], 'images')
cfg['filenames_path'] = os.path.join(cfg['data_path'], 'filenames.txt')
cfg['disp_path'] = os.path.join(cfg['data_path'], 'output', 'disparities_pp.npy')
cfg['csv_path'] = os.path.join(cfg['data_path'], 'depths.csv')

disp = load(cfg['disp_path'], mmap_mode='r')

if os.path.exists(cfg['csv_path']):
    df = pandas.read_csv(cfg['csv_path'])
else:
    data = []
    with open(cfg['filenames_path'], 'rt') as f:
        for index, filename in enumerate(f):
            filename = os.path.basename(filename.strip('\n'))
            print(filename)
            mask = img_as_float(imread(os.path.join(cfg['image_path'], filename.replace('.png', '_mask.png'))))[:, :, 0]
            mask_r = resize(mask, disp.shape[1:3], mode='reflect', anti_aliasing=True)
            avg_disp = average(disp[index], weights=mask_r)
            m = re.match(cfg['filename_re'], filename)
            image = m.group('image')
            obj = m.group('object')
            distance = float(m.group('distance'))
            data.append({'image': image, 'object': obj, 'distance': distance, 'avg_disp': avg_disp, 'inv_disp': 1.0 / avg_disp})
            if re.match(cfg['print_filenames'], filename):
                img = imread(os.path.join(cfg['image_path'], filename))
                disp_r = resize(disp[index], img.shape[0:2], mode='reflect', anti_aliasing=True)
                disp_r -= np.min(disp_r)
                disp_r /= np.max(disp_r)
                disp_r = img_as_ubyte(gray2rgb(disp_r))
                preview = vstack([img, disp_r])
                imsave(os.path.join(cfg['data_path'], filename), preview)
    df = pandas.DataFrame(data)
    df.to_csv(cfg['csv_path'], index=False)


pivot = df.pivot_table(index=['object', 'image'], columns='distance', values='inv_disp')
pivot = pivot.div(pivot.values[:, 0], axis='index')
X = pivot.columns.values
plt.figure()
for object in pivot.index.levels[0]:
    Ymean = pivot.xs(object).mean(axis='index').values
    Ystd = pivot.xs(object).std(axis='index').values
    plt.fill_between(X, Ymean - Ystd, Ymean + Ystd, alpha=0.2)
    plt.plot(X, Ymean, label=object)
plt.plot([1.0, 3.0], [1.0, 3.0], 'k--', label='Expected')
plt.legend()
plt.xlabel('True relative distance [-]')
plt.ylabel('Measured relative distance [-]')