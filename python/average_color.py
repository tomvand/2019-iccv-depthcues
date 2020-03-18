import csv
import glob
import os

from numpy import array, zeros, zeros_like
from skimage.io import imread, imshow, imsave

cfg = {
    'image_path': '/media/tom/seagate-8tb-ext4/data/kitti/data_semantics/training/image_2',
    'semantics_path': '/media/tom/seagate-8tb-ext4/data/kitti/data_semantics/training/semantic',
    'csv_path': '/home/tom/Dropbox/PhD/Experiments/ICCV2019_depth/avg_colors.csv',
    'output_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/class_avg_colors',
}

img_filenames = sorted(glob.glob(os.path.join(cfg['image_path'], '*_10.png')))

if not os.path.isfile(cfg['csv_path']):
    data = {}
    for fn in img_filenames:
        name = os.path.basename(fn)
        print(name)
        img = imread(os.path.join(cfg['image_path'], name))
        sem = imread(os.path.join(cfg['semantics_path'], name))
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                color = img[y, x]
                label = sem[y, x]
                if label not in data:
                    data[label] = { 'mean': zeros(3), 'count': 0 }
                data[label]['mean'] = (data[label]['mean'] * data[label]['count'] + color) / (data[label]['count'] + 1)
                data[label]['count'] += 1

    with open(cfg['csv_path'], 'w') as f:
        w = csv.writer(f)
        for index, value in data.items():
            w.writerow([index, value['mean'][0], value['mean'][1], value['mean'][2]])


avg_colors = {}
with open(cfg['csv_path'], 'r') as f:
    r = csv.reader(f)
    for row in r:
        index = int(row[0])
        r = int(float(row[1]))
        g = int(float(row[2]))
        b = int(float(row[3]))
        avg_colors[index] = array([r, g, b])

with open(os.path.join(cfg['output_path'], 'filenames.txt'), 'w') as f:
    for fn in img_filenames:
        name = os.path.basename(fn)
        print(name)
        sem = imread(os.path.join(cfg['semantics_path'], name))
        img = zeros((sem.shape[0], sem.shape[1], 3), dtype='uint8')
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                img[y, x, :] = avg_colors[sem[y, x]]
        imsave(os.path.join(cfg['output_path'], 'images', name), img)
        f.write(name + '\n')
