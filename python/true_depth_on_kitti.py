import os
import glob
import csv

import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import resize
from numpy import zeros_like, average, load, asarray

cfg = {
    'data_path': '/media/tom/seagate-8tb-ext4/data/kitti/data_scene_flow/training',
    'monodepth_disp_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/monodepth_disparities_kitti',
    'csv_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/true_vs_mono.csv',
}

# Generate subpaths
cfg['image_path'] = os.path.join(cfg['data_path'], 'image_2')
cfg['object_path'] = os.path.join(cfg['data_path'], 'obj_map')
cfg['disp_path'] = os.path.join(cfg['data_path'], 'disp_noc_0')

# Create data if csv does not exist
if not os.path.isfile(cfg['csv_path']):
    # Get list of images
    img_filenames = sorted([os.path.basename(n) for n in glob.glob(os.path.join(cfg['object_path'], '*.png'))])

    # Load monodepth disparities
    mono_disp_array = load(os.path.join(cfg['monodepth_disp_path'], 'disparities_pp.npy'), mmap_mode='r')

    # For each image, ask user to select cars
    data = {
        'image': [],
        'object_index': [],
        'true_disp': [],
        'mono_disp': [],
    }
    for i, img_fn in enumerate(img_filenames):
        # Show image and car labels
        f = plt.figure()
        img = imread(os.path.join(cfg['image_path'], img_fn))
        img_obj = imread(os.path.join(cfg['object_path'], img_fn))
        img_disp = imread(os.path.join(cfg['disp_path'], img_fn)) / 256.0  # See Stereo 2015 devkit readme
        img_mono = resize(mono_disp_array[i, :, :], img_disp.shape) * img_disp.shape[1]  # See https://github.com/mrharicot/monodepth/issues/118
        plt.subplot(4, 1, 1)
        plt.imshow(img)
        plt.subplot(4, 1, 2)
        plt.imshow(img_disp)
        plt.subplot(4, 1, 3)
        plt.imshow(img_mono)
        plt.subplot(4, 1, 4)
        plt.imshow(img_obj)
        # Ask user to select cars (LMB: select, RMB: finish)
        pts = plt.ginput(-1, show_clicks=True, mouse_stop=3, mouse_pop=2)
        for pt in pts:
            index = img_obj[int(pt[1]), int(pt[0])]
            if index is not 0:
                mask = zeros_like(img_disp, dtype=float)
                mask[img_obj == index] = 1.0
                mask[img_disp == 0] = 0.0
                true_disp = average(img_disp, weights=mask)
                mono_disp = average(img_mono, weights=mask)
                print('{:.1f} - {:.1f}'.format(true_disp, mono_disp))
                data['image'].append(img_fn)
                data['object_index'].append(index)
                data['true_disp'].append(true_disp)
                data['mono_disp'].append(mono_disp)
        plt.close(f)

    # Save data as csv
    with open(cfg['csv_path'], 'w') as f:
        writer = csv.DictWriter(f, data.keys())
        writer.writeheader()
        for i in range(len(data['true_disp'])):
            row = {}
            for key, value in data.items():
                row[key] = value[i]
            writer.writerow(row)

# Load data from CSV
data = {
    'image': [],
    'object_index': [],
    'true_disp': [],
    'mono_disp': [],
}
with open(cfg['csv_path']) as f:
    reader = csv.DictReader(f)
    for row in reader:
        for key, value in row.items():
            data[key].append(value)
data['true_disp'] = [float(x) for x in data['true_disp']]
data['mono_disp'] = [float(x) for x in data['mono_disp']]

# Plot true vs predicted depth
true_depth = 1.0 / asarray(data['true_disp'])
mono_depth = 1.0 / asarray(data['mono_disp'])

plt.figure()
plt.scatter(true_depth, mono_depth)
plt.xlabel('True inverse disparity [-]')
plt.ylabel('Predicted inverse disparity [-]')
plt.show()