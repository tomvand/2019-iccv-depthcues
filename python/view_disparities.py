import os
import re
import glob

import matplotlib.pyplot as plt

from numpy import load, max
from skimage.transform import resize

cfg = {
    'data_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/pos_vs_scale',
    'disparities_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/pos_vs_scale/output/disparities',

    'filenames_fmt': 'filenames_{:03d}.txt',
    'filename_re': '(?P<set>[a-zA-Z_]*)/(?P<scene>[0-9]*_[0-9]*)_(?P<object>[a-zA-Z_]*[0-9]*)_(?P<distance>[0-9\.]*).png',
    'disparities_re': '.*disparities_pp_(?P<index>[0-9]*).npy',
}

# Locate disparity files
disparities_files = glob.glob(cfg['disparities_path'] + '/disparities_pp_*.npy')

# Locate corresponding filenames files
filenames_files = []
for disparities_filename in disparities_files:
    index = int(re.match(cfg['disparities_re'], disparities_filename).group('index'))
    filename = os.path.join(cfg['data_path'], cfg['filenames_fmt'].format(index))
    filenames_files.append(filename)
    assert(os.path.isfile(filename))

# Report data files found
print('Found the following data:')
for i in range(len(disparities_files)):
    print('{}\t{}'.format(
        re.match('.*/([^/]*)$', disparities_files[i]).group(1),
        re.match('.*/([^/]*)$', filenames_files[i]).group(1)
    ))

# View disparities
plt.figure()
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
for i in [1]: #range(len(disparities_files)):
    print('Parsing {}...'.format(disparities_files[i]))
    disparities = load(disparities_files[i], mmap_mode='r') # mmap required because of large file size
    with open(filenames_files[i], 'rt') as f:
        for slice, filename in enumerate(f):
            filename = filename.strip('\n')
            print(filename)
            img = plt.imread(os.path.join(cfg['data_path'], filename))
            plt.sca(ax1)
            plt.cla()
            plt.imshow(img)
            plt.sca(ax2)
            plt.cla()
            plt.imshow(resize(disparities[slice, :, :], img.shape)[:, :, 1])
            # plt.imshow(disparities[slice, :, :])
            plt.ginput()