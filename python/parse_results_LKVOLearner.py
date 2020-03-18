import re
import os
import glob
import pickle

from numpy import load, average
from skimage import img_as_float
from skimage.io import imread, imshow
from skimage.transform import resize

cfg = {
    'data_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/pos_vs_scale',
    'disparities_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/pos_vs_scale/output_LKVOLearner',
    'pickle_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/pos_vs_scale/output_LKVOLearner/depths.p',

    'filenames_fmt': 'filenames_{:03d}.txt',
    'filename_re': '(?P<set>[a-zA-Z_]*)/(?P<scene>[0-9]*_[0-9]*)_(?P<object>[a-zA-Z_]*[0-9]*)_(?P<distance>[0-9\.]*).png',
    'disparities_re': '.*/(?P<index>[0-9]*).npy',
}
# Locate disparity files
depths_files = glob.glob(cfg['disparities_path'] + '/*.npy')

# Locate corresponding filenames files
filenames_files = []
for depths_filename in depths_files:
    index = int(re.match(cfg['disparities_re'], depths_filename).group('index'))
    filename = os.path.join(cfg['data_path'], cfg['filenames_fmt'].format(index))
    filenames_files.append(filename)
    assert(os.path.isfile(filename))

# Report data files found
print('Found the following data:')
for i in range(len(depths_files)):
    print('{}\t{}'.format(
        re.match('.*/([^/]*)$', depths_files[i]).group(1),
        re.match('.*/([^/]*)$', filenames_files[i]).group(1)
    ))

# Parse all results
results = {}
for i in range(len(depths_files)):
    print('Parsing {}...'.format(depths_files[i]))
    depths = load(depths_files[i], mmap_mode='r') # mmap required because of large file size
    with open(filenames_files[i], 'rt') as f:
        for slice, filename in enumerate(f):
            filename = filename.strip('\n')
            print(filename)
            mask_filename = os.path.join(cfg['data_path'], filename.replace('.png', '_mask.png'))
            mask = img_as_float(imread(mask_filename))[:, :, 0]
            mask_r = resize(mask, depths.shape[1:3], mode='reflect', anti_aliasing=True)
            depth = depths[slice, :, :] # Correct up-to-scale
            measured_distance = average(depth, weights=mask_r)

            filename_match = re.match(cfg['filename_re'], filename)
            set = filename_match.group('set')
            scene = filename_match.group('scene')
            object = filename_match.group('object')
            distance = float(filename_match.group('distance'))

            if set not in results:
                results[set] = {}
            if scene not in results[set]:
                results[set][scene] = {}
            if object not in results[set][scene]:
                results[set][scene][object] = {}
            results[set][scene][object][distance] = measured_distance

# Save results dict for further processing
pickle.dump(results, open(cfg['pickle_path'], 'wb'))
