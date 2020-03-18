import re
import os
import glob
import pickle

from numpy import load, average
from skimage import img_as_float
from skimage.io import imread, imshow
from skimage.transform import resize
from scipy.io import loadmat

cfg = {
    'data_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/pos_vs_scale',
    'depth_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/pos_vs_scale/output_semodepth',
    'pickle_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/pos_vs_scale/output_semodepth/depths.p',

    'filenames_file': 'filenames_semodepth.txt',
    'filename_re': '(?P<set>[a-zA-Z_]*)/(?P<scene>[0-9]*_[0-9]*)_(?P<object>[a-zA-Z_]*[0-9]*)_(?P<distance>[0-9\.]*).png',
    'disparities_re': '.*disparities_(?P<index>[0-9]*).npy',
}

# Parse all results
results = {}
with open(os.path.join(cfg['data_path'], cfg['filenames_file']), 'rt') as f:
    for filename in f:
        filename = filename.split(',')[0]
        filename = filename[len(cfg['data_path']) + 1:]
        print(filename)

        mat_filename = os.path.join(cfg['depth_path'], filename.replace('.png', '.mat'))
        depth = loadmat(mat_filename)['mat'][0, :, :, 0]

        mask_filename = os.path.join(cfg['data_path'], filename.replace('.png', '_mask.png'))
        mask = img_as_float(imread(mask_filename))[:, :, 0]
        mask_r = resize(mask, depth.shape, mode='reflect', anti_aliasing=True)
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
