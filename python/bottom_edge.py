import os
import re
import pandas
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float
from skimage.io import imread, imshow
from skimage.transform import resize

cfg = {
    'filenames_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/bottom_edge/images/filenames.txt',
    'disparities_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/bottom_edge/output/disparities_pp.npy',
    'filename_re': '(?P<scene>[0-9]*_[0-9]*)_(?P<object>[a-zA-Z]*_[a-zA-Z0-9]*)_(?P<value>[0-9\.\-]*)_(?P<size>[0-9\-]*).png',
    'image_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/bottom_edge/images',
    'mask_fmt': '{scene}_{object}_mask.png',
    'ground_values_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/bottom_edge/images/ground_values.csv',
    'output_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/bottom_edge/output/dist.csv',
}

disp = np.load(cfg['disparities_path'], mmap_mode='r')
ground_values = pandas.read_csv(cfg['ground_values_path'], index_col=('scene', 'object'))
data = []
with open(cfg['filenames_path'], 'rt') as f:
    for index, line in enumerate(f):
        # Parse filename
        m = re.match(cfg['filename_re'], line.strip('\n'))
        scene = m.group('scene')
        obj = m.group('object')
        value = float(m.group('value'))
        size = int(m.group('size'))

        print(line.strip('\n'))

        # Load mask
        mask_fn = cfg['mask_fmt'].format(scene=scene, object=obj)
        mask = img_as_float(imread(os.path.join(cfg['image_path'], mask_fn)))
        mask_r = resize(mask, disp.shape[1:3])

        # Calculate average distance
        depth = 1.0 / disp[index] # up to scale
        dist = np.average(depth, weights=mask_r)

        # Read ground value
        gv = ground_values.loc[(scene, obj)]['ground_value']

        # Store result
        data.append({
            'scene': scene,
            'object': obj,
            'value': value,
            'size': size,
            'ground_value': gv,
            'dist': dist,
        })

df = pandas.DataFrame(data)
df.to_csv(cfg['output_path'], index=False)