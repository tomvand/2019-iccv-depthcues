import glob
import os
import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgb2gray

cfg = {
    'image_path': '/media/tom/seagate-8tb-ext4/data/kitti/data_semantics/training/image_2',
    'output_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/grayscale/images',
}

img_filenames = sorted(glob.glob(os.path.join(cfg['image_path'], '*_10.png')))

for fn in img_filenames:
    name = os.path.basename(fn)
    print(name)
    img = imread(fn)
    img_g = rgb2gray(img)
    img_g = np.stack((img_g, img_g, img_g), axis=2)
    imsave(os.path.join(cfg['output_path'], name), img_g)