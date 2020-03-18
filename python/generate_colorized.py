import glob
import os
import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgb2hsv, hsv2rgb

cfg = {
    'image_path': '/media/tom/seagate-8tb-ext4/data/kitti/data_semantics/training/image_2',
    'sem_rgb_path': '/media/tom/seagate-8tb-ext4/data/kitti/data_semantics/training/semantic_rgb',
    'output_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/colorized/images',
}

img_filenames = sorted(glob.glob(os.path.join(cfg['image_path'], '*_10.png')))

for fn in img_filenames:
    name = os.path.basename(fn)
    print(name)

    img_rgb = imread(os.path.join(cfg['image_path'], name))
    img_sem_rgb = imread(os.path.join(cfg['sem_rgb_path'], name))

    img_hsv = rgb2hsv(img_rgb)
    img_sem_hsv = rgb2hsv(img_sem_rgb)

    img_out_hsv = np.stack((img_sem_hsv[:, :, 0], img_sem_hsv[:, :, 1], img_hsv[:, :, 2]), axis=2)
    img_out_rgb = hsv2rgb(img_out_hsv)
    imsave(os.path.join(cfg['output_path'], name), img_out_rgb)