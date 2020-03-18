import glob
import os
import matplotlib.pyplot as plt

from skimage.io import imread

cfg = {
    'image_path': '/media/tom/seagate-8tb-ext4/data/kitti/data_scene_flow/training/image_2',
    'obj_map_path': '/media/tom/seagate-8tb-ext4/data/kitti/data_scene_flow/training/obj_map',
}

filenames = sorted(glob.glob(os.path.join(cfg['obj_map_path'], '*.png')))

plt.figure()
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
for filename in filenames[:]:
    filename = os.path.basename(filename)
    img = imread(os.path.join(cfg['image_path'], os.path.basename(filename)))
    label = imread(os.path.join(cfg['obj_map_path'], os.path.basename(filename)))
    plt.sca(ax1)
    plt.imshow(img)
    plt.sca(ax2)
    plt.imshow(label)
    plt.title(filename)
    plt.ginput()
