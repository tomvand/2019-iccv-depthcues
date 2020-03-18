import os
import numpy as np
import matplotlib.cm
from skimage import img_as_ubyte
from skimage.io import imread, imsave, imshow
from skimage.transform import resize

cfg = {
    'image_nr': 6,
    'image_fmt': '{index:06d}_10.png',
    'image_size': (375, 1242),
    'output_path': '/home/tom/Dropbox/PhD/Experiments/ICCV2019_depth/colored_images.png',

    'image_classes': ['unmodified', 'grayscale', 'colorized', 'segmap_rgb', 'class_avg_colors'],
    'image_paths': {
        'unmodified': {
            'img': '/media/tom/seagate-8tb-ext4/data/kitti/data_scene_flow/training/image_2',
            'disp': '/media/tom/seagate-8tb-ext4/data/kitti_derived/monodepth_disparities_kitti/disparities.npy',
        },
        'grayscale': {
            'img': '/media/tom/seagate-8tb-ext4/data/kitti_derived/grayscale/images',
            'disp': '/media/tom/seagate-8tb-ext4/data/kitti_derived/grayscale/output/disparities.npy',
        },
        'colorized': {
            'img': '/media/tom/seagate-8tb-ext4/data/kitti_derived/colorized/images',
            'disp': '/media/tom/seagate-8tb-ext4/data/kitti_derived/colorized/output/disparities.npy',
        },
        'segmap_rgb': {
            'img': '/media/tom/seagate-8tb-ext4/data/kitti/data_semantics/training/semantic_rgb',
            'disp': '/media/tom/seagate-8tb-ext4/data/kitti_derived/segmap_rgb/output/disparities.npy',
        },
        'class_avg_colors': {
            'img': '/media/tom/seagate-8tb-ext4/data/kitti_derived/class_avg_colors/images',
            'disp': '/media/tom/seagate-8tb-ext4/data/kitti_derived/class_avg_colors/output/disparities.npy',
        },
    },
}

# total_img = np.zeros((2 * cfg['image_size'][0], len(cfg['image_classes']) * cfg['image_size'][1], 3), dtype=np.uint8)
total_img = np.zeros((len(cfg['image_classes']) * cfg['image_size'][0], 2 * cfg['image_size'][1], 3), dtype=np.uint8)
cm = matplotlib.cm.ScalarMappable(None, cmap='plasma')
cm.set_clim(0.00, 0.08)
for i, c in enumerate(cfg['image_classes']):
    img = imread(os.path.join(cfg['image_paths'][c]['img'], cfg['image_fmt'].format(index=cfg['image_nr'])))
    disp = np.load(os.path.join(cfg['image_paths'][c]['disp']), mmap_mode='r')[cfg['image_nr']]
    disp = resize(disp, cfg['image_size'])
    disp = img_as_ubyte(cm.to_rgba(disp)[:, :, :3])

    # total_img[0:cfg['image_size'][0], (i * cfg['image_size'][1]):((i + 1) * cfg['image_size'][1]), :] = img
    # total_img[cfg['image_size'][0]:(2 * cfg['image_size'][0]), (i * cfg['image_size'][1]):((i + 1) * cfg['image_size'][1]), :] = disp

    total_img[(i * cfg['image_size'][0]):((i + 1) * cfg['image_size'][0]), 0:cfg['image_size'][1], :] = img
    total_img[(i * cfg['image_size'][0]):((i + 1) * cfg['image_size'][0]), cfg['image_size'][1]:(2 * cfg['image_size'][1]), :] = disp

imsave(cfg['output_path'], total_img)