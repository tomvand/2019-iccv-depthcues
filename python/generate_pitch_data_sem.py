from skimage.io import imread, imsave, imshow
from numpy import arange
import glob
import os

cfg = {
    'image_path': '/media/tom/seagate-8tb-ext4/data/kitti/data_scene_flow/training/obj_map',
    'output_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/pitch',
    'crop_height': 300,
    'y_range': arange(-30, 40, 10),
}

img_filenames = sorted(glob.glob(os.path.join(cfg['image_path'], '*_10.png')))

# with open(os.path.join(cfg['output_path'], 'filenames.txt'), 'w') as f:
for fn in img_filenames:
    name = os.path.splitext(os.path.basename(fn))[0]
    img = imread(fn)
    ymin = img.shape[0] // 2 - cfg['crop_height'] // 2 + cfg['y_range']
    ymax = img.shape[0] // 2 + cfg['crop_height'] // 2 + cfg['y_range']
    w = cfg['crop_height'] / img.shape[0] * img.shape[1]
    xmin = int(img.shape[1] / 2 - w / 2)
    xmax = int(img.shape[1] / 2 + w / 2)
    for (y, y0, y1) in zip(cfg['y_range'], ymin, ymax):
        img_crop = img[y0:y1, xmin:xmax]
        save_filename = '{}_{}.png'.format(name, y)
        print(save_filename)
        imsave(os.path.join(cfg['output_path'], 'obj_map', save_filename), img_crop)

