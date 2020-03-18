from skimage.io import imread, imsave, imshow
from skimage.transform import rotate
from numpy import arange, asarray, mean
import glob
import os

cfg = {
    'image_path': '/media/tom/seagate-8tb-ext4/data/kitti/data_scene_flow/training/image_2',
    'output_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/roll',
    'angle_range': arange(-10, 11, 2),
    'crop': 0.65 * asarray([375, 1242]),
}

img_filenames = sorted(glob.glob(os.path.join(cfg['image_path'], '*_10.png')))

with open(os.path.join(cfg['output_path'], 'filenames.txt'), 'w') as f:
    for fn in img_filenames:
        name = os.path.splitext(os.path.basename(fn))[0]
        img = imread(fn)
        ymin = int(img.shape[0] / 2 - cfg['crop'][0] / 2)
        ymax = int(img.shape[0] / 2 + cfg['crop'][0] / 2)
        xmin = int(img.shape[1] / 2 - cfg['crop'][1] / 2)
        xmax = int(img.shape[1] / 2 + cfg['crop'][1] / 2)
        for angle in cfg['angle_range']:
            img_rot = rotate(img, angle)
            img_crop = img_rot[ymin:ymax, xmin:xmax]
            save_filename = '{}_{}.png'.format(name, angle)
            print(save_filename)
            imsave(os.path.join(cfg['output_path'], 'images', save_filename), img_crop)
            f.write(save_filename + '\n')
            # imshow(img_crop)
            # break
        # break