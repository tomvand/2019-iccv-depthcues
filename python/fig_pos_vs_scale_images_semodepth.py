import os
import numpy as np
import matplotlib.cm
import scipy.io as sio
from skimage import img_as_ubyte
from skimage.draw import polygon_perimeter
from skimage.io import imread, imshow, imsave
from skimage.measure import find_contours
from skimage.morphology import binary_dilation, square
from skimage.transform import resize

cfg = {
    'data_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/pos_vs_scale',
    'disp_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/pos_vs_scale/output_semodepth',
    'output_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/pos_vs_scale/output_semodepth/pos_vs_scale_examples.png',
    'filenames_index': 8,
    'scene': '000147_10',
    'object': 'car_rearright6',
    'filename_fmt': '{set}/{scene}_{object}_{distance:3.1f}.png',
    'mask_fmt': '{set}/{scene}_{object}_{distance:3.1f}_mask.png',
    'disp_fmt': '{set}/{scene}_{object}_{distance:3.1f}.mat',
    'sets': ['normal', 'pos_only', 'scale_only'],
    'distances': [1.0, 1.5, 3.0],
    'image_size': (375, 1242)
}

# Find indices of scene/object/distance
# indices = {}
# for s in cfg['sets']:
#     for d in cfg['distances']:
#         with open(os.path.join(cfg['data_path'], 'filenames_{:03d}.txt'.format(cfg['filenames_index'])), 'rt') as f:
#             for index, row in enumerate(f):
#                 if row.strip('\n') == cfg['filename_fmt'].format(set=s, scene=cfg['scene'], object=cfg['object'], distance=d):
#                     indices[(s, d)] = index
#                     break

# Compile images
cm = matplotlib.cm.ScalarMappable(None, cmap='plasma')
cm.set_clim(0.00, 0.20)
# disparities = np.load(os.path.join(cfg['disp_path'], cfg['disp_fmt'].format(cfg['filenames_index'])), mmap_mode='r')
full_image = np.zeros((2 * len(cfg['sets']) * cfg['image_size'][0], len(cfg['distances']) * cfg['image_size'][1], 3), dtype=np.uint8)
for si, s in enumerate(cfg['sets']):
    print('Set: {}'.format(s))
    imgs = np.zeros((cfg['image_size'][0], len(cfg['distances']) * cfg['image_size'][1], 3), dtype=np.uint8)
    disps = np.zeros_like(imgs)
    for di, d in enumerate(cfg['distances']):
        fn = cfg['filename_fmt'].format(set=s, scene=cfg['scene'], object=cfg['object'], distance=d)
        mask_fn = cfg['mask_fmt'].format(set=s, scene=cfg['scene'], object=cfg['object'], distance=d)
        disp_fn = cfg['disp_fmt'].format(set=s, scene=cfg['scene'], object=cfg['object'], distance=d)
        img = imread(os.path.join(cfg['data_path'], fn))
        img = img_as_ubyte(resize(img, cfg['image_size']))
        imgs[:, (di * cfg['image_size'][1]):((di + 1) * cfg['image_size'][1])] = img
        mask = imread(os.path.join(cfg['data_path'], mask_fn))[:, :, 0]
        mask = resize(mask, cfg['image_size'])
        c = find_contours(mask, 0.5)[0]
        # disp = disparities[indices[(s, d)], :, :]
        disp = 1.0 / sio.loadmat(os.path.join(cfg['disp_path'], disp_fn))['mat'][0, :, :, 0]
        disp = resize(disp, cfg['image_size'])
        disp = img_as_ubyte(cm.to_rgba(disp)[:, :, :3])
        rr, cc = polygon_perimeter(c[:, 0], c[:, 1])
        outline = np.zeros(cfg['image_size'])
        outline[rr, cc] = 1
        outline = binary_dilation(outline, square(5))
        disp[outline, :] = [255, 255, 255]
        disps[:, (di * cfg['image_size'][1]):((di + 1) * cfg['image_size'][1])] = disp
    full_image[((si * 2) * cfg['image_size'][0]):((si * 2 + 1) * cfg['image_size'][0]), :] = imgs
    full_image[((si * 2 + 1) * cfg['image_size'][0]):((si * 2 + 2) * cfg['image_size'][0]), :] = disps

imsave(cfg['output_path'], full_image)