import os
import re
import glob
import csv
import pandas
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave, imshow
from skimage.color import rgb2hsv

cfg = {
    'object_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/bottom_edge/src_imgs',
    'object_data_path': '/home/tom/research/ICCV2019_depth/data/objects/objects.csv',
    'scenes_path': '/home/tom/research/ICCV2019_depth/data/scenes.csv',
    'image_path': '/media/tom/seagate-8tb-ext4/data/kitti/data_scene_flow/training/image_2',
    'size_range': list(range(0, 15 + 1)) + [-1],
    'value_range': np.linspace(0.0, 1.0, 11).tolist() + [-1],
    'image_fmt': '{scene}_{object}_{value:.1f}_{size:d}.png',
    'mask_fmt': '{scene}_{object}_mask.png',
    'output_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/bottom_edge/images',
}


def generate_object(shape, interior, value, size):
    _shape = shape.copy()
    _interior = interior.copy()

    if value >= 0:
        _shape[:, :, 0:3] = 0.0
        _shape[:, :, 0:3] = value * _interior[:, :, 3, np.newaxis]

    if size >= 0:
        nonzero_rows = np.argwhere(np.any(_interior[:, :, 3], axis=1))
        bottom_row = nonzero_rows[-1][0]
        top_row = bottom_row - size
        _interior[top_row:-1, :, 3] = 0.0
        _shape[:, :, 3] = _shape[:, :, 3] * (1.0 - _interior[:, :, 3])

    return _shape


def get_ground_value(scene, interior, xmin, ymin):
    nonzero_rows = np.argwhere(np.any(interior[:, :, 3], axis=1))
    bottom_row = nonzero_rows[-1][0]
    sample_mask = interior[bottom_row - 1, :, 3] != 0.0

    scene_hsv = rgb2hsv(scene)
    roi_v = scene_hsv[ymin + bottom_row + 1, xmin:(xmin + interior.shape[1]), 2]
    ground_value = np.median(roi_v[sample_mask])

    return ground_value


# Get object image paths
objects = glob.glob(os.path.join(cfg['object_path'], '*_shape.png'))

# Get object data
objects_raw = []
with open(cfg['object_data_path'], 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        objects_raw.append(row)

# Get scenes
scenes = []
with open(cfg['scenes_path'], 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        scenes.append(row)

# Generate images
with open(os.path.join(cfg['output_path'], 'filenames.txt'), 'w') as f:
    value_data = []
    for scene_d in scenes:
        if not int(scene_d['rear']):
            continue
        scene_img = img_as_float(imread(os.path.join(cfg['image_path'], scene_d['#image_2'])))
        scene_name = scene_d['#image_2'].strip('.png')

        for obj_fn in objects:
            # Load object images
            m = re.match('(?P<path>.*)/(?P<name>[a-zA-Z0-9_]*)_shape.png', obj_fn)
            path = m.group('path')
            name = m.group('name')
            print('Scene: {}, object: {}'.format(scene_d['#image_2'], name))
            shape = img_as_float(imread(os.path.join(path, '{}_shape.png'.format(name))))
            interior = img_as_float(imread(os.path.join(path, '{}_interior.png'.format(name))))

            # Find image position
            insert_pos = None
            for entry in objects_raw:
                if entry['#filename'] == '{}.png'.format(name):
                    insert_pos = (
                        int(entry['image_x']) - int(entry['sprite_x']),
                        int(entry['image_y']) - int(entry['sprite_y'])
                    )
            if not insert_pos:
                print('Error: could not find position for object {}. Skipping...'.format(name))
            xmin = insert_pos[0]
            xmax = insert_pos[0] + shape.shape[1]
            ymin = insert_pos[1]
            ymax = insert_pos[1] + shape.shape[0]

            # Find ground value
            ground_value = get_ground_value(scene_img, interior, xmin, ymin)
            value_data.append({
                'scene': scene_name,
                'object': name,
                'ground_value': ground_value,
            })

            # Generate interior mask
            interior_mask = np.zeros((scene_img.shape[0], scene_img.shape[1]), dtype=np.float64)
            interior_mask[ymin:(ymin + interior.shape[0]), xmin:(xmin + interior.shape[1])] = interior[:, :, 3]
            imsave(os.path.join(cfg['output_path'], cfg['mask_fmt'].format(scene=scene_name, object=name)),
                   img_as_ubyte(interior_mask))

            # Generate test images
            for value in cfg['value_range']:
                for size in cfg['size_range']:
                    # print('{}, {}'.format(value, size))
                    obj = generate_object(shape, interior, value, size)

                    output_img = scene_img.copy()

                    alpha = obj[:, :, 3, np.newaxis]
                    output_img[ymin:ymax, xmin:xmax] = \
                        (1 - alpha) * output_img[ymin:ymax, xmin:xmax, 0:3] + alpha * obj[:, :, 0:3]

                    output_fn = cfg['image_fmt'].format(scene=scene_name, object=name, value=value, size=size)
                    imsave(os.path.join(cfg['output_path'], output_fn),
                           img_as_ubyte(output_img))
                    f.write(output_fn + '\n')

                    # plt.clf()
                    # plt.imshow(output_img)
                    # plt.draw()
                    # plt.pause(.001)
        #     break
        # break

df = pandas.DataFrame(value_data)
df.to_csv(os.path.join(cfg['output_path'], 'ground_values.csv'), index=False)

