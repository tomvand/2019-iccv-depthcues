# Simple file for experimentation
import os
import csv
import re

from numpy import newaxis, arange, zeros_like
from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.transform import resize

cfg = {
    'data_path': '../data',
    'object_path': '../data/objects',
    'kitti_path': '/media/tom/seagate-8tb-ext4/data/kitti/data_scene_flow/training/image_2',
    'output_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/pos_vs_scale',
    'horizon_y': 174,
    'scales': arange(10, 31) / 10.0
}

# CSV reading
# Load scenes
scenes = []
with open(os.path.join(cfg['data_path'], 'scenes.csv'), 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        scenes.append(row)

# Load objects
objects_raw = []
with open(os.path.join(cfg['object_path'], 'objects.csv'), 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        objects_raw.append(row)
# Sort objects by type
objects = {}
for key in scenes[0].keys():
    objects[key] = []
del objects['#image_2']
for row in objects_raw:
    for key in objects.keys():
        object_class = re.match('[a-zA-Z]*_([a-zA-Z]*)[0-9]*.png', row['#filename'])
        if object_class and key == object_class.group(1):
            objects[key].append(row)

print(scenes)
print(objects)


# Create output folders
if not os.path.exists(os.path.join(cfg['output_path'], 'normal')):
    os.makedirs(os.path.join(cfg['output_path'], 'normal'))
if not os.path.exists(os.path.join(cfg['output_path'], 'scale_only')):
    os.makedirs(os.path.join(cfg['output_path'], 'scale_only'))
if not os.path.exists(os.path.join(cfg['output_path'], 'pos_only')):
    os.makedirs(os.path.join(cfg['output_path'], 'pos_only'))


# Generate scenes
def insert_scaled_object(img_scene, object, rel_dist, apply_scale=True, apply_position=True):
    cx = img_scene.shape[1] / 2.0
    cy = cfg['horizon_y']
    img_obj = img_as_float(imread(os.path.join(cfg['object_path'], object['#filename'])))
    img_obj_mask = img_as_float(imread(os.path.join(cfg['object_path'], object['#filename'].replace('.png', '_mask.png'))))
    sx = float(object['sprite_x'])
    sy = float(object['sprite_y'])
    u = float(object['image_x']) - cx
    v = float(object['image_y']) - cy

    scale = 1. / rel_dist
    if apply_scale:
        img_obj = resize(img_obj, (int(scale * img_obj.shape[0]), int(scale * img_obj.shape[1])))
        img_obj_mask = resize(img_obj_mask, (int(scale * img_obj_mask.shape[0]), int(scale * img_obj_mask.shape[1])))
        sx *= scale
        sy *= scale

    if apply_position:
        u *= scale
        v *= scale

    cl = int(cx + u - sx)
    cr = int(cl + img_obj.shape[1])
    rt = int(cy + v - sy)
    rb = int(rt + img_obj.shape[0])
    alpha = img_obj[:, :, 3, newaxis]
    img_scene[rt:rb, cl:cr] = (1 - alpha) * img_scene[rt:rb, cl:cr, 0:3] + alpha * img_obj[:, :, 0:3]

    alpha_mask = img_obj_mask[:, :, 3, newaxis]
    img_mask = zeros_like(img_scene)
    img_mask[rt:rb, cl:cr, :] = alpha_mask

    return img_scene, img_mask


with open(os.path.join(cfg['output_path'], 'filenames.txt'), 'w') as f:
    for scene in scenes:
        img_back = img_as_float(imread(os.path.join(cfg['kitti_path'], scene['#image_2'])))
        for key in objects.keys():
            if scene[key] == '1':
                for object in objects[key]:
                    for scale in cfg['scales']:
                        scene_name = re.match('(.*).png', scene['#image_2']).group(1)
                        object_name = re.match('(.*).png', object['#filename']).group(1)
                        name = scene_name + '_' + object_name + '_' + str(scale)
                        print(name)

                        # Normal
                        img_scene = img_back.copy()
                        img_scene, img_mask = insert_scaled_object(img_scene, object, scale)
                        imsave(os.path.join(cfg['output_path'], 'normal', name + '.png'), img_scene)
                        imsave(os.path.join(cfg['output_path'], 'normal', name + '_mask.png'), img_mask)
                        f.write(os.path.join('normal', name + '.png\n'))

                        # Scale only
                        img_scene = img_back.copy()
                        img_scene, img_mask = insert_scaled_object(img_scene, object, scale, apply_position=False)
                        imsave(os.path.join(cfg['output_path'], 'scale_only', name + '.png'), img_scene)
                        imsave(os.path.join(cfg['output_path'], 'scale_only', name + '_mask.png'), img_mask)
                        f.write(os.path.join('scale_only', name + '.png\n'))

                        # Position only
                        img_scene = img_back.copy()
                        img_scene, img_mask = insert_scaled_object(img_scene, object, scale, apply_scale=False)
                        imsave(os.path.join(cfg['output_path'], 'pos_only', name + '.png'), img_scene)
                        imsave(os.path.join(cfg['output_path'], 'pos_only', name + '_mask.png'), img_mask)
                        f.write(os.path.join('pos_only', name + '.png\n'))