# Simple file for experimentation
import os
import csv
import re

from numpy import newaxis, arange, zeros_like
from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.transform import resize

cfg = {
    'scenes_path': '/home/tom/Dropbox/PhD/Experiments/ICCV2019_depth/data/other_objects',
    'object_path': '/home/tom/Dropbox/PhD/Experiments/ICCV2019_depth/data/other_objects',
    'kitti_path': '/media/tom/seagate-8tb-ext4/data/kitti/data_scene_flow/training/image_2',
    'output_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/other_objects',
    'horizon_y': 174,
    'scales': arange(10, 31) / 10.0
}

# CSV reading
# Load scenes
scenes = []
with open(os.path.join(cfg['scenes_path'], 'scenes.csv'), 'r') as f:
    for line in f:
        scenes.append(line.strip('\n'))

# Load objects
objects = []
with open(os.path.join(cfg['object_path'], 'objects.csv'), 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        objects.append(row)

print(scenes)
print(objects)


# Create output folders
if not os.path.exists(os.path.join(cfg['output_path'], 'images')):
    os.makedirs(os.path.join(cfg['output_path'], 'images'))


# Generate scenes
def insert_scaled_object(img_scene, object, rel_dist, apply_scale=True, apply_position=True):
    cx = img_scene.shape[1] / 2.0
    cy = cfg['horizon_y']
    img_obj = img_as_float(imread(os.path.join(cfg['object_path'], object['#filename'])))
    # img_obj_mask = img_as_float(imread(os.path.join(cfg['object_path'], object['#filename'].replace('.png', '_mask.png'))))
    img_obj_mask = img_obj[:, :, 3]
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

    alpha_mask = img_obj_mask[:, :, newaxis] #[:, :, 3, newaxis]
    img_mask = zeros_like(img_scene)
    img_mask[rt:rb, cl:cr, :] = alpha_mask

    return img_scene, img_mask


with open(os.path.join(cfg['output_path'], 'filenames.txt'), 'w') as f:
    for scene in scenes:
        img_back = img_as_float(imread(os.path.join(cfg['kitti_path'], scene)))
        for object in objects:
            for scale in cfg['scales']:
                scene_name = re.match('(.*).png', scene).group(1)
                object_name = re.match('(.*).png', object['#filename']).group(1)
                name = scene_name + '_' + object_name + '_' + str(scale)
                print(name)

                # Normal
                img_scene = img_back.copy()
                img_scene, img_mask = insert_scaled_object(img_scene, object, scale)
                imsave(os.path.join(cfg['output_path'], 'images', name + '.png'), img_scene)
                imsave(os.path.join(cfg['output_path'], 'images', name + '_mask.png'), img_mask)
                f.write(os.path.join('images', name + '.png\n'))
