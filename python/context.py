import pickle
import re
import matplotlib.pyplot as plt
from cycler import cycler

from numpy import full, nan, array

cfg = {
    'pickle_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/pos_vs_scale/output/depths.p',
}

results = pickle.load(open(cfg['pickle_path'], 'rb'))
scenes = sorted(results['normal'].keys())

def scene_index(scene_name):
    m = re.match('(?P<scene>[0-9]*)_(?P<frame>[0-9]*)', scene_name)
    return float(m.group('scene')) + (0.0 if m.group('frame') == '10' else 0.5)

x = []
for scene_name in scenes:
    x.append(scene_index(scene_name))

# Collect depth per object
data = {}
for scene_name, scene in results['normal'].items():
    for object_name, depths in scene.items():
        if object_name not in data:
            data[object_name] = full(len(scenes), nan)
        data[object_name][scenes.index(scene_name)] = depths[1.0]

# Plot
plt.figure()
plt.rc('axes', prop_cycle=(cycler('marker', ['o', 's', 'd']) *
                           cycler('linestyle', ['-', '--', ':']) *
                           plt.rcParamsDefault['axes.prop_cycle']))
for object, depths in data.items():
    plt.plot(depths)
plt.xlabel('Scene index')
plt.ylabel('Depth estimate [-]')
