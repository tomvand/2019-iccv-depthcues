import pickle
import pandas
import os

from numpy import nan, isnan

cfg = {
    'pickle_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/pos_vs_scale/output/depths.p',
    'output_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/pos_vs_scale/csv',
}


# Load results
results = pickle.load(open(cfg['pickle_path'], 'rb'))

# Prepare scatterplot data
data = {}
X = sorted(list(next(iter(next(iter(next(iter(results.values())).values())).values())).keys())) # X data is shared by all results!
for set, entries in results.items():
    data[set] = []
    for scene in entries.values():
        for object in scene.values():
            y = {}
            for x in X:
                if x in object:
                    y[x] = object[x]
                else:
                    y[x] = (nan)
            data[set].append(y)

df = {}
for set in data.keys():
    df[set] = pandas.DataFrame(data[set])

for set, d in df.items():
    d.to_csv(os.path.join(cfg['output_path'], set + '.csv'), index=False)