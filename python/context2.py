import pickle
import matplotlib.pyplot as plt

from pandas import DataFrame
from scipy.stats import friedmanchisquare

cfg = {
    'pickle_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/pos_vs_scale/output/depths.p',
}

results = pickle.load(open(cfg['pickle_path'], 'rb'))

# Create dataframe
table = []
for scene_name, scene in results['normal'].items():
    for object_name, depths in scene.items():
        row = {
            'scene': scene_name,
            'object': object_name,
        }
        for true_depth in sorted(depths.keys()):
            meas_depth = depths[true_depth]
            row[true_depth] = meas_depth
        table.append(row)
df = DataFrame(table)

def sort_scenes(df):
    avg = df.mean(axis='columns').sort_values()
    # avg = df.rank().mean(axis='columns').sort_values()
    return df.reindex(index=avg.index)

# Show measured depth at 1.0 true depth for objects across scenes
pivot = df.pivot(index='scene', columns='object', values=1.0)
pivot.plot()

# Test for difference between scenes with rearright car
pivot2 = pivot.filter(regex='.*_rearright[0-9]*', axis='columns')
pivot2 = pivot2.dropna(axis='index', how='any')
pivot2 = sort_scenes(pivot2)
pivot2.plot(legend=False)
chi2, p = friedmanchisquare(*pivot2.T.values) # T verified
plt.title(p)

# Test for difference between scenes with rear car
pivot2b = pivot.filter(regex='.*_rear[0-9]+', axis='columns')
pivot2b = pivot2b.dropna(axis='index', how='any')
pivot2b = sort_scenes(pivot2b)
pivot2b.plot(legend=False)
chi2, p = friedmanchisquare(*pivot2b.T.values) # T verified
plt.title(p)

# Sort by average depth to make scene-depth correlation more clear
pivot3 = sort_scenes(pivot)
pivot3.plot(legend=False)