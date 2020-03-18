import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm

cfg = {
    'csv_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/bottom_edge/output/dist.csv',
    'output_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/bottom_edge/output/dist_out.csv',
}

data = pandas.read_csv(cfg['csv_path'])

# Collect reference distances
dist_ref_table = pandas.pivot_table(
    data[(data['value'] == -1) & (data['size'] == -1)],
    index=['scene', 'object'],
    values='dist')
data['dist_ref'] = data.apply(lambda r: dist_ref_table.loc[(r['scene'], r['object'])], axis=1)

# Calculate auxiliary values
data['value_diff'] = data['value'] - data['ground_value']
data['dist_diff'] = data['dist'] - data['dist_ref']
data['dist_diff_ratio'] = data['dist_diff'] / data['dist_ref']

# Influence of value for 15px edge
p = pandas.pivot_table(
    data[(data['size'] == 15) & (data['value'] != -1.0)],
    index=['scene', 'object'],
    columns=['value'],
    values=['value_diff', 'ground_value', 'dist', 'dist_diff', 'dist_diff_ratio'])
# X = p['value_diff'].values
X = p.columns.levels[1].values # values
Y = p['dist_diff_ratio'].values * 100
plt.figure()
plt.plot(X.T, Y.T)
plt.xlabel('Bottom edge value (@15px)')
plt.ylabel('Distance error [%]')

Ymean = np.mean(Y, axis=0)
Ystd = np.std(Y, axis=0)
plt.figure()
plt.fill_between(X, Ymean - Ystd, Ymean + Ystd, alpha=0.5)
plt.plot(X, Ymean)
plt.xlabel('Bottom edge value (@15px)')
plt.ylabel('Distance error [%]')


# Influence of thickness for 0.0 value
p2 = pandas.pivot_table(
     data[(data['value'] == 0.0) & (data['size'] != -1)],
     index=['scene', 'object'],
     columns='size',
     values=['dist', 'dist_diff', 'dist_diff_ratio'])
X = p2.columns.levels[1].values
Y = p2['dist_diff_ratio'] * 100
plt.figure()
plt.plot(X, Y.T)
plt.xlabel('Bottom edge thickness [px] (@value=0.0)')
plt.ylabel('Distance error [%]')

Ymean = np.mean(Y, axis=0)
Ystd = np.std(Y, axis=0)
plt.figure()
plt.fill_between(X, Ymean - Ystd, Ymean + Ystd, alpha=0.5)
plt.plot(X, Ymean)
plt.xlabel('Bottom edge thickness [px] (@value=0.0)')
plt.ylabel('Distance error [%]')

# Combined value/size
p3 = pandas.pivot_table(
    data,  # data[(data['value'] >= 0) & (data['size'] >= 0)],
    index='value',
    columns='size',
    values='dist_diff_ratio'
)
c = p3.columns.tolist()
p3 = p3[c[1:] + c[:1]]
# r = p3.index.tolist()
# p3 = p3.loc[r[1:] + r[:1], :]
p3 = p3.rename(columns={-1: 'full shape'}, index={-1.0: 'real texture'})
plt.figure()
cm = matplotlib.cm.get_cmap('RdBu')
plt.imshow(p3.values * 100, origin='lower', cmap=cm, vmin=-100, vmax=100)
plt.plot([-0.5, len(p3.columns) - 0.5], [0.5, 0.5], 'k')
plt.plot([len(p3.columns) - 1.5, len(p3.columns) - 1.5], [-0.5, len(p3.index) - 0.5], 'k')
plt.xlabel('size')
plt.gca().set_xticks(np.arange(len(p3.columns)))
plt.gca().set_xticklabels(p3.columns.tolist())
plt.gca().get_xticklabels()[-1].set_rotation(90)
plt.ylabel('value')
plt.gca().set_yticks(np.arange(len(p3.index)))
plt.gca().set_yticklabels(p3.index.tolist())
cbar = plt.colorbar()
cbar.ax.set_ylabel('Distance error [%]')
cs = plt.contour(np.arange(16), np.arange(11) + 1, 100 * p3.values[1:, :-1],
                 levels=np.arange(100, step=20),
                 colors='black',
                 linewidths=0.5)
plt.clabel(cs, fmt='%.0f%%')

p3.to_csv(cfg['output_path'])