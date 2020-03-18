import pickle
import matplotlib.pyplot as plt

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

from numpy import asarray, nan, nanmean, nanstd, repeat, isnan, zeros_like


cfg = {
    'pickle_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/pos_vs_scale/output/depths.p',
}


# Load results
results = pickle.load(open(cfg['pickle_path'], 'rb'))

# Prepare scatterplot data
plot_data = {}
plot_data['x'] = sorted(list(next(iter(next(iter(next(iter(results.values())).values())).values())).keys())) # X data is shared by all results!
plot_data['y'] = {}
for set, entries in results.items():
    plot_data['y'][set] = []
    for scene in entries.values():
        for object in scene.values():
            y = []
            baseline_depth = object[1.0]
            for x in plot_data['x']:
                if x in object:
                    y.append(object[x] / baseline_depth)
                else:
                    y.append(nan)
            if not any(isnan(y)): # Skip incomplete data
                plot_data['y'][set].append(y)

# Show scatterplot
plt.figure()
offset = 0.0
for set in plot_data['y']:
    x = asarray(plot_data['x']) + offset
    y = asarray(plot_data['y'][set]).T
    x = repeat(x, y.shape[1])
    y = y.flatten()
    plt.scatter(x, y, alpha=0.03, marker='.', label=set)
    offset += 0.02
plt.plot([1, 3], [1, 3], 'k--')
plt.legend()

# Show means
plt.figure()
for set in plot_data['y']:
    x = asarray(plot_data['x'])
    y = asarray(plot_data['y'][set]).T
    # ymean = nanmean(y, axis=1)

    ymin = zeros_like(x)
    ymean = zeros_like(x)
    ymax = zeros_like(x)
    for i in range(y.shape[0]):
        ci = bs.bootstrap(y[i,:], stat_func=bs_stats.mean, alpha=0.05)
        ymin[i] = ci.lower_bound
        ymean[i] = ci.value
        ymax[i] = ci.upper_bound

    plt.fill_between(x,ymin, ymax, alpha=0.5)
    plt.plot(x, ymean, label=set)
plt.plot([1.0, 3.0], [1.0, 3.0], 'k--', label='expected')
plt.legend()
plt.xlabel('True distance [-]')
plt.ylabel('Predicted distance [-]')

# Show standard deviations
plt.figure()
for set in plot_data['y']:
    x = asarray(plot_data['x'])
    y = asarray(plot_data['y'][set]).T
    ymean = nanmean(y, axis=1)
    ystd = nanstd(y, axis=1)

    plt.fill_between(x, ymean - ystd, ymean + ystd, alpha=0.5)
    plt.plot(x, ymean, label=set)
plt.plot([1.0, 3.0], [1.0, 3.0], 'k--', label='expected')
plt.legend(loc='upper left')
plt.xlabel('True distance [-]')
plt.ylabel('Predicted distance [-]')

plt.show()