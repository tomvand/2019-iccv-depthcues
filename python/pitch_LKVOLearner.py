from numpy import load, asarray, reshape, arange, newaxis
from skimage.transform import resize
from sklearn import linear_model
import matplotlib.pyplot as plt
import re
import pandas

cfg = {
    'filenames_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/pitch/filenames.txt',
    'filename_re': '(?P<filename>[0-9]*_[0-9]*)_(?P<y>[-0-9]*).png',
    'disparities_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/pitch/output_LKVOLearner/output.npy',
    'image_shape': (300, 993),
    'output_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/pitch/output_LKVOLearner/pitch.csv'
}

def horizon_y_from_disp(disp):
    yv = []
    dv = []
    for y in range(100, disp.shape[0]):
        for x in range(disp.shape[1] // 2 - 20, disp.shape[1] // 2 + 20):
            yv.append(y)
            dv.append(disp[y, x])
    dv = asarray(dv).reshape(-1, 1)
    yv = asarray(yv).reshape(-1, 1)

    ransac = linear_model.RANSACRegressor(residual_threshold=2, max_trials=500)
    ransac.fit(dv, yv)
    return ransac.predict([[0.00]])[0][0]


disp = load(cfg['disparities_path'], mmap_mode='r')

data = {}
with open(cfg['filenames_path'], 'rt') as f:
    for index, filename in enumerate(f):
        print(filename)
        n = re.match(cfg['filename_re'], filename).group('filename')
        if n not in data:
            data[n] = {'y_true': [], 'y_meas': [], 'offset': -1}
        y_true = int(re.match(cfg['filename_re'], filename).group('y'))
        disp_r = 1.0 / resize(disp[index], cfg['image_shape'])
        y_meas = horizon_y_from_disp(disp_r)
        data[n]['y_true'].append(y_true)
        data[n]['y_meas'].append(y_meas)
        if y_true is 0:
            data[n]['offset'] = y_meas

X = []
Y = []
for d in data.values():
    for (y_true, y_meas) in zip(d['y_true'], d['y_meas']):
        X.append(y_true)
        Y.append(d['offset'] - y_meas)

plt.plot(X, Y, '.', alpha=0.05)
plt.axis('equal')
plt.grid(True)
plt.ylim([min(X), max(X)])
plt.xlabel('True horizon offset [px]')
plt.ylabel('Measured horizon offset [px]')
plt.show()


data_export = []
for entry in data.values():
    row = {}
    for y_true, y_meas in zip(entry['y_true'], entry['y_meas']):
        row[y_true] = entry['offset'] - y_meas
    data_export.append(row)
df = pandas.DataFrame(data_export)
df.to_csv(cfg['output_path'], index=False)