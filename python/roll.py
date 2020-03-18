import re
import matplotlib.pyplot as plt
import pandas

from numpy import load, linspace, deg2rad, rad2deg, mean, zeros, std, asarray
from skimage.transform import hough_line, hough_line_peaks, resize

cfg = {
    'disparities_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/roll/output/disparities_pp.npy',
    'test_disp': (0.03, 0.031),
    'roll_range': linspace(-30.0, 30.0, 100),
    'filenames_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/roll/filenames.txt',
    'filename_re': '(?P<filename>[0-9]*_[0-9]*)_(?P<angle>[-0-9]*).png',
    'image_shape': (244, 807),
    'output_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/roll/output/roll.csv'
}

def roll_from_disp(disp):
    h, theta, d = hough_line((cfg['test_disp'][0] < disp) & (disp < cfg['test_disp'][1]), theta=linspace(deg2rad(90 - 30), deg2rad(90 + 30), 250))
    _, angles, dists = hough_line_peaks(h, theta, d, num_peaks=1)
    return 90 - rad2deg(angles[0])


disp = load(cfg['disparities_path'], mmap_mode='r')
X = []
Y = []
data_export = {}
with open(cfg['filenames_path'], 'rt') as f:
    for index, filename in enumerate(f):
        print(filename)
        m = re.match(cfg['filename_re'], filename)
        angle_true = float(m.group('angle'))
        image = m.group('filename')
        disp_r = resize(disp[index], cfg['image_shape'])
        angle_meas = roll_from_disp(disp_r)
        X.append(angle_true)
        Y.append(angle_meas)
        if image not in data_export:
            data_export[image] = {}
        data_export[image][angle_true] = angle_meas
df = pandas.DataFrame(data_export).T
df.to_csv(cfg['output_path'], index=False)

plt.figure()
plt.plot(X, Y, '.', alpha=0.05)
plt.xlabel('True angle [deg]')
plt.ylabel('Measured angle [deg]')
plt.ylim(-15, 15)
plt.grid()

xq = sorted(set(X))
ymean = []
ystd = []
for xx in xq:
    angle_x = []
    for x, y in zip(X, Y):
        if x == xx:
            angle_x.append(y)
    ymean.append(mean(angle_x))
    ystd.append(std(angle_x))
ymean = asarray(ymean)
ystd = asarray(ystd)

plt.figure()
plt.fill_between(xq, ymean - ystd, ymean + ystd, alpha=0.5)
plt.plot(xq, ymean)
plt.grid()
plt.xlabel('True roll [deg]')
plt.ylabel('Measured roll [deg]')
