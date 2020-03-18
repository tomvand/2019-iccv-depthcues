import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate.interpnd import LinearNDInterpolator
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn import linear_model

cfg = {
    'disp_path': '/media/tom/seagate-8tb-ext4/data/kitti/data_scene_flow/training/disp_noc_0',
    'disp_fmt': '{:06d}_10.png',
    'monodepth_disp_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/monodepth_disparities_kitti/disparities_pp.npy',
    'output_path': '/home/tom/Dropbox/PhD/Experiments/ICCV2019_depth/matlab/pitch_in_kitti.csv'
}

def interp_disp(disp):
    # Based on monodepth/utils/evaluation_utils.py
    J, I = np.meshgrid(np.arange(disp.shape[1]), np.arange(disp.shape[0]))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    d = disp[IJ[:, 0], IJ[:, 1]]
    ij = IJ[d != 0, :]
    d = d[d != 0]
    intp = LinearNDInterpolator(ij, d, fill_value=0.)
    disp_i = intp(IJ).reshape(disp.shape)
    return disp_i

def horizon_y_from_disp(disp):
    yv = []
    dv = []
    for y in range(160, disp.shape[0]):
        for x in range(disp.shape[1] // 2 - 20, disp.shape[1] // 2 + 20):
            if disp[y, x] != 0:
                yv.append(y)
                dv.append(disp[y, x])
    dv = np.asarray(dv).reshape(-1, 1)
    yv = np.asarray(yv).reshape(-1, 1)

    ransac = linear_model.RANSACRegressor(residual_threshold=2, max_trials=2000)
    ransac.fit(dv, yv)
    return ransac.predict([[0.00]])[0][0]


hor_true = []
shapes = []
for i in range(200):
    fn = cfg['disp_fmt'].format(i)
    disp = imread(os.path.join(cfg['disp_path'], fn)) / 256.0
    shapes.append(disp.shape)
    for i in range(10):
        hor_y = horizon_y_from_disp(disp)
        print(hor_y)
        hor_true.append(hor_y)

hor_est = []
disp = np.load(cfg['monodepth_disp_path'], mmap_mode='r')
for i in range(200):
    shape = shapes[i]
    d = resize(disp[i], shape) * shape[1]
    for i in range(10):
        y = horizon_y_from_disp(d)
        print(y)
        hor_est.append(y)

plt.plot(hor_true, hor_est, '.')
plt.xlim([160, 190])
plt.ylim([160, 190])
plt.xlabel('True horizon [px]')
plt.ylabel('Est. horizon [px]')


# Regression
hor_true_a = np.array(hor_true)
hor_est_a = np.array(hor_est)
inliers = (hor_true_a >= 160) & (hor_true_a <= 190) & (hor_est_a >= 160) & (hor_est_a <= 190)
hor_true_in = hor_true_a[inliers]
hor_est_in = hor_est_a[inliers]

regr = linear_model.LinearRegression()
regr.fit(hor_true_in.reshape(-1, 1), hor_est_in.reshape(-1, 1))
x_pred = np.array([[160], [190]])
y_pred = regr.predict(x_pred)
plt.plot(x_pred, y_pred)

with open(cfg['output_path'], 'wt') as f:
    w = csv.writer(f)
    w.writerow(x_pred.flatten())
    w.writerow(y_pred.flatten())
    w.writerows(np.vstack([hor_true_in, hor_est_in]).T)