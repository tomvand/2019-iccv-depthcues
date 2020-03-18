import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import csv

cfg = {
    'disp_path': '/media/tom/seagate-8tb-ext4/data/kitti_derived/monodepth_disparities_kitti/disparities_pp.npy',
    'output_path': '/home/tom/Dropbox/PhD/Experiments/ICCV2019_depth/matlab/pitch_fit.csv',
    'index': 0,
}

disp = np.load(cfg['disp_path'], mmap_mode='r')
disp = disp[cfg['index']]
disp = disp * disp.shape[1]

yv = []
dv = []
for y in range(100, disp.shape[0]):
    for x in range(disp.shape[1] // 2 - 20, disp.shape[1] // 2 + 20):
        yv.append(y)
        dv.append(disp[y, x])
dv = np.asarray(dv).reshape(-1, 1)
yv = np.asarray(yv).reshape(-1, 1)

ransac = linear_model.RANSACRegressor(residual_threshold=2, max_trials=500)
ransac.fit(dv, yv)

dq = np.array([[0], [25]])
yq = ransac.predict(dq)

plt.plot(dv, yv, '.')
plt.plot(dq, yq)
plt.gca().invert_yaxis()
plt.xlabel('Disparity [px]')
plt.ylabel('y [px]')

with open(cfg['output_path'], 'wt') as f:
    w = csv.writer(f)
    w.writerow(dq.T.tolist()[0])
    w.writerow(yq.T.tolist()[0])
    for d, y in zip(dv.flatten(), yv.flatten()):
        w.writerow([d, y])