import os
import numpy as np

from skimage import img_as_float
from skimage.io import imread, imshow
from sklearn import linear_model

cfg = {
    'disp_path': '/media/tom/seagate-8tb-ext4/data/kitti/data_scene_flow/training/disp_noc_0',
    'sem_path': '/media/tom/seagate-8tb-ext4/data/kitti/data_semantics/training/semantic',
    'filename_fmt': '{:06d}_10.png',
}

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351


for i in range(200):
    fn = cfg['filename_fmt'].format(i)
    disp = imread(os.path.join(cfg['disp_path'], fn)) / 256.0
    mask = disp > 0
    f = width_to_focal[disp.shape[1]]
    depth = np.zeros_like(disp)
    depth += img_as_float(mask) * f * 0.54 / (disp + 0.0000001)
    XYZ = np.zeros((disp.shape[0], disp.shape[1], 3))
    for x in range(depth.shape[1]):
        for y in range(depth.shape[0]):
            u, v = x - depth.shape[1] / 2, y - depth.shape[0] / 2
            X = u / f * depth[y, x]
            Y = v / f * depth[y, x]
            XYZ[y, x, :] = [X, Y, depth[y, x]]

    sem = imread(os.path.join(cfg['sem_path'], fn))
    Y = []
    Z = []
    for x in range(depth.shape[1]):
        for y in range(depth.shape[0]):
            if sem[y, x] == 7 and XYZ[y, x, 2] != 0:
                # Road surface
                Y.append(XYZ[y, x, 1])
                Z.append(XYZ[y, x, 2])

    ransac = linear_model.RANSACRegressor(residual_threshold=2, max_trials=2000)
    ransac.fit(np.array(Z).reshape(-1, 1), np.array(Y).reshape(-1, 1))
    print(ransac.predict([[0.00]])[0][0])

    # break

