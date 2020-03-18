import os
import matplotlib.pyplot as plt

from numpy import load, abs
from skimage.transform import resize

cfg = {
    'data_path': '/home/tom/Dropbox/PhD/Experiments/ICCV2019_depth/simple_features',
}

img_rgb = plt.imread(os.path.join(cfg['data_path'], '000082_10_rgb.png'))
img_sem = plt.imread(os.path.join(cfg['data_path'], '000082_10_sem.png'))

disp_rgb = resize(load(os.path.join(cfg['data_path'], '000082_10_rgb_disp.npy')), img_rgb.shape, preserve_range=True)[:, :, 1]
disp_sem = resize(load(os.path.join(cfg['data_path'], '000082_10_sem_disp.npy')), img_sem.shape, preserve_range=True)[:, :, 1]

disp_diff = disp_rgb - disp_sem

plt.figure()
plt.subplot(221)
plt.imshow(img_rgb)
plt.subplot(223)
plt.imshow(img_sem)
plt.subplot(222)
plt.imshow(disp_rgb)
plt.subplot(224)
plt.imshow(disp_sem)

plt.figure()
plt.imshow(abs(disp_diff * disp_diff.shape[1]))
plt.colorbar()
plt.title('Disparity error [px]')

plt.show()