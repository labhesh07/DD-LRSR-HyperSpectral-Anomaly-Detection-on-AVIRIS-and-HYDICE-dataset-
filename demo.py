

import numpy as np

import scipy.io as sio
from LRSR import LRSR

from dic_constr import dic_constr
from result_show import result_show
from ROC_AUC import ROC_AUC
import HyperProTool as hyper

# data pre-precessing
# A1: Hyperspectral cube captured — load AVIRIS cube (H×W×B) from disk; crop spatial ROI
data = sio.loadmat("Sandiego.mat")
data3d = np.array(data["Sandiego"], dtype=float)
data3d = data3d[0:100, 0:100, :]
# A2.1: Bad-band removal — drop noisy/water-vapor channels along spectral axis (224 → ~186 bands)
remove_bands = np.hstack((range(6), range(32, 35, 1), range(93, 97, 1), range(106, 113), range(152, 166), range(220, 224)))
data3d = np.delete(data3d, remove_bands, axis=2)
rows, cols, bands = data3d.shape
groundtruthfile = sio.loadmat("PlaneGT.mat")
groundtruth = np.array(groundtruthfile["PlaneGT"])
rows, cols, bands = data3d.shape

# background and anomaly dictionary construction
# (A2–A8 run inside dic_constr after A2.1; starts with B×N reshape, then PCA, windows, K-means, …)
data2d, bg_dic, tg_dic,bg_dic_label, tg_dic_label = dic_constr(data3d, groundtruth, 3, 10, 10, 0.05, 200)

# low rank and sparse representaion
# A9: Augmented Lagrangian Method (ALM) — LRSR decomposition X = D_bg·Z + D_an·S + E
Z, E, S = LRSR(bg_dic, tg_dic, data2d, 0.001, 0.01)

# result visualization
background2d, target2d = result_show(bg_dic, tg_dic, Z, S, E, rows, cols, bands, bg_dic_label, tg_dic_label)

# ROC curve show
# A10: ROC / AUC evaluation — anomaly scores vs ground-truth labels
auc = ROC_AUC(target2d, groundtruth)
print("The AUC is: {0}".format(auc))


