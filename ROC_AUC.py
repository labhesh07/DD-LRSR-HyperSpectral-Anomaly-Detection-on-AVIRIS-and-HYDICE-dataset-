# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import HyperProTool as hyper


def ROC_AUC(target2d, groundtruth):
    """
    A10: ROC curve and AUC — evaluate detection performance against ground truth.
    Anomaly Scoring and Detection module.
    
    Methodology Reference: Section 3.9 - Anomaly Scoring and Detection
    - Equation (18): Score(i) = ||D_an s_i||_2
    - Equation (19): Binary anomaly map using threshold τ
    
    :param target2d: the 2D anomaly component (D_an S from decomposition)
    :param groundtruth: the groundtruth labels for evaluation
    :return: auc: the AUC value for performance evaluation
    """
    rows, cols = groundtruth.shape
    label = groundtruth.transpose().reshape(1, rows * cols)
    result = np.zeros((1, rows * cols))
    # A10: Anomaly scoring — Equation (18) Score(i) = ||D_an·s_i||_2, then ROC/AUC vs ground truth
    # Section 3.9: Anomaly Scoring
    # Equation (18): Score(i) = ||D_an s_i||_2
    for i in range(rows * cols):
        result[0, i] = np.linalg.norm(target2d[:, i])

    # result = hyper.hypernorm(result, "minmax")
    # A10: Build ROC from pixel scores; AUC = area under ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(label.transpose(), result.transpose())
    auc = metrics.auc(fpr, tpr)
    plt.figure(2)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    return auc
