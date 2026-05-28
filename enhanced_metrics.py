# coding=utf-8
"""
Enhanced Metrics Module
Adds comprehensive evaluation metrics beyond just AUC
"""
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def calculate_all_metrics(target2d, groundtruth, threshold=None):
    """
    Calculate comprehensive evaluation metrics
    
    :param target2d: the 2D anomaly component  
    :param groundtruth: the groundtruth
    :param threshold: threshold for binary classification (if None, uses optimal)
    :return: dict with all metrics
    """
    rows, cols = groundtruth.shape
    label = groundtruth.transpose().reshape(1, rows * cols)
    result = np.zeros((1, rows * cols))
    
    for i in range(rows * cols):
        result[0, i] = np.linalg.norm(target2d[:, i])
    
    # Normalize result to [0, 1]
    result_norm = (result - result.min()) / (result.max() - result.min() + 1e-10)
    
    # Find optimal threshold if not provided
    if threshold is None:
        fpr, tpr, thresholds = metrics.roc_curve(label.transpose(), result_norm.transpose())
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]
    
    # Binary predictions
    predictions = (result_norm >= threshold).astype(int)
    y_true = label.flatten()
    y_pred = predictions.flatten()
    
    # Calculate metrics
    metrics_dict = {
        'AUC': metrics.roc_auc_score(y_true, result_norm.flatten()),
        'Precision': metrics.precision_score(y_true, y_pred, zero_division=0),
        'Recall': metrics.recall_score(y_true, y_pred, zero_division=0),
        'F1_Score': metrics.f1_score(y_true, y_pred, zero_division=0),
        'Accuracy': metrics.accuracy_score(y_true, y_pred),
        'Optimal_Threshold': float(threshold),
        'True_Positives': int(np.sum((y_true == 1) & (y_pred == 1))),
        'False_Positives': int(np.sum((y_true == 0) & (y_pred == 1))),
        'True_Negatives': int(np.sum((y_true == 0) & (y_pred == 0))),
        'False_Negatives': int(np.sum((y_true == 1) & (y_pred == 0))),
    }
    
    # Confusion matrix
    metrics_dict['Confusion_Matrix'] = metrics.confusion_matrix(y_true, y_pred).tolist()
    
    return metrics_dict, result_norm


def plot_confusion_matrix(metrics_dict, save_path=None):
    """Plot confusion matrix"""
    cm = np.array(metrics_dict['Confusion_Matrix'])
    plt.figure(figsize=(8, 6))
    
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Anomaly'])
    else:
        # Fallback to matplotlib if seaborn not available
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Normal', 'Anomaly'])
        plt.yticks(tick_marks, ['Normal', 'Anomaly'])
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return plt.gcf()


def print_metrics_report(metrics_dict):
    """Print a formatted metrics report"""
    print("\n" + "="*50)
    print("COMPREHENSIVE EVALUATION METRICS")
    print("="*50)
    print(f"AUC (Area Under ROC Curve):     {metrics_dict['AUC']:.4f}")
    print(f"Precision:                       {metrics_dict['Precision']:.4f}")
    print(f"Recall (Sensitivity):            {metrics_dict['Recall']:.4f}")
    print(f"F1-Score:                        {metrics_dict['F1_Score']:.4f}")
    print(f"Accuracy:                        {metrics_dict['Accuracy']:.4f}")
    print(f"Optimal Threshold:               {metrics_dict['Optimal_Threshold']:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {metrics_dict['True_Negatives']}")
    print(f"  False Positives: {metrics_dict['False_Positives']}")
    print(f"  False Negatives: {metrics_dict['False_Negatives']}")
    print(f"  True Positives:  {metrics_dict['True_Positives']}")
    print("="*50 + "\n")

