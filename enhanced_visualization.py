# coding=utf-8
"""
Enhanced Visualization Module
Provides improved visualizations with save functionality
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import os


def plot_anomaly_heatmap(target2d, groundtruth, rows, cols, save_path=None, cmap='hot'):
    """
    Create a heatmap visualization of anomaly scores
    
    :param target2d: 2D anomaly component
    :param groundtruth: groundtruth labels
    :param rows: number of rows
    :param cols: number of cols
    :param save_path: path to save figure
    :param cmap: colormap to use
    """
    # Calculate anomaly scores
    anomaly_scores = np.zeros((rows * cols,))
    for i in range(rows * cols):
        anomaly_scores[i] = np.linalg.norm(target2d[:, i])
    
    # Normalize
    anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-10)
    anomaly_map = anomaly_scores.reshape(rows, cols)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Anomaly score heatmap
    im1 = axes[0].imshow(anomaly_map, cmap=cmap, interpolation='nearest')
    axes[0].set_title('Anomaly Score Heatmap', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Groundtruth overlay
    im2 = axes[1].imshow(groundtruth, cmap='gray', interpolation='nearest')
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Anomaly heatmap saved to: {save_path}")
    
    return fig


def plot_overlay_comparison(target2d, groundtruth, rows, cols, save_path=None):
    """
    Create overlay comparison of predictions and groundtruth
    
    :param target2d: 2D anomaly component
    :param groundtruth: groundtruth labels
    :param rows: number of rows
    :param cols: number of cols
    :param save_path: path to save figure
    """
    # Calculate anomaly scores
    anomaly_scores = np.zeros((rows * cols,))
    for i in range(rows * cols):
        anomaly_scores[i] = np.linalg.norm(target2d[:, i])
    
    # Normalize and threshold
    anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-10)
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(groundtruth.flatten(), anomaly_scores)
    optimal_idx = np.argmax(tpr - fpr)
    threshold = thresholds[optimal_idx]
    predictions = (anomaly_scores >= threshold).astype(int).reshape(rows, cols)
    
    # Create overlay
    overlay = np.zeros((rows, cols, 3))
    overlay[:, :, 0] = groundtruth  # Red for groundtruth
    overlay[:, :, 1] = predictions  # Green for predictions
    overlay[:, :, 2] = (groundtruth & predictions)  # Yellow for correct
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(groundtruth, cmap='Reds', interpolation='nearest')
    axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(predictions, cmap='Greens', interpolation='nearest')
    axes[1].set_title('Predictions', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(overlay, interpolation='nearest')
    axes[2].set_title('Overlay (Red=GT, Green=Pred, Yellow=Correct)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Overlay comparison saved to: {save_path}")
    
    return fig


def plot_spectral_signatures(data3d, target2d, groundtruth, rows, cols, bands, 
                              num_samples=5, save_path=None):
    """
    Plot spectral signatures of detected anomalies vs background
    
    :param data3d: original 3D hyperspectral data
    :param target2d: 2D anomaly component
    :param groundtruth: groundtruth labels
    :param rows: number of rows
    :param cols: number of cols
    :param bands: number of bands
    :param num_samples: number of sample pixels to plot
    :param save_path: path to save figure
    """
    # Find anomaly and background pixels
    anomaly_scores = np.zeros((rows * cols,))
    for i in range(rows * cols):
        anomaly_scores[i] = np.linalg.norm(target2d[:, i])
    
    anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-10)
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(groundtruth.flatten(), anomaly_scores)
    optimal_idx = np.argmax(tpr - fpr)
    threshold = thresholds[optimal_idx]
    predictions = (anomaly_scores >= threshold).astype(int)
    
    anomaly_indices = np.where(predictions == 1)[0]
    background_indices = np.where(predictions == 0)[0]
    
    # Sample indices
    if len(anomaly_indices) > 0:
        anomaly_samples = np.random.choice(anomaly_indices, 
                                         min(num_samples, len(anomaly_indices)), 
                                         replace=False)
    else:
        anomaly_samples = []
    
    if len(background_indices) > 0:
        background_samples = np.random.choice(background_indices, 
                                            min(num_samples, len(background_indices)), 
                                            replace=False)
    else:
        background_samples = []
    
    # Get spectral signatures
    data2d = data3d.reshape(rows * cols, bands)
    wavelengths = np.arange(bands)  # Simplified - could use actual wavelengths
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot background signatures
    for idx in background_samples:
        ax.plot(wavelengths, data2d[idx, :], 'b-', alpha=0.3, linewidth=0.5)
    
    # Plot anomaly signatures
    for idx in anomaly_samples:
        ax.plot(wavelengths, data2d[idx, :], 'r-', alpha=0.5, linewidth=1)
    
    # Plot mean signatures
    if len(background_samples) > 0:
        mean_bg = np.mean(data2d[background_samples, :], axis=0)
        ax.plot(wavelengths, mean_bg, 'b-', linewidth=2, label='Mean Background', alpha=0.8)
    
    if len(anomaly_samples) > 0:
        mean_anomaly = np.mean(data2d[anomaly_samples, :], axis=0)
        ax.plot(wavelengths, mean_anomaly, 'r-', linewidth=2, label='Mean Anomaly', alpha=0.8)
    
    ax.set_xlabel('Spectral Band', fontsize=12)
    ax.set_ylabel('Reflectance', fontsize=12)
    ax.set_title('Spectral Signatures Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spectral signatures saved to: {save_path}")
    
    return fig


def save_all_visualizations(target2d, groundtruth, data3d, rows, cols, bands, output_dir='results'):
    """Save all visualizations to output directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    plot_anomaly_heatmap(target2d, groundtruth, rows, cols, 
                        save_path=os.path.join(output_dir, 'anomaly_heatmap.png'))
    
    plot_overlay_comparison(target2d, groundtruth, rows, cols,
                           save_path=os.path.join(output_dir, 'overlay_comparison.png'))
    
    plot_spectral_signatures(data3d, target2d, groundtruth, rows, cols, bands,
                            save_path=os.path.join(output_dir, 'spectral_signatures.png'))

