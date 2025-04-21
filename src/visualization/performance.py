"""
Visualization utilities for prediction performance metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
from typing import List, Union


def plot_prediction_metrics(y_true: Union[List, np.ndarray],
                          y_pred: Union[List, np.ndarray],
                          threshold: float,
                          output_path: str,
                          figsize: tuple = (15, 10)) -> None:
    """
    Plot various performance metrics for the predictions.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Decision threshold
        output_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # 1. Confusion Matrix
    plt.subplot(2, 2, 1)
    cm = confusion_matrix(y_true, y_pred_binary)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # 2. ROC Curve
    plt.subplot(2, 2, 2)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # 3. Precision-Recall Curve
    plt.subplot(2, 2, 3)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    # 4. Prediction Distribution
    plt.subplot(2, 2, 4)
    plt.hist(y_pred[y_true == 0], bins=50, alpha=0.5, label='No Handover',
             color='green', density=True)
    plt.hist(y_pred[y_true == 1], bins=50, alpha=0.5, label='Handover',
             color='red', density=True)
    plt.axvline(x=threshold, color='black', linestyle='--',
                label=f'Threshold ({threshold:.2f})')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Prediction Distribution')
    plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
