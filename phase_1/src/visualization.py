import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from src.utils import ensure_dir

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.size': 10})

def plot_confusion_matrix(cm, class_names, title, save_path):
    ensure_dir(os.path.dirname(save_path))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontweight='bold')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_pr_curves(y_true, y_score, title, save_path_prefix):
    ensure_dir(os.path.dirname(save_path_prefix))
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR / Recall)')
    plt.title(f'{title} - ROC Curve', fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_roc.png", dpi=300)
    plt.close()

    # PR Curve
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(rec, prec)

    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color='#2ca02c', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{title} - Precision-Recall Curve', fontweight='bold')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_pr.png", dpi=300)
    plt.close()

def plot_feature_importances(importances, feature_names, title, save_path, top_n=20):
    ensure_dir(os.path.dirname(save_path))
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(8, 6))
    bars = plt.barh(range(len(indices)), top_importances[::-1], color='#337ab7')
    plt.yticks(range(len(indices)), top_features[::-1], fontsize=9)
    plt.xlabel('Importance')
    plt.title(f'{title} (Top {top_n} Features)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_reconstruction_distributions(err_normal, err_attack, threshold, title, save_path, err_unseen=None):
    ensure_dir(os.path.dirname(save_path))
    plt.figure(figsize=(9, 5))
    sns.kdeplot(err_normal, label='Normal Operational Baseline', color='#5cb85c', fill=True, alpha=0.4)
    sns.kdeplot(err_attack, label='Known Cyberattacks', color='#d9534f', fill=True, alpha=0.4)
    if err_unseen is not None:
        sns.kdeplot(err_unseen, label='Unseen / Zero-Day Attack', color='#f0ad4e', fill=True, alpha=0.4)

    plt.axvline(threshold, color='black', linestyle='--', lw=2, label=f'Anomaly Threshold ({threshold:.4f})')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Density')
    plt.title(title, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
