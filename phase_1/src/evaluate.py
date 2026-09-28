import time
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    classification_report
)

def evaluate_binary_classifier(model, X_test, y_test, training_time=0.0, feature_names=None, model_filepath=None):
    """
    Evaluates a binary classification model (0=Normal/Negative, 1=Attack/Positive).
    Computes all standard and security-critical metrics:
    Attack Recall, Attack Precision, Macro F1, FPR, FNR, ROC-AUC, PR-AUC, Latency.
    """
    start_inf = time.time()
    y_pred = model.predict(X_test)
    inf_time_total = time.time() - start_inf
    latency_per_sample_ms = (inf_time_total / max(len(X_test), 1)) * 1000.0

    y_proba = None
    roc_auc = None
    pr_auc = None
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X_test)
            if probas.shape[1] == 2:
                y_proba = probas[:, 1]
            else:
                y_proba = probas[:, 0]
            roc_auc = roc_auc_score(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)
        except Exception:
            pass
    elif hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X_test)
            roc_auc = roc_auc_score(y_test, scores)
            pr_auc = average_precision_score(y_test, scores)
        except Exception:
            pass

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    # cm format: [[TN, FP], [FN, TP]]
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    else:
        tn = fp = fn = tp = fpr = fnr = 0.0

    model_size_kb = 0.0
    if model_filepath and os.path.exists(model_filepath):
        model_size_kb = os.path.getsize(model_filepath) / 1024.0

    results = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'fpr': float(fpr),
        'fnr': float(fnr),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'roc_auc': float(roc_auc) if roc_auc is not None else None,
        'pr_auc': float(pr_auc) if pr_auc is not None else None,
        'confusion_matrix': cm.tolist(),
        'training_time_sec': float(training_time),
        'inference_time_total_sec': float(inf_time_total),
        'latency_per_sample_ms': float(latency_per_sample_ms),
        'model_size_kb': float(model_size_kb),
        'num_test_samples': int(len(y_test))
    }
    return results

def evaluate_multiclass_classifier(model, X_test, y_test, class_names=None, training_time=0.0, model_filepath=None):
    """
    Evaluates a multiclass classifier.
    Computes overall Accuracy, Macro/Weighted Precision, Recall, F1, Per-class metrics, and Confusion Matrix.
    """
    start_inf = time.time()
    y_pred = model.predict(X_test)
    inf_time_total = time.time() - start_inf
    latency_per_sample_ms = (inf_time_total / max(len(X_test), 1)) * 1000.0

    acc = accuracy_score(y_test, y_pred)
    macro_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    weighted_rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Per-class metrics
    per_class_p = precision_score(y_test, y_pred, average=None, zero_division=0)
    per_class_r = recall_score(y_test, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)

    unique_labels = sorted(list(set(y_test).union(set(y_pred))))
    labels_map = {}
    for idx, lbl in enumerate(unique_labels):
        name = class_names[lbl] if class_names and lbl < len(class_names) else str(lbl)
        labels_map[name] = {
            'precision': float(per_class_p[idx]) if idx < len(per_class_p) else 0.0,
            'recall': float(per_class_r[idx]) if idx < len(per_class_r) else 0.0,
            'f1': float(per_class_f1[idx]) if idx < len(per_class_f1) else 0.0
        }

    model_size_kb = 0.0
    if model_filepath and os.path.exists(model_filepath):
        model_size_kb = os.path.getsize(model_filepath) / 1024.0

    results = {
        'accuracy': float(acc),
        'macro_precision': float(macro_prec),
        'macro_recall': float(macro_rec),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_prec),
        'weighted_recall': float(weighted_rec),
        'weighted_f1': float(weighted_f1),
        'per_class_metrics': labels_map,
        'confusion_matrix': cm.tolist(),
        'training_time_sec': float(training_time),
        'inference_time_total_sec': float(inf_time_total),
        'latency_per_sample_ms': float(latency_per_sample_ms),
        'model_size_kb': float(model_size_kb),
        'num_test_samples': int(len(y_test)),
        'num_classes': len(unique_labels)
    }
    return results
