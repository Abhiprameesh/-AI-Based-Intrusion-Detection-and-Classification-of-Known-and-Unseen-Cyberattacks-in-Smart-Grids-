import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.utils import ensure_dir, save_json

def evaluate_unseen_attack_detection(
    model,
    X_normal_test,
    X_known_attack_test,
    X_unseen_attack_test,
    model_type="autoencoder",
    threshold=None,
    results_dir=None,
    attack_name="Zero-Day"
):
    """
    Evaluates unseen attack detection.
    Normal: label 0
    Known attacks: label 1
    Unseen attacks: label 1 (target is to detect them as anomalous/novel)
    """
    if model_type == "autoencoder":
        err_normal = model.compute_reconstruction_error(X_normal_test)
        err_known = model.compute_reconstruction_error(X_known_attack_test)
        err_unseen = model.compute_reconstruction_error(X_unseen_attack_test)

        pred_normal_anom = (err_normal >= threshold).astype(int)
        pred_known_anom = (err_known >= threshold).astype(int)
        pred_unseen_anom = (err_unseen >= threshold).astype(int)

        fpr = float(np.mean(pred_normal_anom))
        known_rec = float(np.mean(pred_known_anom))
        unseen_rec = float(np.mean(pred_unseen_anom))

        # Overall binary evaluation: Normal (0) vs All Attacks (1)
        y_true = np.concatenate([np.zeros(len(err_normal)), np.ones(len(err_known) + len(err_unseen))])
        all_errs = np.concatenate([err_normal, err_known, err_unseen])
        y_pred = (all_errs >= threshold).astype(int)

        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_true, all_errs))

        results = {
            'approach': 'Autoencoder_Residual_Thresholding',
            'unseen_attack_name': attack_name,
            'threshold': float(threshold),
            'unseen_attack_detection_rate': float(unseen_rec),
            'known_attack_detection_rate': float(known_rec),
            'false_positive_rate_on_normal': float(fpr),
            'overall_accuracy': acc,
            'overall_f1': f1,
            'overall_roc_auc': roc_auc,
            'num_normal_test': len(X_normal_test),
            'num_known_test': len(X_known_attack_test),
            'num_unseen_test': len(X_unseen_attack_test)
        }

    elif model_type in ["rf_open_set", "xgb_open_set", "mlp_open_set"]:
        # Open-set via confidence rejection
        # Predict probabilities on known classes
        prob_normal = model.predict_proba(X_normal_test)
        prob_known = model.predict_proba(X_known_attack_test)
        prob_unseen = model.predict_proba(X_unseen_attack_test)

        # Max confidence score
        conf_norm = np.max(prob_normal, axis=1)
        conf_known = np.max(prob_known, axis=1)
        conf_unseen = np.max(prob_unseen, axis=1)

        # Reject as unseen if confidence < threshold
        if threshold is None:
            # Calibrated as 5th percentile of known training/val confidence
            threshold = 0.60

        flagged_normal = (conf_norm < threshold).astype(int)
        flagged_known = (conf_known < threshold).astype(int)
        flagged_unseen = (conf_unseen < threshold).astype(int)

        results = {
            'approach': f'{model_type}_confidence_rejection',
            'unseen_attack_name': attack_name,
            'confidence_threshold': float(threshold),
            'unseen_attack_detection_rate': float(np.mean(flagged_unseen)),
            'known_flagged_as_unknown_rate': float(np.mean(flagged_known)),
            'false_alarm_on_normal_rate': float(np.mean(flagged_normal)),
            'num_unseen_test': len(X_unseen_attack_test)
        }

    if results_dir:
        ensure_dir(results_dir)
        save_json(results, os.path.join(results_dir, f"unseen_attack_{attack_name}.json"))

    return results
