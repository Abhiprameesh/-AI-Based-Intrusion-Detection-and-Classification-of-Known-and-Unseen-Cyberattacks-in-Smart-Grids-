"""
Milestone 4.3 Workstream E: Unseen-Anomaly Evaluation Script.
Evaluates the recovered Data4Cyber Deep Autoencoder on the canonical 1,350-sample OpenDSS telemetry matrix.
Uses frozen preprocessor_scaled.joblib.
Evaluates under both recovered validation threshold (0.36402026) and historical threshold (0.42760131).
Saves predictions and detailed metrics to simulation/results/milestone4_3/.
"""

import os
import sys
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)

sys.path.append(os.path.abspath("phase_1"))
from src.anomaly_detection import Autoencoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_dist_stats(arr):
    if len(arr) == 0:
        return None
    return {
        'count': int(len(arr)),
        'min': float(np.min(arr)),
        'p05': float(np.percentile(arr, 5)),
        'p25': float(np.percentile(arr, 25)),
        'median': float(np.median(arr)),
        'mean': float(np.mean(arr)),
        'p75': float(np.percentile(arr, 75)),
        'p95': float(np.percentile(arr, 95)),
        'p99': float(np.percentile(arr, 99)),
        'max': float(np.max(arr)),
        'std': float(np.std(arr))
    }

def run_unseen_evaluation():
    print("=" * 80)
    print("STARTING WORKSTREAM E: UNSEEN ANOMALY EVALUATION")
    print("=" * 80)

    features_path = "simulation/results/milestone4/telemetry_features.csv"
    meta_path = "simulation/results/milestone4/telemetry_metadata.csv"
    scaler_path = "phase_1/results/data4cyber_dataset/binary/models/preprocessor_scaled.joblib"
    weights_path = "phase_1/results/data4cyber_dataset/anomaly/recovery_v1/Autoencoder.pt"
    threshold_meta_path = "phase_1/results/data4cyber_dataset/anomaly/recovery_v1/threshold_metadata.json"
    out_dir = "simulation/results/milestone4_3"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Data
    print(f"[Step 1] Loading canonical telemetry features: {features_path}")
    df_features = pd.read_csv(features_path)
    df_meta = pd.read_csv(meta_path)
    assert len(df_features) == 1350, f"Expected 1350 rows, got {len(df_features)}"
    assert len(df_meta) == 1350, f"Expected 1350 rows, got {len(df_meta)}"
    assert df_features.shape[1] == 150, f"Expected 150 features, got {df_features.shape[1]}"

    # Verify zero nulls
    assert df_features.isna().sum().sum() == 0, "Telemetry features contain NaNs!"

    # 2. Load Scaler and Verify Features Contract
    print(f"[Step 2] Loading scaler: {scaler_path}")
    scaler = joblib.load(scaler_path)
    scaler_hash = compute_file_hash(scaler_path)
    if hasattr(scaler, 'feature_names_in_'):
        assert list(df_features.columns) == list(scaler.feature_names_in_), "Feature column order mismatch!"

    X_scaled = scaler.transform(df_features)
    assert np.all(np.isfinite(X_scaled)), "Scaled features contain non-finite values!"

    # 3. Load Recovered Autoencoder
    print(f"[Step 3] Loading recovered weights: {weights_path}")
    weights_hash = compute_file_hash(weights_path)
    model = Autoencoder(in_features=150, bottleneck_dim=16).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()

    # 4. Load Thresholds
    with open(threshold_meta_path) as f:
        thresh_info = json.load(f)
    recovered_threshold = float(thresh_info['calculated_threshold'])
    historical_threshold = float(thresh_info['historical_threshold'])
    print(f"  Recovered Threshold:  {recovered_threshold:.8f}")
    print(f"  Historical Threshold: {historical_threshold:.8f}")

    # 5. Compute Reconstruction Errors
    print("[Step 4] Computing reconstruction MSE on 1,350 evaluation samples...")
    t_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    t0 = os.times()
    errors = model.compute_reconstruction_error(X_scaled)
    assert len(errors) == 1350, f"Expected 1350 error values, got {len(errors)}"
    assert np.all(np.isfinite(errors)), "Reconstruction errors contain non-finite values!"

    y_true = df_meta['attack_active'].values.astype(int)
    scenarios = df_meta['scenario_id'].values
    is_unseen = df_meta['is_unseen_scenario'].values.astype(bool)

    # 6. Evaluate Metrics Function
    def evaluate_at_threshold(tau, thresh_name):
        y_pred = (errors >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        tnr = float(tn / (fp + tn)) if (fp + tn) > 0 else 0.0

        # Subsets
        # Unseen Attacks (S6 & S7)
        unseen_mask = (y_true == 1) & is_unseen
        s6_att_mask = (y_true == 1) & (scenarios == 'S6_unseen_stealthy_ramp_fdi')
        s7_att_mask = (y_true == 1) & (scenarios == 'S7_unseen_coordinated_masking')
        known_att_mask = (y_true == 1) & (~is_unseen)

        # Benign Subsets
        s0_mask = (scenarios == 'S0_benign_baseline')
        s8_mask = (scenarios == 'S8_benign_cloud_cover_shading')

        unseen_rec = float(np.mean(y_pred[unseen_mask])) if np.sum(unseen_mask) > 0 else 0.0
        s6_rec = float(np.mean(y_pred[s6_att_mask])) if np.sum(s6_att_mask) > 0 else 0.0
        s7_rec = float(np.mean(y_pred[s7_att_mask])) if np.sum(s7_att_mask) > 0 else 0.0
        known_rec = float(np.mean(y_pred[known_att_mask])) if np.sum(known_att_mask) > 0 else 0.0

        s0_fpr = float(np.mean(y_pred[s0_mask])) if np.sum(s0_mask) > 0 else 0.0
        s8_fpr = float(np.mean(y_pred[s8_mask])) if np.sum(s8_mask) > 0 else 0.0

        # Per scenario detailed breakdown
        per_sc = {}
        for sc in sorted(list(np.unique(scenarios))):
            sc_m = (scenarios == sc)
            sc_y = y_true[sc_m]
            sc_yp = y_pred[sc_m]
            sc_att_m = (sc_y == 1)
            sc_ben_m = (sc_y == 0)

            att_cnt = int(np.sum(sc_att_m))
            ben_cnt = int(np.sum(sc_ben_m))
            sc_rec = float(np.mean(sc_yp[sc_att_m])) if att_cnt > 0 else None
            sc_fpr = float(np.mean(sc_yp[sc_ben_m])) if ben_cnt > 0 else 0.0
            sc_acc = float(accuracy_score(sc_y, sc_yp))

            per_sc[sc] = {
                'total_samples': int(len(sc_y)),
                'attack_samples': att_cnt,
                'benign_samples': ben_cnt,
                'accuracy': sc_acc,
                'attack_detection_rate': sc_rec,
                'false_positive_rate': sc_fpr
            }

        return {
            'threshold_name': thresh_name,
            'threshold_value': tau,
            'overall_accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'false_positive_rate': fpr,
            'true_negative_rate': tnr,
            'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)},
            'unseen_attack_detection_rate_overall': unseen_rec,
            's6_unseen_ramp_fdi_detection_rate': s6_rec,
            's7_unseen_coordinated_masking_detection_rate': s7_rec,
            'known_attack_detection_rate_overall': known_rec,
            's0_baseline_false_positive_rate': s0_fpr,
            's8_cloud_shading_false_positive_rate': s8_fpr,
            'per_scenario': per_sc
        }

    eval_recovered = evaluate_at_threshold(recovered_threshold, "recovered_val_p99")
    eval_historical = evaluate_at_threshold(historical_threshold, "historical_phase1")

    # 7. Ranking Metrics (Threshold-Independent ROC and PR)
    roc_auc = float(roc_auc_score(y_true, errors))
    prec_pts, rec_pts, _ = precision_recall_curve(y_true, errors)
    pr_auc = float(auc(rec_pts, prec_pts))
    prevalence = float(np.mean(y_true))

    # 8. Reconstruction Error Distributions
    dist_all = get_dist_stats(errors)
    dist_benign = get_dist_stats(errors[y_true == 0])
    dist_attack = get_dist_stats(errors[y_true == 1])
    dist_known = get_dist_stats(errors[(y_true == 1) & (~is_unseen)])
    dist_unseen = get_dist_stats(errors[(y_true == 1) & is_unseen])

    per_sc_dists = {}
    for sc in sorted(list(np.unique(scenarios))):
        sc_m = (scenarios == sc)
        per_sc_dists[sc] = {
            'all': get_dist_stats(errors[sc_m]),
            'attack': get_dist_stats(errors[sc_m & (y_true == 1)]),
            'benign': get_dist_stats(errors[sc_m & (y_true == 0)])
        }

    # 9. Detection Latency Analysis
    # Latency: timesteps elapsed from attack onset until first true alarm
    latency_records = {}
    for sc in ['S1_fdi_bss_scaling_0.5', 'S2_fdi_substation_bias_5kw', 'S3_industroyer_pv_trip',
               'S4_industroyer_bss_forced_discharge', 'S5_load_b_telemetry_freeze',
               'S6_unseen_stealthy_ramp_fdi', 'S7_unseen_coordinated_masking']:
        sc_m = (scenarios == sc)
        sc_steps = df_meta.loc[sc_m, 'step'].values
        sc_y = y_true[sc_m]
        sc_err = errors[sc_m]
        sc_pred = (sc_err >= recovered_threshold).astype(int)

        att_indices = np.where(sc_y == 1)[0]
        if len(att_indices) > 0:
            onset_idx = att_indices[0]
            onset_step = sc_steps[onset_idx]
            # find first pred==1 at or after onset
            alarm_indices = np.where((sc_pred == 1) & (np.arange(len(sc_pred)) >= onset_idx))[0]
            if len(alarm_indices) > 0:
                first_alarm_idx = alarm_indices[0]
                latency_steps = int(first_alarm_idx - onset_idx)
                detected = True
            else:
                latency_steps = None
                detected = False

            latency_records[sc] = {
                'onset_step': int(onset_step),
                'attack_detected': detected,
                'latency_timesteps': latency_steps,
                'first_alarm_step': int(sc_steps[first_alarm_idx]) if detected else None
            }

    # 10. Assemble and Save Results
    results_payload = {
        'metadata': {
            'features_file': features_path,
            'features_hash': compute_file_hash(features_path),
            'metadata_file': meta_path,
            'metadata_hash': compute_file_hash(meta_path),
            'weights_file': weights_path,
            'weights_hash': weights_hash,
            'scaler_file': scaler_path,
            'scaler_hash': scaler_hash,
            'num_samples': 1350,
            'attack_prevalence': prevalence
        },
        'ranking_metrics': {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'baseline_pr_auc': prevalence
        },
        'evaluation_recovered_threshold': eval_recovered,
        'evaluation_historical_threshold': eval_historical,
        'reconstruction_error_distributions': {
            'all_samples': dist_all,
            'benign_overall': dist_benign,
            'attack_overall': dist_attack,
            'known_attacks': dist_known,
            'unseen_attacks': dist_unseen,
            'per_scenario': per_sc_dists
        },
        'detection_latency': latency_records
    }

    metrics_out_file = os.path.join(out_dir, "unseen_autoencoder_metrics.json")
    with open(metrics_out_file, 'w') as f:
        json.dump(results_payload, f, indent=2)

    # Save predictions dataframe
    df_preds_out = pd.DataFrame({
        'step': df_meta['step'],
        'scenario_id': df_meta['scenario_id'],
        'attack_active_ground_truth': y_true,
        'is_unseen_scenario': is_unseen,
        'reconstruction_mse': errors,
        'pred_recovered_threshold': (errors >= recovered_threshold).astype(int),
        'pred_historical_threshold': (errors >= historical_threshold).astype(int)
    })
    preds_out_file = os.path.join(out_dir, "unseen_autoencoder_predictions.csv")
    df_preds_out.to_csv(preds_out_file, index=False)

    print(f"\nWorkstream E completed successfully!")
    print(f"Metrics saved to: {metrics_out_file}")
    print(f"Predictions saved to: {preds_out_file}")

    print("\n--- SUMMARY OF RESULTS (Recovered Threshold = 0.3640) ---")
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} (Baseline: {prevalence:.4f})")
    print(f"Overall Accuracy: {eval_recovered['overall_accuracy']:.4f}")
    print(f"Overall Precision: {eval_recovered['precision']:.4f}")
    print(f"Overall Attack Recall: {eval_recovered['recall']:.4f}")
    print(f"Overall F1-Score: {eval_recovered['f1_score']:.4f}")
    print(f"False Positive Rate (Overall Benign): {eval_recovered['false_positive_rate']:.4f}")
    print(f"S0 False Positive Rate: {eval_recovered['s0_baseline_false_positive_rate']:.4f}")
    print(f"S8 False Positive Rate: {eval_recovered['s8_cloud_shading_false_positive_rate']:.4f}")
    print(f"Unseen Attack Detection Rate (S6 & S7): {eval_recovered['unseen_attack_detection_rate_overall']:.4f}")
    print(f"  - S6 (Ramp FDI): {eval_recovered['s6_unseen_ramp_fdi_detection_rate']:.4f}")
    print(f"  - S7 (Coordinated Masking): {eval_recovered['s7_unseen_coordinated_masking_detection_rate']:.4f}")
    print(f"Known Attack Detection Rate (S1-S5): {eval_recovered['known_attack_detection_rate_overall']:.4f}")

if __name__ == '__main__':
    run_unseen_evaluation()
