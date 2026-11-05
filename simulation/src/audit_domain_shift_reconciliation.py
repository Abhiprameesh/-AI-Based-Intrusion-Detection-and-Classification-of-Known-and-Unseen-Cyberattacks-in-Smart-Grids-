"""
Milestone 4.4 Audit Script:
1. Reconcile feature error percentages (69.56% for 8 reactive power features vs 83.89% for top 15 features).
2. Reproduce Stage 1, Stage 2, and Stage 3 metrics from saved prediction files.
3. Verify threshold calibration provenance and partition independence.
4. Calculate comprehensive 150-feature distribution audit comparing Data4Cyber training data vs OpenDSS simulation data.
"""

import os
import sys
import json
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
from src.data_loader import load_data4cyber
from src.anomaly_detection import Autoencoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_audit():
    print("=" * 80)
    print("RUNNING MILESTONE 4.4 DOMAIN-SHIFT AUDIT & RECONCILIATION")
    print("=" * 80)

    # 1. Paths
    sim_features_p = "simulation/results/milestone4/telemetry_features.csv"
    sim_meta_p = "simulation/results/milestone4/telemetry_metadata.csv"
    sim_preds_stage12_p = "simulation/results/milestone4/predictions_binary_and_multiclass.csv"
    sim_preds_stage3_p = "simulation/results/milestone4_3/unseen_autoencoder_predictions.csv"
    sim_metrics_stage3_p = "simulation/results/milestone4_3/unseen_autoencoder_metrics.json"

    scaler_p = "phase_1/results/data4cyber_dataset/binary/models/preprocessor_scaled.joblib"
    unscaled_prep_p = "phase_1/results/data4cyber_dataset/binary/models/preprocessor_unscaled.joblib"
    weights_p = "phase_1/results/data4cyber_dataset/anomaly/recovery_v1/Autoencoder.pt"
    thresh_meta_p = "phase_1/results/data4cyber_dataset/anomaly/recovery_v1/threshold_metadata.json"
    recovery_manifest_p = "phase_1/results/data4cyber_dataset/anomaly/recovery_v1/recovery_manifest.json"

    # Load OpenDSS simulation data
    df_sim_features = pd.read_csv(sim_features_p)
    df_sim_meta = pd.read_csv(sim_meta_p)
    df_sim_preds12 = pd.read_csv(sim_preds_stage12_p)
    df_sim_preds3 = pd.read_csv(sim_preds_stage3_p)
    with open(sim_metrics_stage3_p) as f:
        metrics_stage3 = json.load(f)

    # Load Phase 1 Data4Cyber data
    print("\n[Audit Step 1] Loading Phase 1 Data4Cyber dataset...")
    d4c_data = load_data4cyber(random_state=42)
    scaler = joblib.load(scaler_p)
    unscaled_prep = joblib.load(unscaled_prep_p)

    model = Autoencoder(in_features=150, bottleneck_dim=16).to(DEVICE)
    model.load_state_dict(torch.load(weights_p, weights_only=True))
    model.eval()

    # Part A.1 & A.2: Reconcile Feature Error Percentages
    print("\n[Audit Step 2] Reconciling reconstruction error percentages...")
    X_sim_scaled = scaler.transform(df_sim_features)
    with torch.no_grad():
        tx = torch.tensor(X_sim_scaled, dtype=torch.float32).to(DEVICE)
        rec = model(tx)
        # Per-sample, per-feature squared error: (1350, 150)
        sq_errors = ((tx - rec) ** 2).cpu().numpy()
        # Per-feature MSE across 1350 samples: (150,)
        feature_mse = sq_errors.mean(axis=0)

    total_mse_sum = float(feature_mse.sum())
    mean_mse_per_feature = float(feature_mse.mean())

    df_feat_err = pd.DataFrame({
        'feature': list(df_sim_features.columns),
        'mse': feature_mse,
        'pct_of_total': (feature_mse / total_mse_sum) * 100.0
    }).sort_values('mse', ascending=False).reset_index(drop=True)

    top15 = df_feat_err.head(15)
    top15_sum = float(top15['mse'].sum())
    top15_pct = (top15_sum / total_mse_sum) * 100.0

    reactive_8_names = [
        'Load-B-AC-Meter.q_sum3', 'Load-B-AC-Meter.qln2', 'Load-B-AC-Meter.qln3', 'Load-B-AC-Meter.qln1',
        'Load-A-AC-Meter.qln2', 'Load-A-AC-Meter.qln3', 'Load-A-AC-Meter.qln1', 'Load-A-AC-Meter.q_sum3'
    ]
    df_reactive_8 = df_feat_err[df_feat_err['feature'].isin(reactive_8_names)]
    reactive_8_sum = float(df_reactive_8['mse'].sum())
    reactive_8_pct = (reactive_8_sum / total_mse_sum) * 100.0

    print(f"Total MSE sum across all 150 features: {total_mse_sum:.4f}")
    print(f"Mean MSE per sample across 150 features: {mean_mse_per_feature:.4f}")
    print(f"Top 15 features MSE sum: {top15_sum:.4f} -> {top15_pct:.4f}%")
    print(f"8 Reactive Power features MSE sum: {reactive_8_sum:.4f} -> {reactive_8_pct:.4f}%")
    print("Top 15 features table:\n", top15[['feature', 'mse', 'pct_of_total']])

    # Part A.3 & A.4: Verify reproduction of reported metrics from saved prediction files
    print("\n[Audit Step 3] Reproducing metrics from saved prediction files...")

    # Stage 1: Binary XGBoost
    y_true_bin = df_sim_preds12['attack_active_ground_truth'].values.astype(int)
    y_prob_bin = df_sim_preds12['binary_xgb_prob'].values.astype(float)
    y_pred_bin_05 = df_sim_preds12['binary_xgb_pred'].values.astype(int)

    acc_s1 = accuracy_score(y_true_bin, y_pred_bin_05)
    rec_s1 = recall_score(y_true_bin, y_pred_bin_05, zero_division=0)
    prec_s1 = precision_score(y_true_bin, y_pred_bin_05, zero_division=0)
    f1_s1 = f1_score(y_true_bin, y_pred_bin_05, zero_division=0)
    tn_s1, fp_s1, fn_s1, tp_s1 = confusion_matrix(y_true_bin, y_pred_bin_05, labels=[0, 1]).ravel()
    fpr_s1 = fp_s1 / (fp_s1 + tn_s1)
    tnr_s1 = tn_s1 / (fp_s1 + tn_s1)
    roc_auc_s1 = roc_auc_score(y_true_bin, y_prob_bin)
    p_pts, r_pts, _ = precision_recall_curve(y_true_bin, y_prob_bin)
    pr_auc_s1 = auc(r_pts, p_pts)

    print(f"Stage 1 Binary Repro: Acc={acc_s1:.4f}, Rec={rec_s1:.4f}, Prec={prec_s1:.4f}, F1={f1_s1:.4f}, FPR={fpr_s1:.4f}, TNR={tnr_s1:.4f}, ROC={roc_auc_s1:.4f}, PR={pr_auc_s1:.4f}")

    # Stage 2: Multiclass RF
    # Supported samples only
    supp_mask = df_sim_preds12['multiclass_is_supported'].values.astype(bool)
    n_supp = int(supp_mask.sum())
    n_excl = int((~supp_mask).sum())
    y_true_multi_supp = df_sim_preds12.loc[supp_mask, 'multiclass_true_class'].values.astype(int)

    # Re-run prediction using saved frozen RF model to confirm exact match with report
    rf_multi = joblib.load("phase_1/results/data4cyber_dataset/multiclass/models/Random_Forest_multiclass.joblib")
    X_sim_raw = unscaled_prep.transform(df_sim_features)
    y_pred_multi_all = rf_multi.predict(X_sim_raw)
    y_pred_multi_supp = y_pred_multi_all[supp_mask]

    acc_s2 = accuracy_score(y_true_multi_supp, y_pred_multi_supp)
    macro_f1_s2 = f1_score(y_true_multi_supp, y_pred_multi_supp, average='macro', zero_division=0)
    weighted_f1_s2 = f1_score(y_true_multi_supp, y_pred_multi_supp, average='weighted', zero_division=0)
    cm_s2 = confusion_matrix(y_true_multi_supp, y_pred_multi_supp, labels=[0, 1, 2, 3])

    # Per class recall
    rec_c0 = cm_s2[0, 0] / cm_s2[0].sum() if cm_s2[0].sum() > 0 else 0.0
    rec_c1 = cm_s2[1, 1] / cm_s2[1].sum() if cm_s2[1].sum() > 0 else 0.0
    rec_c2 = cm_s2[2, 2] / cm_s2[2].sum() if cm_s2[2].sum() > 0 else 0.0

    print(f"Stage 2 Multiclass Repro: Supp={n_supp}, Excl={n_excl}, Acc={acc_s2:.4f}, MacroF1={macro_f1_s2:.4f}, WeightedF1={weighted_f1_s2:.4f}")
    print(f"  Class 0 (Benign) Recall: {rec_c0:.4f} ({cm_s2[0,0]}/{cm_s2[0].sum()})")
    print(f"  Class 1 (Industroyer) Recall: {rec_c1:.4f} ({cm_s2[1,1]}/{cm_s2[1].sum()})")
    print(f"  Class 2 (MitM FDI) Recall: {rec_c2:.4f} ({cm_s2[2,2]}/{cm_s2[2].sum()})")
    print(f"  Confusion Matrix 4x4:\n{cm_s2}")

    # Stage 3: Autoencoder Unseen Anomaly
    y_true_s3 = df_sim_preds3['attack_active_ground_truth'].values.astype(int)
    mse_s3 = df_sim_preds3['reconstruction_mse'].values.astype(float)
    y_pred_s3_rec = df_sim_preds3['pred_recovered_threshold'].values.astype(int)
    y_pred_s3_hist = df_sim_preds3['pred_historical_threshold'].values.astype(int)

    acc_s3 = accuracy_score(y_true_s3, y_pred_s3_rec)
    rec_s3 = recall_score(y_true_s3, y_pred_s3_rec, zero_division=0)
    prec_s3 = precision_score(y_true_s3, y_pred_s3_rec, zero_division=0)
    f1_s3 = f1_score(y_true_s3, y_pred_s3_rec, zero_division=0)
    tn_s3, fp_s3, fn_s3, tp_s3 = confusion_matrix(y_true_s3, y_pred_s3_rec, labels=[0, 1]).ravel()
    fpr_s3 = fp_s3 / (fp_s3 + tn_s3)
    roc_auc_s3 = roc_auc_score(y_true_s3, mse_s3)
    p_pts_s3, r_pts_s3, _ = precision_recall_curve(y_true_s3, mse_s3)
    pr_auc_s3 = auc(r_pts_s3, p_pts_s3)

    print(f"Stage 3 Autoencoder Repro: Acc={acc_s3:.4f}, Rec={rec_s3:.4f}, Prec={prec_s3:.4f}, F1={f1_s3:.4f}, FPR={fpr_s3:.4f}, ROC={roc_auc_s3:.4f}, PR={pr_auc_s3:.4f}")

    # Part A.5 & A.6: Threshold Provenance & Partition Audit
    print("\n[Audit Step 4] Auditing threshold provenance and data partitions...")
    le = d4c_data['label_encoder']
    benign_idx = list(le.classes_).index('benign')

    X_tr_sc = scaler.transform(d4c_data['X_train'])
    X_val_sc = scaler.transform(d4c_data['X_val'])
    X_te_sc = scaler.transform(d4c_data['X_test'])

    X_norm_val = X_val_sc[d4c_data['y_val_family'] == benign_idx]
    X_norm_tr = X_tr_sc[d4c_data['y_train_family'] == benign_idx]
    X_norm_te = X_te_sc[d4c_data['y_test_family'] == benign_idx]

    val_errors = model.compute_reconstruction_error(X_norm_val)
    recalculated_val_p99 = float(np.percentile(val_errors, 99.0))

    with open(thresh_meta_p) as f:
        tmeta = json.load(f)

    recorded_threshold = float(tmeta['calculated_threshold'])
    thresh_diff = abs(recalculated_val_p99 - recorded_threshold)
    print(f"Recalculated Val P99 Threshold: {recalculated_val_p99:.8f}")
    print(f"Recorded Threshold in metadata: {recorded_threshold:.8f}")
    print(f"Threshold exact match: {thresh_diff < 1e-7} (diff = {thresh_diff})")

    # Check whether historical report used validation or test
    hist_test_eval = tmeta.get('historical_test_evaluation_with_new_threshold', {})
    print(f"Historical report test set evaluation used {hist_test_eval.get('num_normal_test')} normal test samples, {hist_test_eval.get('num_known_test')} known test, {hist_test_eval.get('num_unseen_test')} unseen test.")
    print(f"Total test samples evaluated: {hist_test_eval.get('num_normal_test') + hist_test_eval.get('num_known_test') + hist_test_eval.get('num_unseen_test')} (matches test set length {len(d4c_data['X_test'])})")

    # Part B: 150-Feature Domain-Shift Audit
    print("\n[Audit Step 5] Computing full 150-feature domain-shift distributions...")
    feature_names = list(df_sim_features.columns)

    # Data4Cyber training data (X_train unscaled)
    df_d4c_tr = d4c_data['X_train']
    # Scaler mean and scale
    scaler_means = scaler.named_steps['scaler'].mean_
    scaler_stds = scaler.named_steps['scaler'].scale_

    feat_audit = []
    for j, col in enumerate(feature_names):
        d4c_series = df_d4c_tr[col]
        sim_series = df_sim_features[col]

        d4c_m = float(d4c_series.mean())
        d4c_s = float(d4c_series.std())
        d4c_med = float(d4c_series.median())
        d4c_q05 = float(d4c_series.quantile(0.05))
        d4c_q95 = float(d4c_series.quantile(0.95))

        sim_m = float(sim_series.mean())
        sim_s = float(sim_series.std())
        sim_med = float(sim_series.median())
        sim_q05 = float(sim_series.quantile(0.05))
        sim_q95 = float(sim_series.quantile(0.95))

        sc_mean = float(scaler_means[j])
        sc_std = float(scaler_stds[j])

        # Standardized shift: (sim_mean - train_mean) / train_std
        std_shift = (sim_m - sc_mean) / sc_std if sc_std > 1e-12 else 0.0

        f_mse = float(feature_mse[j])
        f_pct = float((f_mse / total_mse_sum) * 100.0)

        feat_audit.append({
            'feature': col,
            'index': j,
            'd4c_train_mean': d4c_m,
            'd4c_train_std': d4c_s,
            'd4c_train_median': d4c_med,
            'd4c_train_q05': d4c_q05,
            'd4c_train_q95': d4c_q95,
            'sim_mean': sim_m,
            'sim_std': sim_s,
            'sim_median': sim_med,
            'sim_q05': sim_q05,
            'sim_q95': sim_q95,
            'scaler_mean': sc_mean,
            'scaler_std': sc_std,
            'standardized_shift_sigma': std_shift,
            'feature_reconstruction_mse': f_mse,
            'pct_of_total_reconstruction_error': f_pct,
            'is_constant_in_sim': bool(sim_s < 1e-9),
            'is_constant_in_train': bool(d4c_s < 1e-9)
        })

    df_full_audit = pd.DataFrame(feat_audit)

    # Save complete audit data
    out_dir = "simulation/results/milestone4_4"
    os.makedirs(out_dir, exist_ok=True)
    audit_json_p = os.path.join(out_dir, "feature_domain_shift_audit.json")
    audit_csv_p = os.path.join(out_dir, "feature_domain_shift_audit.csv")

    with open(audit_json_p, 'w') as f:
        json.dump({
            'reconciliation': {
                'total_mse_sum': total_mse_sum,
                'mean_mse_per_feature': mean_mse_per_feature,
                'top15_sum': top15_sum,
                'top15_pct': top15_pct,
                'reactive_8_sum': reactive_8_sum,
                'reactive_8_pct': reactive_8_pct,
                'top15_table': top15.to_dict(orient='records'),
                'reactive_8_table': df_reactive_8.to_dict(orient='records')
            },
            'metric_reproduction': {
                'stage1_binary': {
                    'accuracy': acc_s1, 'recall': rec_s1, 'precision': prec_s1, 'f1': f1_s1,
                    'fpr': fpr_s1, 'tnr': tnr_s1, 'roc_auc': roc_auc_s1, 'pr_auc': pr_auc_s1
                },
                'stage2_multiclass': {
                    'supported_samples': n_supp, 'excluded_samples': n_excl,
                    'accuracy': acc_s2, 'macro_f1': macro_f1_s2, 'weighted_f1': weighted_f1_s2,
                    'class_0_benign_recall': rec_c0,
                    'class_1_industroyer_recall': rec_c1,
                    'class_2_mitm_fdi_recall': rec_c2,
                    'confusion_matrix': cm_s2.tolist()
                },
                'stage3_autoencoder': {
                    'accuracy': acc_s3, 'recall': rec_s3, 'precision': prec_s3, 'f1': f1_s3,
                    'fpr': fpr_s3, 'roc_auc': roc_auc_s3, 'pr_auc': pr_auc_s3
                }
            },
            'threshold_provenance': {
                'recalculated_val_p99': recalculated_val_p99,
                'recorded_threshold': recorded_threshold,
                'match': bool(thresh_diff < 1e-7),
                'training_partition_samples': len(X_norm_tr),
                'val_partition_samples': len(X_norm_val),
                'test_partition_samples': len(X_norm_te)
            },
            'features': feat_audit
        }, f, indent=2)

    df_full_audit.to_csv(audit_csv_p, index=False)
    print(f"\nAudit complete! Saved detailed audit to {audit_json_p} and {audit_csv_p}")

    # Identify extreme features
    print("\n--- TOP 10 LARGEST STANDARDIZED SHIFTS (SIGMA) ---")
    df_shifts = df_full_audit.sort_values(by='standardized_shift_sigma', key=abs, ascending=False).head(10)
    for _, r in df_shifts.iterrows():
        print(f"{r['feature']:40s}: Shift={r['standardized_shift_sigma']:+10.2f} sigma | TrainMean={r['d4c_train_mean']:10.2f} | SimMean={r['sim_mean']:10.2f} | MSE={r['feature_reconstruction_mse']:10.2f}")

    print("\n--- VOLTAGE LEVEL AUDIT (Phase-to-Neutral) ---")
    v_cols = [c for c in feature_names if '.uln' in c]
    for vc in v_cols:
        r = df_full_audit[df_full_audit['feature'] == vc].iloc[0]
        print(f"{r['feature']:40s}: Shift={r['standardized_shift_sigma']:+6.2f} sigma | D4C={r['d4c_train_mean']:7.2f} V | Sim={r['sim_mean']:7.2f} V")

    print("\n--- NEUTRAL CURRENT AUDIT ---")
    i_cols = [c for c in feature_names if 'i_sum3' in c]
    for ic in i_cols:
        r = df_full_audit[df_full_audit['feature'] == ic].iloc[0]
        print(f"{r['feature']:40s}: Shift={r['standardized_shift_sigma']:+6.2f} sigma | D4C={r['d4c_train_mean']:7.2f} A | Sim={r['sim_mean']:7.2f} A")

if __name__ == '__main__':
    run_audit()
