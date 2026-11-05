"""
Milestone 4.3 Workstream B: Threshold-Independent Analysis Script.
Processes canonical saved predictions and metadata from Milestone 4 without modifying them.
Calculates ROC-AUC, PR-AUC, score distribution percentiles, scenario comparisons,
and exploratory threshold sweep metrics.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from scipy.stats import mannwhitneyu

def run_threshold_independent_analysis():
    preds_path = "simulation/results/milestone4/predictions_binary_and_multiclass.csv"
    meta_path = "simulation/results/milestone4/telemetry_metadata.csv"
    out_dir = "simulation/results/milestone4_3"
    os.makedirs(out_dir, exist_ok=True)

    df_preds = pd.read_csv(preds_path)
    df_meta = pd.read_csv(meta_path)

    # 1. Alignment Verification
    assert len(df_preds) == 1350, f"Expected 1350 rows, got {len(df_preds)}"
    assert len(df_meta) == 1350, f"Expected 1350 rows, got {len(df_meta)}"
    assert (df_preds['step'] == df_meta['step']).all(), "Step mismatch between preds and meta"
    assert (df_preds['scenario_id'] == df_meta['scenario_id']).all(), "Scenario mismatch between preds and meta"

    y_true = df_preds['attack_active_ground_truth'].values.astype(int)
    y_prob = df_preds['binary_xgb_prob'].values.astype(float)
    scenarios = df_preds['scenario_id'].values

    num_samples = len(y_true)
    num_attacks = int(np.sum(y_true == 1))
    num_benign = int(np.sum(y_true == 0))
    prevalence = num_attacks / num_samples

    # 2. Ranking Metrics (Threshold-Independent)
    roc_auc = float(roc_auc_score(y_true, y_prob))
    precision_pts, recall_pts, pr_thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recall_pts, precision_pts))

    # Mann-Whitney U Test (ranks comparison)
    benign_probs = y_prob[y_true == 0]
    attack_probs = y_prob[y_true == 1]
    mwu_stat, mwu_pval = mannwhitneyu(attack_probs, benign_probs, alternative='greater')

    # 3. Baseline threshold tau = 0.50 metrics
    y_pred_05 = (y_prob >= 0.50).astype(int)
    tn_05, fp_05, fn_05, tp_05 = confusion_matrix(y_true, y_pred_05, labels=[0, 1]).ravel()
    prec_05 = float(precision_score(y_true, y_pred_05, zero_division=0))
    rec_05 = float(recall_score(y_true, y_pred_05, zero_division=0))
    f1_05 = float(f1_score(y_true, y_pred_05, zero_division=0))
    fpr_05 = float(fp_05 / (fp_05 + tn_05)) if (fp_05 + tn_05) > 0 else 0.0
    tnr_05 = float(tn_05 / (fp_05 + tn_05)) if (fp_05 + tn_05) > 0 else 0.0

    # 4. Score Distributions by Class Category
    # Categories: Benign Overall (813), Known Attacks (437), Unseen Attacks (100)
    is_unseen = df_meta['is_unseen_scenario'].values.astype(bool)
    known_attack_probs = y_prob[(y_true == 1) & (~is_unseen)]
    unseen_attack_probs = y_prob[(y_true == 1) & (is_unseen)]

    def get_dist_stats(arr):
        return {
            'count': int(len(arr)),
            'min': float(np.min(arr)),
            'p05': float(np.percentile(arr, 5)),
            'p25': float(np.percentile(arr, 25)),
            'median': float(np.median(arr)),
            'mean': float(np.mean(arr)),
            'p75': float(np.percentile(arr, 75)),
            'p95': float(np.percentile(arr, 95)),
            'max': float(np.max(arr)),
            'std': float(np.std(arr))
        }

    cat_distributions = {
        'all_samples': get_dist_stats(y_prob),
        'benign_overall': get_dist_stats(benign_probs),
        'attack_overall': get_dist_stats(attack_probs),
        'known_attacks': get_dist_stats(known_attack_probs),
        'unseen_attacks': get_dist_stats(unseen_attack_probs)
    }

    # 5. Per-Scenario Score Distributions
    scenario_order = [
        "S0_benign_baseline",
        "S1_fdi_bss_scaling_0.5",
        "S2_fdi_substation_bias_5kw",
        "S3_industroyer_pv_trip",
        "S4_industroyer_bss_forced_discharge",
        "S5_load_b_telemetry_freeze",
        "S6_unseen_stealthy_ramp_fdi",
        "S7_unseen_coordinated_masking",
        "S8_benign_cloud_cover_shading"
    ]

    per_scenario_stats = {}
    for sc in scenario_order:
        sc_mask = (df_preds['scenario_id'] == sc)
        sc_probs = y_prob[sc_mask]
        sc_y = y_true[sc_mask]
        sc_att_probs = sc_probs[sc_y == 1]
        sc_ben_probs = sc_probs[sc_y == 0]

        per_scenario_stats[sc] = {
            'total_samples': int(len(sc_probs)),
            'attack_samples': int(len(sc_att_probs)),
            'benign_samples': int(len(sc_ben_probs)),
            'all_probs': get_dist_stats(sc_probs),
            'attack_probs': get_dist_stats(sc_att_probs) if len(sc_att_probs) > 0 else None,
            'benign_probs': get_dist_stats(sc_ben_probs) if len(sc_ben_probs) > 0 else None
        }

    # 6. Exploratory Threshold Sweep (for diagnostic insight only, preserving tau=0.50)
    sweep_thresholds = [
        0.005, 0.008, 0.010, 0.012, 0.014, 0.015, 0.016, 0.018, 0.020,
        0.025, 0.030, 0.035, 0.040, 0.044, 0.050, 0.100, 0.200, 0.500
    ]

    sweep_results = []
    for tau in sweep_thresholds:
        yp = (y_prob >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()
        p = float(precision_score(y_true, yp, zero_division=0))
        r = float(recall_score(y_true, yp, zero_division=0))
        f1 = float(f1_score(y_true, yp, zero_division=0))
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        tnr = float(tn / (fp + tn)) if (fp + tn) > 0 else 0.0

        # Also calculate per scenario recall/false alarms at this threshold
        s8_probs = y_prob[df_preds['scenario_id'] == 'S8_benign_cloud_cover_shading']
        s8_fa = int(np.sum(s8_probs >= tau))
        s0_probs = y_prob[df_preds['scenario_id'] == 'S0_benign_baseline']
        s0_fa = int(np.sum(s0_probs >= tau))
        s3_probs = y_prob[(df_preds['scenario_id'] == 'S3_industroyer_pv_trip') & (y_true == 1)]
        s3_rec = float(np.mean(s3_probs >= tau)) if len(s3_probs) > 0 else 0.0

        sweep_results.append({
            'threshold': tau,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'precision': p,
            'recall': r,
            'f1_score': f1,
            'fpr': fpr,
            'tnr': tnr,
            's0_false_alarms': s0_fa,
            's8_false_alarms': s8_fa,
            's3_attack_recall': s3_rec
        })

    # Find empirical max F1 on this dataset (diagnostic only)
    best_f1_item = max(sweep_results, key=lambda x: x['f1_score'])

    output_payload = {
        'provenance': {
            'predictions_file': preds_path,
            'metadata_file': meta_path,
            'num_samples': num_samples,
            'num_attacks': num_attacks,
            'num_benign': num_benign,
            'prevalence': prevalence
        },
        'ranking_metrics': {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'baseline_pr_auc': prevalence,
            'mann_whitney_u': {
                'statistic': float(mwu_stat),
                'p_value': float(mwu_pval)
            }
        },
        'production_threshold_metrics_tau_0_50': {
            'threshold': 0.50,
            'confusion_matrix': {'tp': int(tp_05), 'fp': int(fp_05), 'tn': int(tn_05), 'fn': int(fn_05)},
            'precision': prec_05,
            'recall': rec_05,
            'f1_score': f1_05,
            'false_positive_rate': fpr_05,
            'true_negative_rate': tnr_05
        },
        'category_distributions': cat_distributions,
        'per_scenario_distributions': per_scenario_stats,
        'exploratory_threshold_sweep': sweep_results,
        'exploratory_peak_f1_operating_point': best_f1_item
    }

    out_file = os.path.join(out_dir, "threshold_analysis_metrics.json")
    with open(out_file, 'w') as f:
        json.dump(output_payload, f, indent=2)

    print(f"Analysis complete. Saved metrics to {out_file}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f} (Baseline: {prevalence:.4f})")
    print(f"MWU p-value: {mwu_pval:.4e}")
    print(f"Production tau=0.50: Recall={rec_05:.4f}, FPR={fpr_05:.4f}")
    print(f"Exploratory peak F1 threshold: tau={best_f1_item['threshold']} -> F1={best_f1_item['f1_score']:.4f}, Rec={best_f1_item['recall']:.4f}, FPR={best_f1_item['fpr']:.4f}, S8 False Alarms={best_f1_item['s8_false_alarms']}/150")

if __name__ == '__main__':
    run_threshold_independent_analysis()
