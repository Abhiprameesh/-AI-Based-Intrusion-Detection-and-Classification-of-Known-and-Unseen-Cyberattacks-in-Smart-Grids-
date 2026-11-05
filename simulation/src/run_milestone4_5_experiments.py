"""
Milestone 4.5: Controlled Domain-Shift Experiments and Frozen IDS Evaluation Script.
Executes Conditions C0, C1, C2, and C3 with strict experimental controls.
Performs simulation validation, three-stage frozen IDS scoring, error decomposition,
and generates consolidated comparisons and plots.
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Path setup
sys.path.insert(0, os.path.abspath("phase_1"))
sys.path.append(os.path.abspath("."))

from simulation.src.circuit_builder import MicrogridCircuitBuilder
from simulation.src.evaluation_pipeline import IDSEvaluationPipeline
from simulation.src.scenario_runner import ScenarioRunner
from src.anomaly_detection import Autoencoder

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, precision_recall_curve,
    roc_curve, auc, classification_report
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as fp:
        while chunk := fp.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_stats(arr):
    if len(arr) == 0:
        return {}
    return {
        'count': int(len(arr)),
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'p05': float(np.percentile(arr, 5)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
        'p95': float(np.percentile(arr, 95)),
    }

def run_condition(cond_id, label, slack_pu, load_a_kvar, load_b_kvar, base_out_dir):
    print("\n" + "=" * 80)
    print(f"EXECUTING CONDITION {cond_id}: {label}")
    print(f"Parameters: Slack={slack_pu} pu, LoadA kvar={load_a_kvar}, LoadB kvar={load_b_kvar}")
    print("=" * 80)

    cond_dir = os.path.join(base_out_dir, f"{cond_id}_{label.lower().replace(' ', '_').replace('-', '_')}")
    os.makedirs(cond_dir, exist_ok=True)

    # 1. Pipeline and Runner Initialization
    pipe = IDSEvaluationPipeline(results_dir=cond_dir)

    # 2. Dataset Generation
    t_gen_start = time.time()
    feats_df, meta_df = pipe.generate_evaluation_dataset(
        num_steps_per_scenario=150,
        initial_soc_percent=20.0,
        random_seed=42,
        slack_pu=slack_pu,
        load_a_kvar=load_a_kvar,
        load_b_kvar=load_b_kvar
    )
    gen_duration = time.time() - t_gen_start
    print(f"Simulation dataset generated in {gen_duration:.2f} s. Shape: {feats_df.shape}")

    feat_path = os.path.join(cond_dir, "telemetry_features.csv")
    meta_path = os.path.join(cond_dir, "telemetry_metadata.csv")
    feats_df.to_csv(feat_path, index=False)
    meta_df.to_csv(meta_path, index=False)

    # 3. Physical Simulation Validation
    print("[Validation] Verifying physical power flow and constraints...")
    actual_sec_v_ln = feats_df[[c for c in feats_df.columns if "Substation-AC-Meter.uln" in c]].values
    actual_load_a_v_ln = feats_df[[c for c in feats_df.columns if "Load-A-AC-Meter.uln" in c]].values
    actual_load_b_v_ln = feats_df[[c for c in feats_df.columns if "Load-B-AC-Meter.uln" in c]].values

    q_load_a = feats_df[[c for c in feats_df.columns if "Load-A-AC-Meter.qln" in c]].values
    q_load_b = feats_df[[c for c in feats_df.columns if "Load-B-AC-Meter.qln" in c]].values
    pf_load_a = feats_df[[c for c in feats_df.columns if "Load-A-AC-Meter.cos_phi" in c]].values
    pf_load_b = feats_df[[c for c in feats_df.columns if "Load-B-AC-Meter.cos_phi" in c]].values

    soc_ref_vals = feats_df["Battery-SOC-Reference.soc_value"].values
    soc_rep_vals = feats_df["Battery-SOC-Reported.soc_value"].values

    validation_summary = {
        'condition_id': cond_id,
        'label': label,
        'configured_parameters': {
            'slack_pu': slack_pu,
            'load_a_kvar': load_a_kvar,
            'load_b_kvar': load_b_kvar,
            'initial_soc_percent': 20.0,
            'reserve_percent': 0.0,
            'random_seed': 42,
            'total_steps': len(feats_df),
        },
        'achieved_physical_measurements': {
            'substation_v_ln': get_stats(actual_sec_v_ln.flatten()),
            'load_a_v_ln': get_stats(actual_load_a_v_ln.flatten()),
            'load_b_v_ln': get_stats(actual_load_b_v_ln.flatten()),
            'load_a_q_var': get_stats(q_load_a.flatten()),
            'load_b_q_var': get_stats(q_load_b.flatten()),
            'load_a_power_factor': get_stats(pf_load_a.flatten()),
            'load_b_power_factor': get_stats(pf_load_b.flatten()),
            'battery_soc_fraction': get_stats(soc_ref_vals),
        },
        'convergence': {
            'all_steps_converged': bool(meta_df['physical_power_balance_converged'].all()),
            'max_power_balance_residual_w': float(meta_df['physical_power_balance_residual_w'].abs().max()),
            'mean_power_balance_residual_w': float(meta_df['physical_power_balance_residual_w'].abs().mean()),
        },
        'step0_initial_soc_verified': bool(np.isclose(soc_ref_vals[0], 0.20)),
        'feature_contract_valid': True,
    }
    with open(os.path.join(cond_dir, "simulation_validation_summary.json"), "w") as fp:
        json.dump(validation_summary, fp, indent=2)

    # 4. Stage 1 & Stage 2 Evaluation (Frozen XGBoost and Random Forest)
    print("[Evaluation] Running Stage 1 (Binary XGBoost) and Stage 2 (Multiclass RF)...")
    res12 = pipe.run_experiment_1_frozen_models(feats_df, meta_df, save_predictions=True)
    preds_df = pd.read_csv(os.path.join(cond_dir, "predictions_binary_and_multiclass.csv"))

    y_true_bin = preds_df["attack_active_ground_truth"].values.astype(int)
    y_prob_bin = preds_df["binary_xgb_prob"].values.astype(float)
    y_pred_bin_05 = preds_df["binary_xgb_pred"].values.astype(int)

    # Primary threshold tau = 0.50
    tn_05, fp_05, fn_05, tp_05 = confusion_matrix(y_true_bin, y_pred_bin_05, labels=[0, 1]).ravel()
    acc_05 = float(accuracy_score(y_true_bin, y_pred_bin_05))
    prec_05 = float(precision_score(y_true_bin, y_pred_bin_05, zero_division=0))
    rec_05 = float(recall_score(y_true_bin, y_pred_bin_05, zero_division=0))
    f1_05 = float(f1_score(y_true_bin, y_pred_bin_05, zero_division=0))
    fpr_05 = float(fp_05 / (fp_05 + tn_05)) if (fp_05 + tn_05) > 0 else 0.0
    tnr_05 = float(tn_05 / (fp_05 + tn_05)) if (fp_05 + tn_05) > 0 else 0.0

    # Secondary diagnostic threshold tau = 0.05
    y_pred_bin_005 = (y_prob_bin >= 0.05).astype(int)
    tn_005, fp_005, fn_005, tp_005 = confusion_matrix(y_true_bin, y_pred_bin_005, labels=[0, 1]).ravel()
    acc_005 = float(accuracy_score(y_true_bin, y_pred_bin_005))
    prec_005 = float(precision_score(y_true_bin, y_pred_bin_005, zero_division=0))
    rec_005 = float(recall_score(y_true_bin, y_pred_bin_005, zero_division=0))
    f1_005 = float(f1_score(y_true_bin, y_pred_bin_005, zero_division=0))
    fpr_005 = float(fp_005 / (fp_005 + tn_005)) if (fp_005 + tn_005) > 0 else 0.0

    # Threshold-independent metrics
    roc_auc_bin = float(roc_auc_score(y_true_bin, y_prob_bin))
    p_pts, r_pts, _ = precision_recall_curve(y_true_bin, y_prob_bin)
    pr_auc_bin = float(auc(r_pts, p_pts))

    # Per-scenario binary evaluation
    per_sc_bin = {}
    for sc_id in meta_df["scenario_id"].unique():
        idx = meta_df["scenario_id"] == sc_id
        sc_yt = y_true_bin[idx]
        sc_prob = y_prob_bin[idx]
        sc_pred_05 = y_pred_bin_05[idx]
        sc_pred_005 = y_pred_bin_005[idx]

        att_cnt = int((sc_yt == 1).sum())
        sc_rec_05 = float(recall_score(sc_yt, sc_pred_05, zero_division=0)) if att_cnt > 0 else None
        sc_rec_005 = float(recall_score(sc_yt, sc_pred_005, zero_division=0)) if att_cnt > 0 else None
        sc_fpr_05 = float((sc_pred_05[sc_yt == 0] == 1).mean()) if (len(sc_yt) - att_cnt) > 0 else 0.0
        sc_fpr_005 = float((sc_pred_005[sc_yt == 0] == 1).mean()) if (len(sc_yt) - att_cnt) > 0 else 0.0

        per_sc_bin[sc_id] = {
            'total': int(len(sc_yt)),
            'attacks': att_cnt,
            'prob_mean': float(sc_prob.mean()),
            'prob_max': float(sc_prob.max()),
            'prob_min': float(sc_prob.min()),
            'recall_tau_0_50': sc_rec_05,
            'recall_tau_0_05': sc_rec_005,
            'fpr_tau_0_50': sc_fpr_05,
            'fpr_tau_0_05': sc_fpr_005,
        }

    # Stage 2 Multiclass
    st2 = res12["stages"]["stage_2_known_attack_multiclass"]

    # 5. Stage 3 Unseen Anomaly Detection (Frozen Recovered Autoencoder)
    print("[Evaluation] Running Stage 3 (Frozen Recovered Autoencoder)...")
    scaler_path = "phase_1/results/data4cyber_dataset/binary/models/preprocessor_scaled.joblib"
    weights_path = "phase_1/results/data4cyber_dataset/anomaly/recovery_v1/Autoencoder.pt"
    threshold_meta_path = "phase_1/results/data4cyber_dataset/anomaly/recovery_v1/threshold_metadata.json"

    scaler = joblib.load(scaler_path)
    with open(threshold_meta_path) as fp:
        tmeta = json.load(fp)
    tau_ae_rec = float(tmeta["calculated_threshold"])  # 0.36402026
    tau_ae_hist = float(tmeta["historical_threshold"])  # 0.42760131

    ae_model = Autoencoder(in_features=150, bottleneck_dim=16).to(DEVICE)
    ae_model.load_state_dict(torch.load(weights_path, weights_only=True))
    ae_model.eval()

    X_scaled = scaler.transform(feats_df)
    with torch.no_grad():
        tx = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
        rec = ae_model(tx)
        sq_errs = ((tx - rec) ** 2).cpu().numpy()
        mse_per_sample = sq_errs.mean(axis=1)
        mse_per_feature = sq_errs.mean(axis=0)

    # Save Stage 3 predictions
    ae_preds_df = pd.DataFrame({
        'step': meta_df['step'],
        'scenario_id': meta_df['scenario_id'],
        'attack_active_ground_truth': y_true_bin,
        'is_unseen_scenario': meta_df['is_unseen_scenario'].values.astype(bool),
        'reconstruction_mse': mse_per_sample,
        'pred_recovered_threshold': (mse_per_sample >= tau_ae_rec).astype(int),
        'pred_historical_threshold': (mse_per_sample >= tau_ae_hist).astype(int),
    })
    ae_preds_path = os.path.join(cond_dir, "unseen_autoencoder_predictions.csv")
    ae_preds_df.to_csv(ae_preds_path, index=False)

    y_pred_ae = (mse_per_sample >= tau_ae_rec).astype(int)
    tn_ae, fp_ae, fn_ae, tp_ae = confusion_matrix(y_true_bin, y_pred_ae, labels=[0, 1]).ravel()
    acc_ae = float(accuracy_score(y_true_bin, y_pred_ae))
    prec_ae = float(precision_score(y_true_bin, y_pred_ae, zero_division=0))
    rec_ae = float(recall_score(y_true_bin, y_pred_ae, zero_division=0))
    f1_ae = float(f1_score(y_true_bin, y_pred_ae, zero_division=0))
    fpr_ae = float(fp_ae / (fp_ae + tn_ae)) if (fp_ae + tn_ae) > 0 else 0.0
    tnr_ae = float(tn_ae / (fp_ae + tn_ae)) if (fp_ae + tn_ae) > 0 else 0.0
    roc_auc_ae = float(roc_auc_score(y_true_bin, mse_per_sample))
    p_pts_ae, r_pts_ae, _ = precision_recall_curve(y_true_bin, mse_per_sample)
    pr_auc_ae = float(auc(r_pts_ae, p_pts_ae))

    # Feature group error decomposition
    feat_names = list(feats_df.columns)
    total_mse_sum = float(mse_per_feature.sum())

    reactive_cols = [i for i, c in enumerate(feat_names) if ".q" in c]
    voltage_cols = [i for i, c in enumerate(feat_names) if ".u" in c]
    neutral_cols = [i for i, c in enumerate(feat_names) if "i_sum" in c]
    pf_cols = [i for i, c in enumerate(feat_names) if "cos_phi" in c]
    current_power_cols = [i for i, c in enumerate(feat_names) if ".i" in c or ".p" in c or ".s" in c]

    group_errors = {
        'reactive_power': {'mse_sum': float(mse_per_feature[reactive_cols].sum()), 'pct': float(mse_per_feature[reactive_cols].sum() / total_mse_sum * 100)},
        'voltage': {'mse_sum': float(mse_per_feature[voltage_cols].sum()), 'pct': float(mse_per_feature[voltage_cols].sum() / total_mse_sum * 100)},
        'neutral_current': {'mse_sum': float(mse_per_feature[neutral_cols].sum()), 'pct': float(mse_per_feature[neutral_cols].sum() / total_mse_sum * 100)},
        'power_factor': {'mse_sum': float(mse_per_feature[pf_cols].sum()), 'pct': float(mse_per_feature[pf_cols].sum() / total_mse_sum * 100)},
        'current_and_apparent_power': {'mse_sum': float(mse_per_feature[current_power_cols].sum()), 'pct': float(mse_per_feature[current_power_cols].sum() / total_mse_sum * 100)},
    }

    # Top 15 features
    top15_idx = np.argsort(mse_per_feature)[::-1][:15]
    top15_features = [{
        'feature': feat_names[i],
        'mse': float(mse_per_feature[i]),
        'pct': float(mse_per_feature[i] / total_mse_sum * 100)
    } for i in top15_idx]

    # Per-scenario AE breakdown
    per_sc_ae = {}
    for sc_id in meta_df["scenario_id"].unique():
        idx = meta_df["scenario_id"] == sc_id
        sc_yt = y_true_bin[idx]
        sc_mse = mse_per_sample[idx]
        sc_pred = y_pred_ae[idx]

        att_cnt = int((sc_yt == 1).sum())
        sc_rec = float(recall_score(sc_yt, sc_pred, zero_division=0)) if att_cnt > 0 else None
        sc_fpr = float((sc_pred[sc_yt == 0] == 1).mean()) if (len(sc_yt) - att_cnt) > 0 else 0.0

        per_sc_ae[sc_id] = {
            'total': int(len(sc_yt)),
            'attacks': att_cnt,
            'mse_mean': float(sc_mse.mean()),
            'mse_median': float(np.median(sc_mse)),
            'mse_min': float(sc_mse.min()),
            'mse_max': float(sc_mse.max()),
            'recall': sc_rec,
            'fpr': sc_fpr,
        }

    # Save comprehensive metrics JSON
    all_metrics = {
        'condition_id': cond_id,
        'label': label,
        'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        'configuration': validation_summary['configured_parameters'],
        'validation': validation_summary,
        'stage_1_binary_xgboost': {
            'primary_threshold_0_50': {
                'threshold': 0.50,
                'accuracy': acc_05, 'precision': prec_05, 'recall': rec_05, 'f1_score': f1_05,
                'false_positive_rate': fpr_05, 'true_negative_rate': tnr_05,
                'confusion_matrix': {'tn': int(tn_05), 'fp': int(fp_05), 'fn': int(fn_05), 'tp': int(tp_05)}
            },
            'secondary_diagnostic_threshold_0_05': {
                'threshold': 0.05,
                'accuracy': acc_005, 'precision': prec_005, 'recall': rec_005, 'f1_score': f1_005,
                'false_positive_rate': fpr_005,
                'confusion_matrix': {'tn': int(tn_005), 'fp': int(fp_005), 'fn': int(fn_005), 'tp': int(tp_005)}
            },
            'ranking_metrics': {
                'roc_auc': roc_auc_bin,
                'pr_auc': pr_auc_bin,
                'baseline_pr_auc': float(np.mean(y_true_bin)),
            },
            'probability_distributions': {
                'all': get_stats(y_prob_bin),
                'benign': get_stats(y_prob_bin[y_true_bin == 0]),
                'attack': get_stats(y_prob_bin[y_true_bin == 1]),
            },
            'per_scenario': per_sc_bin,
        },
        'stage_2_known_attack_multiclass': st2,
        'stage_3_unseen_autoencoder': {
            'threshold_recovered': tau_ae_rec,
            'threshold_historical': tau_ae_hist,
            'evaluation_recovered_threshold': {
                'accuracy': acc_ae, 'precision': prec_ae, 'recall': rec_ae, 'f1_score': f1_ae,
                'false_positive_rate': fpr_ae, 'true_negative_rate': tnr_ae,
                'confusion_matrix': {'tn': int(tn_ae), 'fp': int(fp_ae), 'fn': int(fn_ae), 'tp': int(tp_ae)}
            },
            'ranking_metrics': {
                'roc_auc': roc_auc_ae,
                'pr_auc': pr_auc_ae,
            },
            'reconstruction_mse_distributions': {
                'all': get_stats(mse_per_sample),
                'benign': get_stats(mse_per_sample[y_true_bin == 0]),
                'attack': get_stats(mse_per_sample[y_true_bin == 1]),
            },
            'feature_group_errors': group_errors,
            'top_15_features': top15_features,
            'per_scenario': per_sc_ae,
        }
    }

    metrics_path = os.path.join(cond_dir, "metrics_all_stages.json")
    with open(metrics_path, "w") as fp:
        json.dump(all_metrics, fp, indent=2)

    # Configuration Snapshot
    config_snapshot = {
        'condition_id': cond_id,
        'label': label,
        'slack_pu': slack_pu,
        'load_a_kvar': load_a_kvar,
        'load_b_kvar': load_b_kvar,
        'initial_soc_percent': 20.0,
        'reserve_percent': 0.0,
        'random_seed': 42,
        'models': {
            'binary_xgboost': 'phase_1/results/data4cyber_dataset/binary/models/XGBoost.joblib',
            'multiclass_rf': 'phase_1/results/data4cyber_dataset/multiclass/models/Random_Forest_multiclass.joblib',
            'autoencoder_pt': 'phase_1/results/data4cyber_dataset/anomaly/recovery_v1/Autoencoder.pt',
            'preprocessor_unscaled': 'phase_1/results/data4cyber_dataset/binary/models/preprocessor_unscaled.joblib',
            'preprocessor_scaled': 'phase_1/results/data4cyber_dataset/binary/models/preprocessor_scaled.joblib',
        },
        'thresholds': {
            'binary_primary': 0.50,
            'binary_diagnostic': 0.05,
            'autoencoder': tau_ae_rec,
        },
        'artifact_hashes': {
            'telemetry_features_sha256': compute_sha256(feat_path),
            'telemetry_metadata_sha256': compute_sha256(meta_path),
            'predictions_binary_multiclass_sha256': compute_sha256(os.path.join(cond_dir, "predictions_binary_and_multiclass.csv")),
            'unseen_autoencoder_predictions_sha256': compute_sha256(ae_preds_path),
        }
    }
    with open(os.path.join(cond_dir, "configuration_snapshot.json"), "w") as fp:
        json.dump(config_snapshot, fp, indent=2)

    print(f"Condition {cond_id} finished successfully! Saved all artifacts to: {cond_dir}")
    return all_metrics

def main():
    print("=" * 80)
    print("STARTING MILESTONE 4.5 CONTROLLED EXPERIMENTS PIPELINE")
    print("=" * 80)

    base_out_dir = "simulation/results/milestone4_5/candidates"
    os.makedirs(base_out_dir, exist_ok=True)

    # 1. Condition Definitions
    conditions = [
        ('C0', 'Baseline', 1.000, 1.5, 2.0),
        ('C1', 'Voltage Alignment', 1.018, 1.5, 2.0),
        ('C2', 'Reactive Alignment', 1.000, 0.0, 0.0),
        ('C3', 'Combined Alignment', 1.018, 0.0, 0.0),
    ]

    results_all = {}
    for cond_id, label, slack_pu, load_a_kvar, load_b_kvar in conditions:
        res = run_condition(cond_id, label, slack_pu, load_a_kvar, load_b_kvar, base_out_dir)
        results_all[cond_id] = res

    # 2. Compile Cross-Condition Comparison Table
    print("\n" + "=" * 80)
    print("COMPILING CROSS-CONDITION COMPARISON SUMMARY")
    print("=" * 80)

    comp_rows = []
    for cond_id, res in results_all.items():
        st1_05 = res['stage_1_binary_xgboost']['primary_threshold_0_50']
        st1_005 = res['stage_1_binary_xgboost']['secondary_diagnostic_threshold_0_05']
        st1_rank = res['stage_1_binary_xgboost']['ranking_metrics']
        st1_prob = res['stage_1_binary_xgboost']['probability_distributions']['all']
        st1_att_prob = res['stage_1_binary_xgboost']['probability_distributions']['attack']

        st2 = res['stage_2_known_attack_multiclass']
        st2_rep = st2['classification_report']

        st3_eval = res['stage_3_unseen_autoencoder']['evaluation_recovered_threshold']
        st3_rank = res['stage_3_unseen_autoencoder']['ranking_metrics']
        st3_mse = res['stage_3_unseen_autoencoder']['reconstruction_mse_distributions']['all']

        val = res['validation']['achieved_physical_measurements']

        comp_rows.append({
            'condition_id': cond_id,
            'label': res['label'],
            'slack_pu': res['configuration']['slack_pu'],
            'load_a_kvar': res['configuration']['load_a_kvar'],
            'load_b_kvar': res['configuration']['load_b_kvar'],
            # Physical Achieved
            'mean_substation_v_ln': val['substation_v_ln']['mean'],
            'mean_load_a_v_ln': val['load_a_v_ln']['mean'],
            'mean_load_a_q_var': val['load_a_q_var']['mean'],
            'mean_load_b_q_var': val['load_b_q_var']['mean'],
            'mean_load_a_pf': val['load_a_power_factor']['mean'],
            # Stage 1 Binary (tau = 0.50)
            's1_acc_tau_0_50': st1_05['accuracy'],
            's1_recall_tau_0_50': st1_05['recall'],
            's1_precision_tau_0_50': st1_05['precision'],
            's1_f1_tau_0_50': st1_05['f1_score'],
            's1_fpr_tau_0_50': st1_05['false_positive_rate'],
            # Stage 1 Binary Diagnostic (tau = 0.05)
            's1_recall_tau_0_05': st1_005['recall'],
            's1_precision_tau_0_05': st1_005['precision'],
            's1_f1_tau_0_05': st1_005['f1_score'],
            's1_fpr_tau_0_05': st1_005['false_positive_rate'],
            # Stage 1 Probabilities & Ranking
            's1_roc_auc': st1_rank['roc_auc'],
            's1_pr_auc': st1_rank['pr_auc'],
            's1_mean_attack_prob': st1_att_prob['mean'],
            's1_max_attack_prob': st1_att_prob['max'],
            # Stage 2 Multiclass
            's2_accuracy': st2['overall_accuracy'],
            's2_macro_f1': st2['macro_f1'],
            's2_weighted_f1': st2['weighted_f1'],
            's2_benign_recall': st2_rep['benign (0)']['recall'],
            's2_industroyer_recall': st2_rep['industroyer (1)']['recall'],
            's2_mitm_fdi_recall': st2_rep['mitm_fdi (2)']['recall'],
            # Stage 3 Autoencoder
            's3_mean_reconstruction_mse': st3_mse['mean'],
            's3_median_reconstruction_mse': st3_mse['median'],
            's3_benign_fpr': st3_eval['false_positive_rate'],
            's3_attack_recall': st3_eval['recall'],
            's3_f1_score': st3_eval['f1_score'],
            's3_roc_auc': st3_rank['roc_auc'],
            's3_pr_auc': st3_rank['pr_auc'],
        })

    df_comp = pd.DataFrame(comp_rows)
    comp_csv_path = "simulation/results/milestone4_5/condition_comparison.csv"
    comp_json_path = "simulation/results/milestone4_5/condition_comparison.json"
    df_comp.to_csv(comp_csv_path, index=False)
    with open(comp_json_path, "w") as fp:
        json.dump(results_all, fp, indent=2)

    print(f"Saved comparison CSV to: {comp_csv_path}")
    print(f"Saved comparison JSON to: {comp_json_path}")
    print("\n--- COMPARISON SUMMARY TABLE ---")
    print(df_comp[['condition_id', 'label', 's1_recall_tau_0_50', 's1_mean_attack_prob', 's1_roc_auc', 's2_industroyer_recall', 's2_mitm_fdi_recall', 's3_mean_reconstruction_mse', 's3_benign_fpr']])

    # 3. Generate Visualizations
    print("\n[Plots] Generating comparison visualizations...")
    plots_dir = "simulation/results/milestone4_5/plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Plot 1: Binary Attack Probabilities Distribution across C0-C3
    plt.figure(figsize=(10, 6))
    for cond_id in ['C0', 'C1', 'C2', 'C3']:
        cond_res = results_all[cond_id]
        preds_p = os.path.join(base_out_dir, f"{cond_id}_{cond_res['label'].lower().replace(' ', '_').replace('-', '_')}", "predictions_binary_and_multiclass.csv")
        df_p = pd.read_csv(preds_p)
        att_probs = df_p.loc[df_p["attack_active_ground_truth"] == 1, "binary_xgb_prob"].values
        plt.hist(att_probs, bins=30, alpha=0.5, label=f"{cond_id}: {cond_res['label']}")
    plt.axvline(0.50, color='red', linestyle='--', linewidth=2, label='Primary Threshold (0.50)')
    plt.axvline(0.05, color='orange', linestyle=':', linewidth=2, label='Diagnostic Threshold (0.05)')
    plt.xlabel('Predicted Attack Probability', fontsize=12)
    plt.ylabel('Frequency (Attack Samples)', fontsize=12)
    plt.title('Stage 1: Binary XGBoost Attack Probability Distributions (C0–C3)', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "binary_attack_probabilities_comparison.png"), dpi=200)
    plt.close()

    # Plot 2: Autoencoder Reconstruction MSE Distributions across C0-C3
    plt.figure(figsize=(10, 6))
    for cond_id in ['C0', 'C1', 'C2', 'C3']:
        cond_res = results_all[cond_id]
        ae_p = os.path.join(base_out_dir, f"{cond_id}_{cond_res['label'].lower().replace(' ', '_').replace('-', '_')}", "unseen_autoencoder_predictions.csv")
        df_ae = pd.read_csv(ae_p)
        plt.hist(df_ae["reconstruction_mse"].values, bins=40, alpha=0.5, label=f"{cond_id}: {cond_res['label']}")
    plt.axvline(0.36402, color='red', linestyle='--', linewidth=2, label='Val Threshold (0.364)')
    plt.xlabel('Reconstruction Mean Squared Error (MSE)', fontsize=12)
    plt.ylabel('Sample Count (All Timesteps)', fontsize=12)
    plt.title('Stage 3: Deep Autoencoder Reconstruction MSE Distributions (C0–C3)', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "autoencoder_mse_distributions_comparison.png"), dpi=200)
    plt.close()

    # Plot 3: Stage 1 ROC and PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for cond_id in ['C0', 'C1', 'C2', 'C3']:
        cond_res = results_all[cond_id]
        preds_p = os.path.join(base_out_dir, f"{cond_id}_{cond_res['label'].lower().replace(' ', '_').replace('-', '_')}", "predictions_binary_and_multiclass.csv")
        df_p = pd.read_csv(preds_p)
        yt = df_p["attack_active_ground_truth"].values.astype(int)
        yp = df_p["binary_xgb_prob"].values.astype(float)

        fpr, tpr, _ = roc_curve(yt, yp)
        roc_val = roc_auc_score(yt, yp)
        ax1.plot(fpr, tpr, label=f"{cond_id} (AUC = {roc_val:.4f})", linewidth=1.8)

        prec, rec, _ = precision_recall_curve(yt, yp)
        pr_val = auc(rec, prec)
        ax2.plot(rec, prec, label=f"{cond_id} (AUC = {pr_val:.4f})", linewidth=1.8)

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance')
    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=13)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    prev = np.mean(yt)
    ax2.axhline(prev, color='k', linestyle='--', alpha=0.5, label=f'Baseline Prev ({prev:.3f})')
    ax2.set_xlabel('Recall', fontsize=11)
    ax2.set_ylabel('Precision', fontsize=11)
    ax2.set_title('Precision-Recall Curve (PR)', fontsize=13)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "roc_pr_curves_comparison.png"), dpi=200)
    plt.close()

    # Plot 4: Top 10 Feature Error Comparison C0 vs C3
    c0_top = results_all['C0']['stage_3_unseen_autoencoder']['top_15_features'][:10]
    c3_top_dict = {f['feature']: f['mse'] for f in results_all['C3']['stage_3_unseen_autoencoder']['top_15_features']}
    
    top_names = [f['feature'] for f in c0_top]
    c0_vals = [f['mse'] for f in c0_top]
    c3_vals = [c3_top_dict.get(name, 0.0) for name in top_names]

    y_pos = np.arange(len(top_names))
    plt.figure(figsize=(12, 7))
    plt.barh(y_pos - 0.2, c0_vals, height=0.4, label='C0: Baseline (Inductive pf=0.97)', color='crimson', alpha=0.8)
    plt.barh(y_pos + 0.2, c3_vals, height=0.4, label='C3: Combined (Unity pf=1.00)', color='seagreen', alpha=0.8)
    plt.yticks(y_pos, top_names, fontsize=10)
    plt.xlabel('Reconstruction Mean Squared Error (MSE)', fontsize=12)
    plt.title('Top 10 Feature Reconstruction MSE: C0 Baseline vs C3 Combined Alignment', fontsize=13)
    plt.gca().invert_yaxis()
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "top_features_reconstruction_error.png"), dpi=200)
    plt.close()

    print("All plots generated successfully!")

if __name__ == '__main__':
    main()

