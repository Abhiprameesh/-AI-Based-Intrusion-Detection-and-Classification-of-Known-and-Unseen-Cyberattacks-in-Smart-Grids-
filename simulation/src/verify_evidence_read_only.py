"""
Milestone 4.2 Read-Only Evidence Verification Script
Inspects all relevant files, confirms alignment, recalculates metrics,
verifies Phase 1 hashes, and audits FDI failure characteristics without modifying any files.
"""

import os
import sys
import json
import hashlib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify():
    print("=" * 70)
    print("MILESTONE 4.2: READ-ONLY EVIDENCE VERIFICATION")
    print("=" * 70)

    # 1. Protected Phase 1 Artifact Hashes
    baseline_hashes = {
        'binary_xgboost': 'f14e1c87d3640bcc8bec44d59925c2fd91b7c630a4cfe5e8b1a8d912e8a67e02',
        'binary_random_forest': '6a66af5e955bc0a51413b96da549f78bbeefbadffcb19476da90da9263b172a4',
        'binary_preprocessor_unscaled': '2c5ef4a896e75d237f4c59ca35cd3c3d6afd4b2abe23edf16bbc4e5f5d694b12',
        'binary_preprocessor_scaled': 'd80f1484077092c1ba64c9ceec793041e8b9cf037b4f9cd101b6d2b236608ac5',
        'multiclass_xgboost': '2ec535dac31d9eafc31252fee3cc0335d8fd33bab47db527efa4e85a392459b9',
        'multiclass_random_forest': '8987a92549af94330e33266c2eada17ee7ece6a037c43ad3e8925475d8f47501',
        'multiclass_preprocessor_unscaled': '2c5ef4a896e75d237f4c59ca35cd3c3d6afd4b2abe23edf16bbc4e5f5d694b12',
        'unseen_metrics_json': '72a343107bca8dc3b916447832a7dd24398ae862cea0572012cb71c57bff30fa',
    }
    phase1_paths = {
        'binary_xgboost': 'phase_1/results/data4cyber_dataset/binary/models/XGBoost.joblib',
        'binary_random_forest': 'phase_1/results/data4cyber_dataset/binary/models/Random_Forest.joblib',
        'binary_preprocessor_unscaled': 'phase_1/results/data4cyber_dataset/binary/models/preprocessor_unscaled.joblib',
        'binary_preprocessor_scaled': 'phase_1/results/data4cyber_dataset/binary/models/preprocessor_scaled.joblib',
        'multiclass_xgboost': 'phase_1/results/data4cyber_dataset/multiclass/models/XGBoost_multiclass.joblib',
        'multiclass_random_forest': 'phase_1/results/data4cyber_dataset/multiclass/models/Random_Forest_multiclass.joblib',
        'multiclass_preprocessor_unscaled': 'phase_1/results/data4cyber_dataset/multiclass/models/preprocessor_unscaled.joblib',
        'unseen_metrics_json': 'phase_1/results/data4cyber_dataset/unseen_attack/unseen_attack_MitM_FDI.json',
    }

    print("\n--- Phase 1 Artifact Hashes ---")
    all_p1_match = True
    for k, rel_p in phase1_paths.items():
        abs_p = os.path.join(ROOT_DIR, rel_p)
        assert os.path.exists(abs_p), f"Missing: {rel_p}"
        with open(abs_p, "rb") as fp:
            h = hashlib.sha256(fp.read()).hexdigest()
        match = (h == baseline_hashes[k])
        if not match:
            all_p1_match = False
        print(f"  {k}: {h[:16]}... Match: {match}")
    print(f"All Phase 1 Hashes Match Baseline: {all_p1_match}")

    # 2. Files and shapes
    m4_dir = os.path.join(ROOT_DIR, "simulation", "results", "milestone4")
    feats_df = pd.read_csv(os.path.join(m4_dir, "telemetry_features.csv"))
    meta_df = pd.read_csv(os.path.join(m4_dir, "telemetry_metadata.csv"))
    preds_df = pd.read_csv(os.path.join(m4_dir, "predictions_binary_and_multiclass.csv"))

    print("\n--- File Shapes and Alignment ---")
    print(f"  telemetry_features.csv: {feats_df.shape}")
    print(f"  telemetry_metadata.csv: {meta_df.shape}")
    print(f"  predictions_binary_and_multiclass.csv: {preds_df.shape}")

    assert len(feats_df) == 1350 and len(meta_df) == 1350 and len(preds_df) == 1350
    step_aligned = (preds_df["step"] == meta_df["step"]).all()
    scenario_aligned = (preds_df["scenario_id"] == meta_df["scenario_id"]).all()
    print(f"  Step Alignment: {step_aligned}")
    print(f"  Scenario Alignment: {scenario_aligned}")

    # 3. Binary Metrics
    y_true_bin = preds_df["attack_active_ground_truth"].values
    y_pred_bin = preds_df["binary_xgb_pred"].values
    y_prob_bin = preds_df["binary_xgb_prob"].values

    cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm_bin.ravel()
    print("\n--- Binary XGBoost Metrics ---")
    print(f"  TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print(f"  Accuracy: {accuracy_score(y_true_bin, y_pred_bin):.6f}")
    print(f"  Precision: {precision_score(y_true_bin, y_pred_bin, zero_division=0):.6f}")
    print(f"  Recall: {recall_score(y_true_bin, y_pred_bin, zero_division=0):.6f}")
    print(f"  F1: {f1_score(y_true_bin, y_pred_bin, zero_division=0):.6f}")
    print(f"  FPR: {fp / (fp + tn):.6f}")
    print(f"  TNR: {tn / (fp + tn):.6f}")
    print(f"  Max Attack Prob: {y_prob_bin.max():.6f}")
    print(f"  Min Attack Prob: {y_prob_bin.min():.6f}")
    print(f"  All Predictions Benign (All 0s): {(y_pred_bin == 0).all()}")

    # 4. Multiclass Random Forest Inspection
    is_supp = preds_df["multiclass_is_supported"].values.astype(bool)
    print("\n--- Multiclass Sample Breakdown ---")
    print(f"  Total samples: {len(preds_df)}")
    print(f"  Supported samples: {is_supp.sum()}")
    print(f"  Excluded samples: {(~is_supp).sum()}")

    # Breakdown of excluded scenarios
    excl_scenarios = preds_df[~is_supp]["scenario_id"].value_counts().to_dict()
    print(f"  Excluded Scenarios: {excl_scenarios}")

    # Inspect Stage 2 predictions by scenario and true class
    # Classes: 0: benign, 1: industroyer, 2: mitm_fdi, 3: mqtt
    import joblib
    rf_multi_path = os.path.join(ROOT_DIR, "phase_1/results/data4cyber_dataset/multiclass/models/Random_Forest_multiclass.joblib")
    rf_prep_path = os.path.join(ROOT_DIR, "phase_1/results/data4cyber_dataset/multiclass/models/preprocessor_unscaled.joblib")
    rf_multi = joblib.load(rf_multi_path)
    rf_prep = joblib.load(rf_prep_path)

    X_supp = rf_prep.transform(feats_df)[is_supp]
    y_supp_true = preds_df.loc[is_supp, "multiclass_true_class"].values
    y_supp_pred = rf_multi.predict(X_supp)
    y_supp_probs = rf_multi.predict_proba(X_supp)

    cm_multi = confusion_matrix(y_supp_true, y_supp_pred, labels=[0, 1, 2, 3])
    print("\n--- Multiclass Confusion Matrix (4x4) ---")
    print("  [0: Benign, 1: Industroyer, 2: MitM FDI, 3: MQTT]")
    print(cm_multi)

    print("\n--- Multiclass FDI Details (Class 2) ---")
    fdi_mask = (y_supp_true == 2)
    print(f"  Total MitM FDI active attack samples: {fdi_mask.sum()}")
    print(f"  Predicted as Class 0 (Benign): {(y_supp_pred[fdi_mask] == 0).sum()}")
    print(f"  Predicted as Class 1 (Industroyer): {(y_supp_pred[fdi_mask] == 1).sum()}")
    print(f"  Predicted as Class 2 (MitM FDI): {(y_supp_pred[fdi_mask] == 2).sum()}")
    print(f"  Predicted as Class 3 (MQTT): {(y_supp_pred[fdi_mask] == 3).sum()}")
    
    # Check probabilities assigned to FDI samples
    fdi_probs = y_supp_probs[fdi_mask]
    print(f"  Mean Prob(Class 0 - Benign) on FDI samples: {fdi_probs[:, 0].mean():.4f}")
    print(f"  Mean Prob(Class 1 - Industroyer) on FDI samples: {fdi_probs[:, 1].mean():.4f}")
    print(f"  Mean Prob(Class 2 - MitM FDI) on FDI samples: {fdi_probs[:, 2].mean():.4f}")
    print(f"  Mean Prob(Class 3 - MQTT) on FDI samples: {fdi_probs[:, 3].mean():.4f}")

    # 5. Check Autoencoder file existence
    ae_path = os.path.join(ROOT_DIR, "phase_1/results/data4cyber_dataset/anomaly/Autoencoder.pt")
    print("\n--- Stage 3 Autoencoder Artifact Audit ---")
    print(f"  Target path: {ae_path}")
    print(f"  Exists: {os.path.exists(ae_path)}")
    unseen_json = os.path.join(ROOT_DIR, "phase_1/results/data4cyber_dataset/unseen_attack/unseen_attack_MitM_FDI.json")
    print(f"  Unseen JSON exists: {os.path.exists(unseen_json)}")
    if os.path.exists(unseen_json):
        with open(unseen_json, "r") as fp:
            u_data = json.load(fp)
        print(f"  Unseen JSON threshold: {u_data.get('threshold')}")
        print(f"  Unseen JSON approach: {u_data.get('approach')}")

    print("=" * 70)
    print("VERIFICATION COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    verify()
