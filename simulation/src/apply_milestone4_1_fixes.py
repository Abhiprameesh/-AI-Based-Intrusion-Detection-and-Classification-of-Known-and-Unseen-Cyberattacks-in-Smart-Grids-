"""
Milestone 4.1 Minimal Fixes Application Script
Faithfully executes the authorized fixes:
1. Regenerates simulation/results/milestone4/predictions_binary_and_multiclass.csv
   from the original 1,350-sample telemetry_features.csv and telemetry_metadata.csv,
   without regenerating telemetry or altering feature data.
2. Updates metrics_experiment1_frozen_models.json so that zero-attack scenarios
   (S0 and S8) have recall set to None (JSON null).
3. Validates row counts, sample ordering, metric recalculation, and Phase 1 artifact immutability.
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import joblib

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT_DIR, "simulation"))

from src.evaluation_pipeline import IDSEvaluationPipeline


def apply_fixes():
    print("=" * 70)
    print("APPLYING MILESTONE 4.1 AUTHORIZED MINIMAL FIXES")
    print("=" * 70)

    m4_dir = os.path.join(ROOT_DIR, "simulation", "results", "milestone4")
    feats_path = os.path.join(m4_dir, "telemetry_features.csv")
    meta_path = os.path.join(m4_dir, "telemetry_metadata.csv")
    pred_path = os.path.join(m4_dir, "predictions_binary_and_multiclass.csv")
    metrics_path = os.path.join(m4_dir, "metrics_experiment1_frozen_models.json")

    # 1. Load saved 1,350-sample telemetry features and metadata
    feats_df = pd.read_csv(feats_path)
    meta_df = pd.read_csv(meta_path)
    assert len(feats_df) == 1350, f"Expected 1350 features rows, got {len(feats_df)}"
    assert len(meta_df) == 1350, f"Expected 1350 metadata rows, got {len(meta_df)}"
    print(f"[Step 1] Verified saved feature matrix: {feats_df.shape}")
    print(f"[Step 1] Verified saved metadata: {meta_df.shape}")

    # 2. Preserve before-fix metrics JSON for audit manifest
    with open(metrics_path, "r") as fp:
        metrics_before = json.load(fp)
    before_s0_rec = metrics_before["stages"]["stage_1_binary_xgboost"]["per_scenario"]["S0_benign_baseline"]["recall"]
    before_s8_rec = metrics_before["stages"]["stage_1_binary_xgboost"]["per_scenario"]["S8_benign_cloud_cover_shading"]["recall"]
    print(f"[Step 2] Before-fix S0 reported recall: {before_s0_rec}")
    print(f"[Step 2] Before-fix S8 reported recall: {before_s8_rec}")

    # 3. Instantiate pipeline and run experiment 1 with save_predictions=True
    pipeline = IDSEvaluationPipeline()
    pre_hashes = pipeline._audit_phase1_hashes()

    print("\n[Step 3] Executing run_experiment_1_frozen_models on saved 1,350 features...")
    exp1_res = pipeline.run_experiment_1_frozen_models(feats_df, meta_df, save_predictions=True)

    # 4. Save updated metrics_experiment1_frozen_models.json with null recall for zero-attack scenarios
    with open(metrics_path, "w") as fp:
        json.dump(exp1_res, fp, indent=2)
    print(f"[Step 4] Saved updated metrics to: {metrics_path}")

    # 5. Verify regenerated predictions CSV
    regenerated_pred_df = pd.read_csv(pred_path)
    assert len(regenerated_pred_df) == 1350, f"Regenerated CSV row count mismatch: {len(regenerated_pred_df)}"
    print(f"\n[Step 5] Regenerated predictions CSV verified: exactly {len(regenerated_pred_df)} data rows.")

    # Alignment verification: step and scenario_id must match row-for-row
    step_match = (regenerated_pred_df["step"] == meta_df["step"]).all()
    scenario_match = (regenerated_pred_df["scenario_id"] == meta_df["scenario_id"]).all()
    assert step_match, "Step alignment mismatch between predictions and metadata!"
    assert scenario_match, "Scenario ID alignment mismatch between predictions and metadata!"
    print(f"[Step 5] Sample alignment and row-for-row ordering verified: 100% match.")

    # 6. Verify recall fields
    s0_rec_after = exp1_res["stages"]["stage_1_binary_xgboost"]["per_scenario"]["S0_benign_baseline"]["recall"]
    s8_rec_after = exp1_res["stages"]["stage_1_binary_xgboost"]["per_scenario"]["S8_benign_cloud_cover_shading"]["recall"]
    s1_rec_after = exp1_res["stages"]["stage_1_binary_xgboost"]["per_scenario"]["S1_fdi_bss_scaling_0.5"]["recall"]
    assert s0_rec_after is None, f"Expected S0 recall to be None, got {s0_rec_after}"
    assert s8_rec_after is None, f"Expected S8 recall to be None, got {s8_rec_after}"
    assert isinstance(s1_rec_after, float), f"Expected S1 recall to be float, got {s1_rec_after}"
    print(f"[Step 6] After-fix S0 recall in JSON: {s0_rec_after} (null in JSON)")
    print(f"[Step 6] After-fix S8 recall in JSON: {s8_rec_after} (null in JSON)")
    print(f"[Step 6] After-fix S1 recall in JSON: {s1_rec_after} (numeric float)")

    # 7. Checksum regenerated predictions CSV
    with open(pred_path, "rb") as fp:
        pred_hash = hashlib.sha256(fp.read()).hexdigest()
    print(f"[Step 7] Regenerated predictions CSV SHA-256: {pred_hash}")

    # 8. Verify Phase 1 artifact immutability
    post_hashes = pipeline._audit_phase1_hashes()
    assert pre_hashes == post_hashes, "Phase 1 artifact hash mismatch detected!"
    print(f"[Step 8] Phase 1 artifacts strictly immutable: Verified True.")

    # 9. Record validation manifest
    manifest = {
        "milestone": "Milestone 4.1 Fixes",
        "status": "VALIDATED",
        "predictions_csv": {
            "path": "simulation/results/milestone4/predictions_binary_and_multiclass.csv",
            "row_count": len(regenerated_pred_df),
            "sha256": pred_hash,
            "alignment_verified": True,
        },
        "preserved_truncated_audit_csv": {
            "path": "simulation/results/milestone4/audit_truncated_predictions_45rows.csv",
            "row_count": 45,
            "sha256": "05f8e3c4bb245618171613e119878a03ab861046040ed9cb3fb57ccd76cabaf8",
        },
        "recall_fix_verification": {
            "S0_benign_baseline_recall_before": before_s0_rec,
            "S0_benign_baseline_recall_after": s0_rec_after,
            "S8_benign_cloud_cover_recall_before": before_s8_rec,
            "S8_benign_cloud_cover_recall_after": s8_rec_after,
            "S1_fdi_bss_scaling_recall": s1_rec_after,
        },
        "phase1_hashes": post_hashes,
    }

    manifest_path = os.path.join(m4_dir, "milestone4_1_fixes_manifest.json")
    with open(manifest_path, "w") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"\nSaved validation manifest to: {manifest_path}")

    print("=" * 70)
    print("MILESTONE 4.1 MINIMAL FIXES COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    apply_fixes()
