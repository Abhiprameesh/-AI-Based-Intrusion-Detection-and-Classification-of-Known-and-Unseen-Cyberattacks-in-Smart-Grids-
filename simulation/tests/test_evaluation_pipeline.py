"""
Automated Test Suite for Milestone 4: IDS Telemetry Ingestion and Three-Part Evaluation.
Validates:
1. Feature contract validation (150 columns, exact order, no NaN, no Inf, fractional SOC).
2. Label isolation (zero ground-truth labels inside model feature matrix).
3. Frozen binary XGBoost inference and metrics calculation.
4. Frozen multiclass Random Forest inference and supported class mapping.
5. Autoencoder missing artifact blocker detection (no silent fabrication).
6. Matched benign-vs-attack delta calculations (zero delta during benign interval).
7. Variability sensitivity reproducibility under fixed seeds.
8. Phase 1 artifact immutability (SHA-256 integrity verification).
"""

import unittest
import os
import sys
import hashlib
import numpy as np
import pandas as pd
import joblib

# Ensure simulation package is discoverable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT_DIR, "simulation"))

from src.evaluation_pipeline import TelemetryContractValidator, IDSEvaluationPipeline
from src.attack_injector import FalseDataInjectionAttack, UnauthorizedControlAttack


class TestEvaluationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = IDSEvaluationPipeline()
        cls.validator = TelemetryContractValidator()

    def test_01_feature_contract_validation(self):
        """Verify TelemetryContractValidator passes valid data and catches contract violations."""
        # Valid test row
        feats, meta = self.pipeline.runner.run_scenario(num_steps=5, initial_soc_percent=20.0)
        is_valid, errors = self.validator.validate_feature_matrix(feats)
        self.assertTrue(is_valid, f"Valid features failed validation: {errors}")
        self.assertEqual(len(errors), 0)

        # Invalid: Missing column
        bad_feats_1 = feats.drop(columns=[feats.columns[0]])
        is_valid, errors = self.validator.validate_feature_matrix(bad_feats_1)
        self.assertFalse(is_valid)
        self.assertTrue(any("count mismatch" in e for e in errors))

        # Invalid: Null value injected
        bad_feats_2 = feats.copy()
        bad_feats_2.iloc[0, 0] = np.nan
        is_valid, errors = self.validator.validate_feature_matrix(bad_feats_2)
        self.assertFalse(is_valid)
        self.assertTrue(any("null/NaN" in e for e in errors))

        # Invalid: Non-fractional SOC (> 1.0)
        bad_feats_3 = feats.copy()
        bad_feats_3["Battery-SOC-Reference.soc_value"] = 75.0
        is_valid, errors = self.validator.validate_feature_matrix(bad_feats_3)
        self.assertFalse(is_valid)
        self.assertTrue(any("exceeds 1.0" in e for e in errors))

        # Invalid: Ground truth leakage
        bad_feats_4 = feats.copy()
        bad_feats_4["attack_active"] = 1
        is_valid, errors = self.validator.validate_feature_matrix(bad_feats_4)
        self.assertFalse(is_valid)
        self.assertTrue(any("leakage" in e.lower() for e in errors))

    def test_02_label_isolation_in_dataset_generation(self):
        """Verify that generate_evaluation_dataset strictly separates features from metadata."""
        feats, meta = self.pipeline.generate_evaluation_dataset(num_steps_per_scenario=10, initial_soc_percent=20.0)

        # Verify shapes: 9 scenarios * 10 steps = 90 rows
        self.assertEqual(feats.shape[0], 90)
        self.assertEqual(feats.shape[1], 150)
        self.assertEqual(meta.shape[0], 90)

        # Ensure no metadata columns leaked into features
        for forbidden in ["attack_active", "attack_family", "scenario_id", "timestamp", "is_unseen", "control_modified"]:
            self.assertNotIn(forbidden, feats.columns)

        # Ensure metadata contains ground-truth labels
        self.assertIn("attack_active", meta.columns)
        self.assertIn("attack_family", meta.columns)
        self.assertIn("scenario_id", meta.columns)

    def test_03_frozen_binary_xgboost_inference(self):
        """Verify that frozen Phase 1 XGBoost model predicts valid binary classes and probabilities."""
        feats, meta = self.pipeline.generate_evaluation_dataset(num_steps_per_scenario=10, initial_soc_percent=20.0)
        xgb = joblib.load(self.pipeline.PHASE1_DATA4CYBER_MODELS["binary_xgboost"])
        prep = joblib.load(self.pipeline.PHASE1_DATA4CYBER_MODELS["binary_preprocessor_unscaled"])

        X_trans = prep.transform(feats)
        preds = xgb.predict(X_trans)
        probs = xgb.predict_proba(X_trans)

        # Check binary labels and probability bounds
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))
        self.assertEqual(probs.shape, (90, 2))
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

    def test_04_frozen_multiclass_rf_inference(self):
        """Verify that frozen Phase 1 Random Forest multiclass predicts valid class labels."""
        feats, meta = self.pipeline.generate_evaluation_dataset(num_steps_per_scenario=10, initial_soc_percent=20.0)
        rf_multi = joblib.load(self.pipeline.PHASE1_DATA4CYBER_MODELS["multiclass_random_forest"])
        prep = joblib.load(self.pipeline.PHASE1_DATA4CYBER_MODELS["multiclass_preprocessor_unscaled"])

        X_trans = prep.transform(feats)
        preds = rf_multi.predict(X_trans)

        # Check multiclass labels {0, 1, 2, 3}
        self.assertTrue(set(np.unique(preds)).issubset({0, 1, 2, 3}))

    def test_05_autoencoder_missing_artifact_blocker(self):
        """
        Verify that Experiment 1 detects the absence of Data4Cyber Autoencoder.pt
        and marks Stage 3 as BLOCKED without fabricating weights or thresholds.
        Ensures test fixture passes save_predictions=False to prevent CSV overwrite.
        """
        feats, meta = self.pipeline.generate_evaluation_dataset(num_steps_per_scenario=5, initial_soc_percent=20.0)
        exp1_res = self.pipeline.run_experiment_1_frozen_models(feats, meta, save_predictions=False)

        ae_stage = exp1_res["stages"]["stage_3_unseen_autoencoder"]
        self.assertEqual(ae_stage["status"], "BLOCKED")
        self.assertIn("Autoencoder.pt", ae_stage["reason"])
        self.assertAlmostEqual(ae_stage["recorded_phase1_threshold"], 0.4276013, places=5)

    def test_06_matched_benign_delta_computation(self):
        """Verify Experiment 2 matched pair delta computation and pre-attack synchronization."""
        exp2_res = self.pipeline.run_experiment_2_matched_deltas(num_steps=20, initial_soc_percent=20.0)
        scenarios = exp2_res["scenarios"]

        self.assertIn("S3_industroyer_pv_trip", scenarios)
        pv_summary = scenarios["S3_industroyer_pv_trip"]

        # Pre-attack synchronization must have near-zero delta
        self.assertLess(pv_summary["pre_attack_synchronization_pv_delta_w"], 0.01)

    def test_07_variability_sensitivity_reproducibility(self):
        """Verify Experiment 3 sensitivity analysis execution and reproducibility."""
        feats, meta = self.pipeline.generate_evaluation_dataset(num_steps_per_scenario=10, initial_soc_percent=20.0)
        sens1 = self.pipeline.run_experiment_3_variability_sensitivity(feats, meta, random_seed=99)
        sens2 = self.pipeline.run_experiment_3_variability_sensitivity(feats, meta, random_seed=99)

        self.assertEqual(sens1["deterministic_baseline"]["f1_score"], sens2["deterministic_baseline"]["f1_score"])
        self.assertEqual(sens1["perturbation_1_sensor_noise"]["f1_score"], sens2["perturbation_1_sensor_noise"]["f1_score"])

    def test_08_phase1_artifact_immutability(self):
        """Verify that Phase 1 model and preprocessor hashes remain 100% unchanged."""
        expected_hashes = {
            "binary_xgboost": "f14e1c87d3640bcc8bec44d59925c2fd91b7c630a4cfe5e8b1a8d912e8a67e02",
            "binary_random_forest": "6a66af5e955bc0a51413b96da549f78bbeefbadffcb19476da90da9263b172a4",
            "binary_preprocessor_unscaled": "2c5ef4a896e75d237f4c59ca35cd3c3d6afd4b2abe23edf16bbc4e5f5d694b12",
            "binary_preprocessor_scaled": "d80f1484077092c1ba64c9ceec793041e8b9cf037b4f9cd101b6d2b236608ac5",
            "multiclass_xgboost": "2ec535dac31d9eafc31252fee3cc0335d8fd33bab47db527efa4e85a392459b9",
            "multiclass_random_forest": "8987a92549af94330e33266c2eada17ee7ece6a037c43ad3e8925475d8f47501",
            "unseen_metrics_json": "72a343107bca8dc3b916447832a7dd24398ae862cea0572012cb71c57bff30fa",
        }
        current_hashes = self.pipeline._audit_phase1_hashes()
        for k, v in expected_hashes.items():
            self.assertEqual(current_hashes[k], v, f"Phase 1 artifact {k} hash changed!")

    def test_09_test_fixture_does_not_overwrite_prediction_csv(self):
        """
        Regression Test for Fix 1:
        Verify that executing run_experiment_1_frozen_models with save_predictions=False
        leaves the production predictions CSV untouched.
        """
        pred_csv_path = os.path.join(self.pipeline.results_dir, "predictions_binary_and_multiclass.csv")
        initial_exists = os.path.exists(pred_csv_path)
        initial_mtime = os.path.getmtime(pred_csv_path) if initial_exists else None

        # Run small 3-step test fixture with save_predictions=False
        feats_small, meta_small = self.pipeline.generate_evaluation_dataset(num_steps_per_scenario=3, initial_soc_percent=20.0)
        self.pipeline.run_experiment_1_frozen_models(feats_small, meta_small, save_predictions=False)

        if initial_exists:
            post_mtime = os.path.getmtime(pred_csv_path)
            self.assertEqual(initial_mtime, post_mtime, "Production predictions CSV was modified despite save_predictions=False!")

    def test_10_recall_undefined_for_zero_attack_scenarios(self):
        """
        Regression Test for Fix 2:
        Verify that per-scenario attack recall is None (JSON null) for scenarios with
        zero attack-positive samples (e.g. S0, S8), while remaining numeric for attack scenarios.
        """
        feats, meta = self.pipeline.generate_evaluation_dataset(num_steps_per_scenario=50, initial_soc_percent=20.0)
        exp1_res = self.pipeline.run_experiment_1_frozen_models(feats, meta, save_predictions=False)

        per_sc = exp1_res["stages"]["stage_1_binary_xgboost"]["per_scenario"]

        # S0 has 0 attack samples -> recall must be None (not 1.0)
        self.assertEqual(per_sc["S0_benign_baseline"]["num_attack_samples"], 0)
        self.assertIsNone(per_sc["S0_benign_baseline"]["recall"])

        # S8 (benign cloud cover) has 0 attack samples -> recall must be None
        self.assertEqual(per_sc["S8_benign_cloud_cover_shading"]["num_attack_samples"], 0)
        self.assertIsNone(per_sc["S8_benign_cloud_cover_shading"]["recall"])

        # S1 has attack samples -> recall must be a float (numerically defined)
        self.assertGreater(per_sc["S1_fdi_bss_scaling_0.5"]["num_attack_samples"], 0)
        self.assertIsInstance(per_sc["S1_fdi_bss_scaling_0.5"]["recall"], float)


if __name__ == "__main__":
    unittest.main()
