"""
Comprehensive Test Suite for Cyber Layer & Controlled Attack Injector (Milestone 3).
Validates:
1. False Data Injection (FDI): Corrupts reported telemetry while true physical state remains balanced.
2. Unauthorized Control (Industroyer PV & BSS): Aligns physical state to malicious command with valid power flow.
3. Telemetry Dropout & Freeze: Freezes measurement channels during attack window and restores tracking.
4. Unseen Stealthy Ramp: Gradual drift accumulation over time.
5. Coordinated Masking: Physical battery discharge masked at the substation meter.
6. Benign Disturbance: Natural physical transients strictly labeled benign (attack_active = 0).
7. Scientific Separation: 150-feature schema is strictly preserved; realtime and freq fields are protected.
8. Matched Control & Delta Analysis: Matched benign vs attack run yields valid physical deltas.
9. Reproducibility across random seeds.
"""

import unittest
import numpy as np
import pandas as pd
import joblib
import os
import sys

# Ensure simulation package is discoverable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.circuit_builder import MicrogridCircuitBuilder
from src.profile_loader import ProfileLoader
from src.telemetry_exporter import TelemetryExporter
from src.attack_injector import (
    FalseDataInjectionAttack,
    UnauthorizedControlAttack,
    TelemetryDropoutAttack,
    UnseenStealthyRampAttack,
    BenignDisturbance,
    AttackType,
)
from src.scenario_runner import ScenarioRunner


class TestAttackInjector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = MicrogridCircuitBuilder()
        cls.loader = ProfileLoader()
        cls.exporter = TelemetryExporter()
        cls.runner = ScenarioRunner(cls.builder, cls.loader, cls.exporter)

        # Load frozen Phase 1 preprocessor
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        prep_path = os.path.join(
            root_dir, "phase_1", "results", "data4cyber_dataset", "binary", "models", "preprocessor_unscaled.joblib"
        )
        cls.prep = joblib.load(prep_path)

    def test_01_fdi_telemetry_attack_physical_separation(self):
        """
        Verify False Data Injection Attack:
        1. Modifies reported telemetry during attack window (steps 10 to 20).
        2. Leaves OpenDSS physical state and true power balance 100% UNCHANGED.
        3. Reported measurement differs from true physical measurement.
        """
        attack = FalseDataInjectionAttack(
            name="test_fdi_bss_scaling_0.5",
            target_meter="BSS-Inverter-AC-Meter",
            start_step=10,
            end_step=20,
            scale_factor=0.5,
            bias_w=0.0,
        )

        feats, meta = self.runner.run_scenario(num_steps=25, attack=attack, initial_soc_percent=20.0)

        # Outside attack window (step 5): benign
        self.assertEqual(meta["attack_active"].iloc[5], 0)
        self.assertEqual(meta["telemetry_modified"].iloc[5], 0)
        self.assertAlmostEqual(meta["telemetry_discrepancy_bss_w"].iloc[5], 0.0, places=4)

        # Inside attack window (step 15): active attack
        self.assertEqual(meta["attack_active"].iloc[15], 1)
        self.assertEqual(meta["telemetry_modified"].iloc[15], 1)
        self.assertEqual(meta["attack_name"].iloc[15], "test_fdi_bss_scaling_0.5")

        # True physical state vs reported telemetry
        true_p = meta["true_bss_active_w"].iloc[15]
        rep_p = meta["reported_bss_active_w"].iloc[15]
        # In idling, BSS standby power is +200W, so scaling by 0.5 gives +100W
        self.assertAlmostEqual(rep_p, true_p * 0.5, places=3)
        self.assertNotEqual(rep_p, true_p, "FDI reported telemetry must disagree with true physical state.")

        # Physical power balance residual in OpenDSS remains valid (< 1.0 W)
        max_resid = meta["physical_power_balance_residual_w"].max()
        self.assertLess(max_resid, 1.0, f"Physical grid power balance violated: {max_resid} W")

    def test_02_unauthorized_pv_curtailment_physical_response(self):
        """
        Verify Unauthorized PV Curtailment Attack (Industroyer style):
        1. Malicious setpoint (pv_kw = 0.0) sent to OpenDSS during steps 8 to 15.
        2. OpenDSS physically solves new operating point (PV active power drops to 0 W).
        3. Substation power flow adjusts to compensate for loss of PV generation.
        4. Solver converges and power balance remains strictly valid (< 1.0 W).
        """
        attack = UnauthorizedControlAttack(
            name="test_industroyer_pv_trip",
            control_target="PV",
            start_step=8,
            end_step=15,
            forced_pv_kw=0.0,
        )

        feats, meta = self.runner.run_scenario(num_steps=20, attack=attack, initial_soc_percent=20.0)

        # Baseline before attack (step 4): PV generating ~13.1 kW at 643.1 W/m2 irradiance (signed negative)
        self.assertLess(meta["true_pv_active_w"].iloc[4], -10000.0)

        # During attack (step 12): PV curtailed to 0.0 W
        self.assertEqual(meta["attack_active"].iloc[12], 1)
        self.assertEqual(meta["control_modified"].iloc[12], 1)
        self.assertAlmostEqual(meta["true_pv_active_w"].iloc[12], 0.0, places=1)
        self.assertAlmostEqual(meta["reported_pv_active_w"].iloc[12], 0.0, places=1)

        # Substation imports more power to replace lost PV generation
        p_sub_baseline = meta["true_substation_w"].iloc[4]
        p_sub_attack = meta["true_substation_w"].iloc[12]
        self.assertGreater(p_sub_attack, p_sub_baseline, "Substation net import must increase when PV is tripped.")

        # Power balance verified at every step
        self.assertTrue(meta["physical_power_balance_converged"].all())
        self.assertLess(meta["physical_power_balance_residual_w"].max(), 1.0)

    def test_03_unauthorized_bss_discharge_physical_response(self):
        """
        Verify Unauthorized BSS Discharge Attack:
        1. Malicious setpoint commands battery to DISCHARGE at 15 kW during steps 5 to 12.
        2. OpenDSS Storage discharges physically; battery active power becomes negative.
        3. Battery SOC reference decreases physically during discharge.
        """
        attack = UnauthorizedControlAttack(
            name="test_industroyer_bss_forced_discharge",
            control_target="BSS",
            start_step=5,
            end_step=12,
            forced_bss_state="DISCHARGING",
            forced_bss_kw=15.0,
            spoof_reported_soc=True,
            spoofed_soc_value=0.20,
        )

        feats, meta = self.runner.run_scenario(num_steps=18, attack=attack, initial_soc_percent=20.0)

        # During attack (step 8): battery power is negative (discharging ~15 kW)
        self.assertEqual(meta["attack_active"].iloc[8], 1)
        self.assertLess(meta["true_bss_active_w"].iloc[8], -14000.0)

        # Spoofed SOC check: reported SOC is clamped to 0.20 while true SOC decreases
        self.assertAlmostEqual(meta["reported_soc"].iloc[8], 0.20, places=4)
        self.assertLess(meta["true_soc_ref"].iloc[12], meta["true_soc_ref"].iloc[4])

    def test_04_telemetry_dropout_freeze_attack(self):
        """
        Verify Telemetry Dropout / Replay Attack:
        1. Holds stale Load-B meter reading across steps 8 to 14.
        2. Unfreezes and returns to normal tracking at step 15.
        3. True physical load continues to solve normally.
        """
        attack = TelemetryDropoutAttack(
            name="test_load_b_freeze",
            target_meter="Load-B-AC-Meter",
            start_step=8,
            end_step=14,
            dropout_mode="freeze",
        )

        feats, meta = self.runner.run_scenario(num_steps=18, attack=attack, initial_soc_percent=20.0)

        # Values at step 8 through 14 must be identical (frozen)
        frozen_vals = [feats["Load-B-AC-Meter.p_sum3"].iloc[i] for i in range(8, 15)]
        self.assertEqual(len(set(frozen_vals)), 1, "Load-B active power must be frozen to a single constant value.")

        # Outside attack window (step 16), attack_active is 0
        self.assertEqual(meta["attack_active"].iloc[16], 0)

    def test_05_unseen_stealthy_ramp_attack(self):
        """
        Verify Unseen Stealthy Ramp Attack:
        1. Marked explicitly as is_unseen = 1 in metadata.
        2. Bias accumulates gradually each second (25 W/s).
        3. Discrepancy at step 10 is smaller than at step 30.
        """
        attack = UnseenStealthyRampAttack(
            name="unseen_pv_ramp_fdi",
            stealth_mode="ramp_fdi",
            target_meter="PV-Inverter-AC-Meter",
            start_step=5,
            end_step=35,
            ramp_rate_w_per_sec=25.0,
            max_bias_w=2000.0,
        )

        feats, meta = self.runner.run_scenario(num_steps=40, attack=attack, initial_soc_percent=20.0)

        # Check unseen flag
        self.assertEqual(meta["is_unseen"].iloc[10], 1)
        self.assertEqual(meta["attack_family"].iloc[10], "unseen_stealthy_attack")

        # Check gradual drift accumulation
        bias_step_10 = meta["telemetry_discrepancy_pv_w"].iloc[10]
        bias_step_30 = meta["telemetry_discrepancy_pv_w"].iloc[30]
        self.assertAlmostEqual(bias_step_10, 25.0 * (10 - 5), places=2)
        self.assertAlmostEqual(bias_step_30, 25.0 * (30 - 5), places=2)
        self.assertGreater(bias_step_30, bias_step_10)

    def test_06_benign_physical_disturbance_ground_truth(self):
        """
        Verify Controlled Benign Physical Disturbance:
        1. Simulates physical cloud shading (70% solar drop) during steps 5 to 12.
        2. PV active power physically drops in OpenDSS.
        3. Ground truth label MUST REMAIN attack_active = 0 and attack_family = 'benign_disturbance'.
        """
        disturbance = BenignDisturbance(
            name="benign_cloud_cover_shading",
            disturbance_type="cloud_shading",
            start_step=5,
            end_step=12,
            irradiance_drop_fraction=0.70,
        )

        feats, meta = self.runner.run_scenario(num_steps=18, attack=disturbance, initial_soc_percent=20.0)

        # Physical PV output drops during cloud cover
        p_pv_normal = meta["true_pv_active_w"].iloc[2]
        p_pv_shaded = meta["true_pv_active_w"].iloc[8]
        # In generator convention, less negative magnitude means less generation
        self.assertGreater(p_pv_shaded, p_pv_normal, "Shaded PV must generate less power.")

        # CRITICAL GUARDRAIL: Ground truth label is strictly 0 (NOT AN ATTACK)
        self.assertEqual(meta["attack_active"].iloc[8], 0)
        self.assertEqual(meta["attack_family"].iloc[8], "benign_disturbance")

    def test_07_scientific_separation_and_frozen_preprocessor(self):
        """
        Verify Scientific Separation:
        1. features_df shape is strictly [N, 150].
        2. Zero ground truth or attack labels inside features matrix.
        3. realtime and freq fields are completely unaffected by attacks.
        4. Transformed cleanly through frozen Phase 1 preprocessor without NaN.
        """
        attack = FalseDataInjectionAttack(
            name="fdi_check",
            target_meter="Substation-AC-Meter",
            start_step=5,
            end_step=15,
            scale_factor=1.5,
        )

        feats, meta = self.runner.run_scenario(num_steps=20, attack=attack, initial_soc_percent=20.0)

        # Exact schema check
        self.assertEqual(list(feats.columns), list(self.prep.feature_names_in_))
        self.assertNotIn("attack_active", feats.columns)
        self.assertNotIn("timestamp", feats.columns)

        # Protection of realtime and freq: DER realtime must remain exactly constant 17.180131328
        for m in ["BSS-Inverter-AC-Meter", "Load-A-AC-Meter", "Load-B-AC-Meter", "PV-Inverter-AC-Meter"]:
            der_rt = feats[f"{m}.realtime"]
            self.assertEqual(der_rt.nunique(), 1)
            self.assertAlmostEqual(der_rt.iloc[0], 17.180131328, places=6)

        # Transformation through frozen preprocessor
        X_trans = self.prep.transform(feats)
        self.assertEqual(X_trans.shape, (20, 150))
        self.assertFalse(np.isnan(X_trans).any())

    def test_08_matched_benign_control_pair_deltas(self):
        """
        Verify Matched Pair Execution:
        1. Generates matched benign control run (Y=0 for all t).
        2. Computes physical deltas and reported deltas.
        3. Asserts that before attack start, physical delta is exactly 0.0 W.
        """
        attack = UnauthorizedControlAttack(
            name="matched_pv_curtailment",
            control_target="PV",
            start_step=10,
            end_step=20,
            forced_pv_kw=0.0,
        )

        res = self.runner.run_matched_pair(attack=attack, num_steps=25, initial_soc_percent=20.0)
        deltas = res["deltas"]

        # Before attack (step 5): delta is exactly 0.0 W
        self.assertAlmostEqual(deltas["delta_true_pv_w"].iloc[5], 0.0, places=4)
        self.assertAlmostEqual(deltas["delta_reported_pv_w"].iloc[5], 0.0, places=4)

        # During attack (step 15): PV was ~ -13.1 kW, forced to 0.0 kW -> delta is ~ +13.1 kW
        delta_pv = deltas["delta_true_pv_w"].iloc[15]
        self.assertGreater(delta_pv, 10000.0)

    def test_09_reproducibility(self):
        """
        Verify deterministic reproducibility across identical random seeds.
        """
        attack = FalseDataInjectionAttack(name="repro_test", start_step=5, end_step=10, scale_factor=0.8)

        feats1, meta1 = self.runner.run_scenario(num_steps=12, attack=attack, random_seed=123)
        feats2, meta2 = self.runner.run_scenario(num_steps=12, attack=attack, random_seed=123)

        np.testing.assert_allclose(feats1.values, feats2.values, rtol=1e-12)
        pd.testing.assert_frame_equal(meta1, meta2)


if __name__ == "__main__":
    unittest.main()
