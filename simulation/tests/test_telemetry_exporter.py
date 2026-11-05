"""
Milestone 2 Automated Validation Suite: Telemetry Exporter and Profile Integration.
Verifies schema ordering, units, missing values, power balance, BSS idle regression, and reproducibility.
"""

import os
import sys
import unittest
import joblib
import numpy as np
import pandas as pd

# Setup import path
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(PKG_DIR)
sys.path.insert(0, PKG_DIR)
sys.path.insert(0, ROOT_DIR)

from src.circuit_builder import MicrogridCircuitBuilder
from src.profile_loader import ProfileLoader
from src.telemetry_exporter import TelemetryExporter


class TestTelemetryExporter(unittest.TestCase):
    """
    Test suite verifying the 150-feature telemetry exporter and physical profile integration.
    """

    @classmethod
    def setUpClass(cls):
        cls.prep_path = os.path.join(
            ROOT_DIR, "phase_1", "results", "data4cyber_dataset", "binary", "models", "preprocessor_unscaled.joblib"
        )
        cls.prep = joblib.load(cls.prep_path)
        cls.expected_features = list(cls.prep.feature_names_in_)

        cls.builder = MicrogridCircuitBuilder()
        cls.loader = ProfileLoader()
        cls.exporter = TelemetryExporter(preprocessor_path=cls.prep_path)

    def test_01_feature_count_and_schema_ordering(self):
        """Verify exactly 150 features matching frozen preprocessor names and order."""
        self.builder.build_circuit()
        step_prof = self.loader.get_step_profile(0)
        row_df, meta = self.exporter.export_telemetry_vector(
            circuit_builder=self.builder,
            profile_signals=step_prof["raw_profiles"],
            step=0
        )

        self.assertEqual(row_df.shape, (1, 150), f"Exported row shape {row_df.shape} != (1, 150)")
        self.assertEqual(
            list(row_df.columns), self.expected_features,
            "Exported feature columns do not match expected preprocessor feature order exactly."
        )
        self.assertTrue(all(row_df.dtypes == np.float64), "All features must be float64.")

    def test_02_zero_missing_values_and_no_arbitrary_zeros(self):
        """Verify zero nulls and non-trivial genuine physical values."""
        self.builder.build_circuit()
        step_prof = self.loader.get_step_profile(0)
        row_df, meta = self.exporter.export_telemetry_vector(
            circuit_builder=self.builder,
            profile_signals=step_prof["raw_profiles"],
            step=0
        )

        # Zero nulls
        self.assertEqual(int(row_df.isnull().sum().sum()), 0, "Telemetry row contains NaN or null values.")

        # Check key signals are non-zero
        self.assertNotEqual(row_df["Substation-AC-Meter.uln1"].iloc[0], 0.0, "Voltage must be non-zero.")
        self.assertNotEqual(row_df["PV-Inverter-AC-Meter.p_sum3"].iloc[0], 0.0, "PV power must be non-zero.")
        self.assertNotEqual(row_df["Load-A-AC-Meter.p_sum3"].iloc[0], 0.0, "Load A power must be non-zero.")
        self.assertNotEqual(row_df["Battery-SOC-Reference.soc_value"].iloc[0], 0.0, "SOC must be non-zero.")
        self.assertNotEqual(row_df["Profile.total_irradiance"].iloc[0], 0.0, "Irradiance must be non-zero.")

    def test_03_physical_unit_and_range_validity(self):
        """Verify voltages, currents, powers, and power factors conform to physical limits."""
        self.builder.build_circuit()
        step_prof = self.loader.get_step_profile(0)
        row_df, meta = self.exporter.export_telemetry_vector(
            circuit_builder=self.builder,
            profile_signals=step_prof["raw_profiles"],
            step=0
        )

        # 1. Voltages in Volts (230V +/- 5% = [218.5, 241.5])
        for m in ["Substation-AC-Meter", "PV-Inverter-AC-Meter", "BSS-Inverter-AC-Meter", "Load-A-AC-Meter", "Load-B-AC-Meter"]:
            v_ln = row_df[f"{m}.uln1"].iloc[0]
            self.assertTrue(218.5 <= v_ln <= 241.5, f"{m} voltage {v_ln:.2f} V out of bounds.")
            v_ll = row_df[f"{m}.ull12"].iloc[0]
            self.assertTrue(380.0 <= v_ll <= 420.0, f"{m} line voltage {v_ll:.2f} V out of bounds.")

        # 2. Powers in Watts
        pv_p = row_df["PV-Inverter-AC-Meter.p_sum3"].iloc[0]
        self.assertLess(pv_p, -1000.0, f"PV active power {pv_p} W should be negative generation in Watts.")
        la_p = row_df["Load-A-AC-Meter.p_sum3"].iloc[0]
        self.assertGreater(la_p, 1000.0, f"Load A active power {la_p} W should be positive consumption in Watts.")

        # 3. Power factors
        for m in ["Substation-AC-Meter", "PV-Inverter-AC-Meter", "Load-A-AC-Meter"]:
            for p_idx in [1, 2, 3]:
                pf = row_df[f"{m}.cos_phi{p_idx}"].iloc[0]
                self.assertTrue(-1.0 <= pf <= 1.0, f"{m} pf {pf} not in [-1, 1].")

        # 4. Battery SOC (fractional per-unit [0.0, 1.0])
        soc = row_df["Battery-SOC-Reference.soc_value"].iloc[0]
        self.assertTrue(0.0 <= soc <= 1.0, f"Battery SOC {soc} not in fractional range [0, 1].")

    def test_04_neutral_current_and_apparent_power_derivation(self):
        """Verify Janitza UMG508 apparent power aggregation (arithmetic sum) and neutral current vector sum."""
        self.builder.build_circuit()
        step_prof = self.loader.get_step_profile(0)
        row_df, meta = self.exporter.export_telemetry_vector(
            circuit_builder=self.builder,
            profile_signals=step_prof["raw_profiles"],
            step=0
        )

        # In Janitza UMG508 reg 19034, s_sum3 = sln1 + sln2 + sln3
        s_sum = row_df["Load-A-AC-Meter.s_sum3"].iloc[0]
        s_parts = (
            row_df["Load-A-AC-Meter.sln1"].iloc[0]
            + row_df["Load-A-AC-Meter.sln2"].iloc[0]
            + row_df["Load-A-AC-Meter.sln3"].iloc[0]
        )
        self.assertAlmostEqual(s_sum, s_parts, places=3, msg="s_sum3 must equal sln1 + sln2 + sln3.")

        # Neutral current vector sum |I1 + I2 + I3| near 0 A in balanced 3-phase
        in_curr = row_df["Load-A-AC-Meter.i_sum3"].iloc[0]
        self.assertLess(in_curr, 0.05, f"Balanced neutral current {in_curr} A should be near 0.")

    def test_05_bess_idling_standby_loss_regression(self):
        """
        Regression Test for BSS IDLING Power:
        Confirms that +200 W is OpenDSS native %IdlingkW standby loss (1% of 20 kW = 200W).
        Verifies that when %IdlingkW=0, BSS power is exactly 0.0 W.
        """
        self.builder.build_circuit()

        # Case 1: Default OpenDSS %IdlingkW = 1.0% -> draws +200 W standby
        self.builder.set_operating_point(pv_kw=20.0, load_a_kw=7.0, load_b_kw=8.0, bss_state="IDLING", bss_idling_kw_pct=1.0)
        meas_idle_1 = self.builder.get_meter_measurements("BSS-Inverter-AC-Meter")
        self.assertAlmostEqual(
            meas_idle_1["p_sum3"], 200.0, places=1,
            msg="When %IdlingkW=1.0, BSS standby power must be +200.0 W."
        )

        # Case 2: Idealized zero-loss idle %IdlingkW = 0.0% -> draws 0.0 W
        self.builder.set_operating_point(pv_kw=20.0, load_a_kw=7.0, load_b_kw=8.0, bss_state="IDLING", bss_idling_kw_pct=0.0)
        meas_idle_0 = self.builder.get_meter_measurements("BSS-Inverter-AC-Meter")
        self.assertAlmostEqual(
            meas_idle_0["p_sum3"], 0.0, places=1,
            msg="When %IdlingkW=0.0, BSS standby power must be 0.0 W."
        )

    def test_06_profile_driven_stream_and_frozen_preprocessor(self):
        """Verify 10-step stream generation and transformation through frozen Phase 1 preprocessor."""
        features_df, metadata_df = self.exporter.export_stream(
            circuit_builder=self.builder,
            profile_loader=self.loader,
            num_steps=10,
            step_seconds=1.0,
            random_seed=42
        )

        self.assertEqual(features_df.shape, (10, 150), "Features stream shape must be (10, 150).")
        self.assertEqual(metadata_df.shape[0], 10, "Metadata stream must have 10 rows.")

        # Ensure timestamps and attack labels are NOT in feature matrix
        self.assertNotIn("timestamp", features_df.columns)
        self.assertNotIn("attack_active", features_df.columns)
        self.assertNotIn("attack_family", features_df.columns)

        # Test transformation via frozen Phase 1 preprocessor
        X_trans = self.prep.transform(features_df)
        self.assertEqual(X_trans.shape, (10, 150))
        self.assertFalse(np.isnan(X_trans).any(), "Transformed feature array contains NaN.")

    def test_07_reproducibility(self):
        """Verify deterministic outputs given same seed and profile step."""
        self.builder.build_circuit()
        step_prof = self.loader.get_step_profile(5)

        row1, meta1 = self.exporter.export_telemetry_vector(
            circuit_builder=self.builder,
            profile_signals=step_prof["raw_profiles"],
            step=5,
            random_seed=100
        )
        row2, meta2 = self.exporter.export_telemetry_vector(
            circuit_builder=self.builder,
            profile_signals=step_prof["raw_profiles"],
            step=5,
            random_seed=100
        )

        np.testing.assert_allclose(
            row1.values, row2.values, rtol=1e-12,
            err_msg="Exporter outputs are not deterministically reproducible."
        )

    def test_08_battery_soc_fractional_normalization_regression(self):
        """
        Regression Test for Battery SOC Semantics:
        Confirms that OpenDSS native percentage (%stored=75%) is converted to
        fractional per-unit (0.75) for both Reference and Reported SOC features,
        matching Data4Cyber schema and preventing extreme out-of-distribution scaling.
        """
        self.builder.build_circuit()
        step_prof = self.loader.get_step_profile(0)
        row_df, meta = self.exporter.export_telemetry_vector(
            circuit_builder=self.builder,
            profile_signals=step_prof["raw_profiles"],
            step=0
        )
        soc_ref = row_df["Battery-SOC-Reference.soc_value"].iloc[0]
        soc_rep = row_df["Battery-SOC-Reported.soc_value"].iloc[0]

        # Verify exact fractional normalization: 75% -> 0.75
        self.assertAlmostEqual(soc_ref, 0.75, places=4, msg="Reference SOC must be 0.75, not 75.0")
        self.assertAlmostEqual(soc_rep, 0.75, places=4, msg="Reported SOC must be 0.75, not 75.0")
        self.assertLessEqual(soc_ref, 1.0, "SOC must be fractional per-unit <= 1.0")
        self.assertGreaterEqual(soc_ref, 0.0, "SOC must be fractional per-unit >= 0.0")


if __name__ == "__main__":
    unittest.main()
