"""
Unit and Integration Tests for Milestone 1 OpenDSS Minimal Circuit.
Uses standard Python unittest framework for fully reproducible test execution.
"""

import os
import sys
import unittest

# Ensure simulation package is in sys.path
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_DIR)

from src.circuit_builder import MicrogridCircuitBuilder
from src.runner import Milestone1Runner


class TestMinimalCircuit(unittest.TestCase):
    """
    Test suite verifying electrical and physical properties of the Milestone 1 OpenDSS circuit.
    """

    @classmethod
    def setUpClass(cls):
        cls.builder = MicrogridCircuitBuilder()
        cls.runner = Milestone1Runner()

    def test_01_circuit_compiles_and_solves(self):
        """Verify circuit compiles and power flow converges in OpenDSS."""
        converged = self.builder.build_circuit()
        self.assertTrue(converged, "OpenDSS failed to converge on initial circuit compilation.")

    def test_02_bus_voltages_within_standard_limits(self):
        """Verify phase-to-neutral voltages are within 230 V +/- 5% (EN 50160 / ANSI C84.1)."""
        self.builder.build_circuit()
        self.builder.set_operating_point(pv_kw=20.0, load_a_kw=7.0, load_b_kw=8.0, bss_state="IDLING")

        for meter_name, cfg in self.builder.METER_LOCATIONS.items():
            bus = cfg["bus"]
            v = self.builder.get_bus_voltages(bus)
            for phase_v in ["uln1", "uln2", "uln3"]:
                self.assertGreaterEqual(
                    v[phase_v], 218.5,
                    f"Bus {bus} {phase_v} voltage {v[phase_v]:.2f} V is below lower limit 218.5 V."
                )
                self.assertLessEqual(
                    v[phase_v], 241.5,
                    f"Bus {bus} {phase_v} voltage {v[phase_v]:.2f} V is above upper limit 241.5 V."
                )
            for ll_v in ["ull12", "ull23", "ull31"]:
                self.assertGreaterEqual(
                    v[ll_v], 380.0,
                    f"Bus {bus} {ll_v} line-to-line voltage {v[ll_v]:.2f} V is below 380.0 V."
                )
                self.assertLessEqual(
                    v[ll_v], 420.0,
                    f"Bus {bus} {ll_v} line-to-line voltage {v[ll_v]:.2f} V is above 420.0 V."
                )

    def test_03_power_sign_conventions_and_units(self):
        """Verify generation is negative, loads are positive, and powers are in Watts (W)."""
        self.builder.build_circuit()
        self.builder.set_operating_point(pv_kw=20.0, load_a_kw=7.0, load_b_kw=8.0, bss_state="IDLING")

        meas_pv = self.builder.get_meter_measurements("PV-Inverter-AC-Meter")
        meas_la = self.builder.get_meter_measurements("Load-A-AC-Meter")
        meas_lb = self.builder.get_meter_measurements("Load-B-AC-Meter")
        meas_sub = self.builder.get_meter_measurements("Substation-AC-Meter")

        # PV is generating 20 kW -> in Watts, P = -20000 W
        self.assertAlmostEqual(meas_pv["p_sum3"], -20000.0, delta=1.0, msg="PV power should be -20,000 W.")
        self.assertLess(meas_pv["p_sum3"], 0.0, "PV inverter active power must be negative (export/generation).")

        # Loads are consuming 7 kW and 8 kW -> in Watts, P = +7000 W and +8000 W
        self.assertAlmostEqual(meas_la["p_sum3"], 7000.0, delta=1.0, msg="Load A power should be +7,000 W.")
        self.assertAlmostEqual(meas_lb["p_sum3"], 8000.0, delta=1.0, msg="Load B power should be +8,000 W.")
        self.assertGreater(meas_la["p_sum3"], 0.0, "Load A power must be positive (consumption).")
        self.assertGreater(meas_lb["p_sum3"], 0.0, "Load B power must be positive (consumption).")

        # Net surplus generation (20 kW gen > 15.2 kW load) exported to substation -> negative sign in Data4Cyber
        self.assertLess(meas_sub["p_sum3"], 0.0, "Substation power must be negative when exporting surplus to grid.")

    def test_04_busbar_and_microgrid_power_balance(self):
        """Verify Kirchhoff's Power Law holds with justified numerical tolerance (< 1.0 W)."""
        self.builder.build_circuit()
        self.builder.set_operating_point(pv_kw=20.0, load_a_kw=7.0, load_b_kw=8.0, bss_state="IDLING")

        is_balanced, residual, breakdown = self.builder.verify_power_balance()
        self.assertTrue(is_balanced, f"Power balance failed: residual = {residual:.4f} W")
        self.assertLess(breakdown["busbar_residual_w"], 0.01, "Substation busbar KPL residual must be < 0.01 W.")
        self.assertLess(breakdown["conservation_residual_w"], 1.0, "Microgrid conservation residual must be < 1.0 W.")

    def test_05_neutral_current_physics(self):
        """Verify neutral vector sum current IN = |I1 + I2 + I3| is near 0 A under balanced loading."""
        self.builder.build_circuit()
        self.builder.set_operating_point(pv_kw=20.0, load_a_kw=7.0, load_b_kw=8.0, bss_state="IDLING")

        meas_pv = self.builder.get_meter_measurements("PV-Inverter-AC-Meter")
        # In balanced 3-phase, neutral vector sum current i_sum3 is near zero
        self.assertLess(meas_pv["i_sum3"], 0.05, f"Neutral current {meas_pv['i_sum3']} A exceeds 0.05 A under balanced conditions.")
        # While individual phase currents are non-zero (~28.9 A for 20 kW / 400V)
        self.assertGreater(meas_pv["iln1"], 25.0, "Phase current iln1 must be > 25 A for 20 kW PV injection.")

    def test_06_time_series_and_battery_soc_derivative(self):
        """Verify 1-second time-series convergence and battery SOC rate of change."""
        res = self.runner.run_timeseries_test(num_steps=10)
        self.assertTrue(res["all_steps_converged"], "At least one time-series step failed to converge.")
        self.assertTrue(res["all_steps_power_balanced"], "At least one time-series step failed power balance.")
        self.assertTrue(res["soc_charging_derivative_positive"], "Battery SOC did not increase during charging.")
        self.assertTrue(res["soc_discharging_derivative_negative"], "Battery SOC did not decrease during discharging.")


if __name__ == "__main__":
    unittest.main()
