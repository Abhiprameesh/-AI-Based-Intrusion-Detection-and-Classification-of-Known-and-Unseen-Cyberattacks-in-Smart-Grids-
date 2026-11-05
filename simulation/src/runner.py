"""
Milestone 1 Simulation Runner.
Executes the normal operating-point test and short time-series test.
Performs physical validation checks at each simulation step.
"""

import json
import os
import sys
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.circuit_builder import MicrogridCircuitBuilder


class Milestone1Runner:
    """
    Executes and validates Milestone 1 simulation tests.
    """

    def __init__(self, circuit_file: str = None):
        self.builder = MicrogridCircuitBuilder(circuit_file=circuit_file)

    def run_snapshot_test(self) -> Dict[str, Any]:
        """
        Executes a static operating-point test under nominal conditions.
        Validates:
          1. Solver convergence
          2. Bus voltage ranges (218.5 V to 241.5 V, +/- 5% of 230V)
          3. Power balance across the microgrid (< 5.0 W residual)
          4. Correct power sign conventions (PV negative, loads positive)
        """
        print("=" * 70)
        print("RUNNING OPERATING-POINT SNAPSHOT TEST")
        print("=" * 70)

        # 1. Compile and set baseline
        converged = self.builder.build_circuit()
        if not converged:
            raise RuntimeError("Initial circuit failed to solve.")

        converged = self.builder.set_operating_point(
            pv_kw=20.0,
            load_a_kw=7.0,
            load_b_kw=8.0,
            bss_state="IDLING",
            bss_kw=0.0
        )

        # 2. Extract measurements at all 5 meters
        measurements = {}
        for meter_name in self.builder.METER_LOCATIONS.keys():
            measurements[meter_name] = self.builder.get_meter_measurements(meter_name)

        soc = self.builder.get_battery_soc()

        # 3. Check physical laws
        is_balanced, residual, p_breakdown = self.builder.verify_power_balance()

        # 4. Check voltage bounds (230V +/- 5% = [218.5, 241.5])
        voltage_checks = {}
        for meter_name, m in measurements.items():
            v_ok = (218.5 <= m["uln1"] <= 241.5) and (218.5 <= m["uln2"] <= 241.5) and (218.5 <= m["uln3"] <= 241.5)
            voltage_checks[meter_name] = {
                "uln1_v": m["uln1"],
                "uln2_v": m["uln2"],
                "uln3_v": m["uln3"],
                "in_spec": v_ok
            }

        # 5. Check sign conventions
        pv_p = measurements["PV-Inverter-AC-Meter"]["p_sum3"]
        la_p = measurements["Load-A-AC-Meter"]["p_sum3"]
        lb_p = measurements["Load-B-AC-Meter"]["p_sum3"]
        signs_ok = (pv_p < 0.0) and (la_p > 0.0) and (lb_p > 0.0)

        results = {
            "test_name": "operating_point_snapshot",
            "solver_converged": converged,
            "power_balance": {
                "is_balanced": is_balanced,
                "residual_watts": residual,
                "breakdown_watts": p_breakdown,
            },
            "voltage_checks": voltage_checks,
            "sign_conventions_valid": signs_ok,
            "battery_soc_percent": soc,
            "pv_active_power_watts": pv_p,
            "load_a_active_power_watts": la_p,
            "load_b_active_power_watts": lb_p,
            "substation_active_power_watts": measurements["Substation-AC-Meter"]["p_sum3"],
        }

        print(f"Solver Converged: {converged}")
        print(f"Power Balance Residual: {residual:.3f} W (Pass: {is_balanced})")
        print(f"PV Active Power: {pv_p:.2f} W (Export/Generation is negative: {pv_p < 0})")
        print(f"Load A Power: {la_p:.2f} W | Load B Power: {lb_p:.2f} W (Load is positive: {la_p > 0})")
        print(f"Substation Power: {measurements['Substation-AC-Meter']['p_sum3']:.2f} W (Net export)")
        print(f"Battery SOC: {soc:.2f}%")
        print(f"All Bus Voltages in Spec (218.5V - 241.5V): {all(vc['in_spec'] for vc in voltage_checks.values())}")
        print("=" * 70 + "\n")
        return results

    def run_timeseries_test(self, num_steps: int = 10) -> Dict[str, Any]:
        """
        Executes a short 10-second Quasi-Static Time-Series test (dt = 1.0 s).
        Tests dynamic transitions:
          - Steps 0 to 3: Steady baseline (BSS IDLING, PV = 20 kW).
          - Steps 4 to 6: Battery charging at 5 kW.
          - Steps 7 to 9: Battery discharging at 5 kW.
        Validates convergence, power balance, and SOC rate of change at each second.
        """
        print("=" * 70)
        print(f"RUNNING TIME-SERIES TEST ({num_steps} STEPS, dt = 1.0 s)")
        print("=" * 70)

        # Setup initial circuit
        self.builder.build_circuit()
        # Daily mode enables time-series step solving
        import opendssdirect as dss
        dss.Command(f"Set Mode=Daily StepSize=1s Number=1")

        step_logs: List[Dict[str, Any]] = []
        all_converged = True
        all_balanced = True
        soc_history = []

        for step in range(num_steps):
            # Control actions across timeline
            if step < 4:
                # Baseline
                state_str = "IDLING"
                bss_kw = 0.0
                pv_kw = 20.0
            elif step < 7:
                # BSS Charging at 5 kW
                state_str = "CHARGING"
                bss_kw = 5.0
                pv_kw = 20.0
            else:
                # BSS Discharging at 5 kW
                state_str = "DISCHARGING"
                bss_kw = 5.0
                pv_kw = 20.0

            # Apply setpoints
            dss.Command(f"PVSystem.PV1.pmpp={pv_kw}")
            dss.Command("PVSystem.PV1.irradiance=1.0")
            if state_str == "CHARGING":
                pct = (bss_kw / 20.0) * 100.0
                dss.Command(f"Storage.BSS1.State=CHARGING %charge={pct}")
            elif state_str == "DISCHARGING":
                pct = (bss_kw / 20.0) * 100.0
                dss.Command(f"Storage.BSS1.State=DISCHARGING %discharge={pct}")
            else:
                dss.Command("Storage.BSS1.State=IDLING")

            # Solve 1 step
            dss.Command("Solve")
            step_conv = bool(dss.Solution.Converged())
            if not step_conv:
                all_converged = False

            # Extract metrics
            soc = self.builder.get_battery_soc()
            soc_history.append(soc)

            is_bal, resid, breakdown = self.builder.verify_power_balance()
            if not is_bal:
                all_balanced = False

            meas_sub = self.builder.get_meter_measurements("Substation-AC-Meter")
            meas_bss = self.builder.get_meter_measurements("BSS-Inverter-AC-Meter")
            meas_pv = self.builder.get_meter_measurements("PV-Inverter-AC-Meter")

            log_entry = {
                "step": step,
                "bss_state": state_str,
                "converged": step_conv,
                "soc_percent": soc,
                "residual_w": resid,
                "p_sub_w": meas_sub["p_sum3"],
                "p_pv_w": meas_pv["p_sum3"],
                "p_bss_w": meas_bss["p_sum3"],
                "substation_v_ln_v": meas_sub["uln1"],
            }
            step_logs.append(log_entry)

            print(f"Step {step:2d} | State: {state_str:11s} | Conv: {str(step_conv):5s} | "
                  f"SoC: {soc:.5f}% | Res: {resid:.3f}W | P_sub: {meas_sub['p_sum3']:8.1f}W | "
                  f"P_bss: {meas_bss['p_sum3']:8.1f}W | V_sub: {meas_sub['uln1']:.2f}V")

        # Verify SoC derivative
        # While charging (step 4 to 6), SoC must increase
        soc_charging_diff = soc_history[6] - soc_history[4]
        soc_charging_valid = soc_charging_diff > 0.0

        # While discharging (step 7 to 9), SoC must decrease
        soc_discharging_diff = soc_history[9] - soc_history[7]
        soc_discharging_valid = soc_discharging_diff < 0.0

        results = {
            "test_name": "short_timeseries_10s",
            "all_steps_converged": all_converged,
            "all_steps_power_balanced": all_balanced,
            "soc_charging_derivative_positive": soc_charging_valid,
            "soc_discharging_derivative_negative": soc_discharging_valid,
            "soc_history": soc_history,
            "step_logs": step_logs,
        }

        print("-" * 70)
        print(f"All Steps Converged: {all_converged}")
        print(f"All Steps Power Balanced: {all_balanced}")
        print(f"SoC Increased During Charging: {soc_charging_valid} (delta={soc_charging_diff:+.6f}%)")
        print(f"SoC Decreased During Discharging: {soc_discharging_valid} (delta={soc_discharging_diff:+.6f}%)")
        print("=" * 70 + "\n")
        return results

    def run_all(self) -> Dict[str, Any]:
        """
        Runs both snapshot and time-series tests.
        """
        snap = self.run_snapshot_test()
        ts = self.run_timeseries_test()
        return {
            "snapshot_test": snap,
            "timeseries_test": ts,
            "overall_pass": (
                snap["solver_converged"]
                and snap["power_balance"]["is_balanced"]
                and snap["sign_conventions_valid"]
                and ts["all_steps_converged"]
                and ts["all_steps_power_balanced"]
                and ts["soc_charging_derivative_positive"]
                and ts["soc_discharging_derivative_negative"]
            )
        }


if __name__ == "__main__":
    runner = Milestone1Runner()
    res = runner.run_all()
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "milestone1_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved test run results to: {out_path}")
    print(f"Milestone 1 Validation Passed: {res['overall_pass']}")
