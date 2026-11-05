"""
Scenario Runner for Cyber-Physical Smart Grid Simulation.
Coordinates matched benign and attack simulations:
1. Executes time-series QSTS power flow driven by ProfileLoader.
2. Intercepts controls and telemetry via BaseAttack instances.
3. Separates true OpenDSS physical states from reported telemetry.
4. Generates matched benign controls and calculates physical and cyber deltas.
5. Strictly exports the 150-feature schema for model evaluation while isolating labels in metadata.
"""

import os
import sys
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

try:
    from simulation.src.circuit_builder import MicrogridCircuitBuilder
    from simulation.src.profile_loader import ProfileLoader
    from simulation.src.telemetry_exporter import TelemetryExporter
    from simulation.src.attack_injector import BaseAttack, AttackType
except ImportError:
    from src.circuit_builder import MicrogridCircuitBuilder
    from src.profile_loader import ProfileLoader
    from src.telemetry_exporter import TelemetryExporter
    from src.attack_injector import BaseAttack, AttackType


class ScenarioRunner:
    """
    Executes controlled cyber-physical simulation scenarios with strict ground-truth separation.
    """

    def __init__(
        self,
        circuit_builder: Optional[MicrogridCircuitBuilder] = None,
        profile_loader: Optional[ProfileLoader] = None,
        telemetry_exporter: Optional[TelemetryExporter] = None,
    ):
        self.builder = circuit_builder or MicrogridCircuitBuilder()
        self.loader = profile_loader or ProfileLoader()
        self.exporter = telemetry_exporter or TelemetryExporter()

    def run_scenario(
        self,
        num_steps: int = 300,
        attack: Optional[BaseAttack] = None,
        initial_soc_percent: float = 20.0,
        random_seed: int = 42,
        step_seconds: float = 1.0,
        slack_pu: Optional[float] = None,
        load_a_kvar: Optional[float] = None,
        load_b_kvar: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes a single scenario over num_steps.
        Returns:
          (features_df [N, 150], metadata_df [N, M])
        """
        # 1. Initialize circuit with specified initial SOC and optional candidate operating parameters
        self.builder.build_circuit(
            initial_soc_percent=initial_soc_percent,
            slack_pu=slack_pu,
            load_a_kvar=load_a_kvar,
            load_b_kvar=load_b_kvar,
        )

        import opendssdirect as dss
        dss.Command(f"Set Mode=Daily StepSize={step_seconds}s Number=1")

        feature_rows: List[pd.DataFrame] = []
        metadata_rows: List[Dict[str, Any]] = []
        telemetry_history: List[Dict[str, float]] = []

        for step in range(num_steps):
            # A. Load driving profile
            step_profile = self.loader.get_step_profile(step)
            raw_profiles = step_profile["raw_profiles"]
            mapped_setpoints = copy_dict = dict(step_profile["mapped_setpoints"])

            # Default BSS baseline state is IDLING with 0.0 kW
            mapped_setpoints.setdefault("bss_state", "IDLING")
            mapped_setpoints.setdefault("bss_kw", 0.0)

            # B. Check control-layer attack injection
            control_modified = False
            if attack is not None:
                mapped_setpoints, control_modified = attack.modify_control_setpoints(
                    mapped_setpoints, step
                )

            # C. Apply setpoints to OpenDSS and solve physical power flow
            pv_kw = mapped_setpoints["pv_pmpp_kw"]
            pv_irr = mapped_setpoints["pv_irradiance_pu"]
            dss.Command(f"PVSystem.PV1.pmpp={pv_kw}")
            dss.Command(f"PVSystem.PV1.irradiance={pv_irr}")

            load_a_kw = mapped_setpoints["load_a_kw"]
            load_b_kw = mapped_setpoints["load_b_kw"]
            dss.Command(f"Load.LoadA.kW={load_a_kw}")
            dss.Command(f"Load.LoadB.kW={load_b_kw}")
            if load_a_kvar is not None and load_a_kvar == 0.0:
                dss.Command("Load.LoadA.kvar=0.0")
            if load_b_kvar is not None and load_b_kvar == 0.0:
                dss.Command("Load.LoadB.kvar=0.0")

            bss_state = mapped_setpoints.get("bss_state", "IDLING").upper()
            bss_kw = mapped_setpoints.get("bss_kw", 0.0)
            dss.Command(f"Storage.BSS1.State={bss_state}")
            if bss_state == "CHARGING":
                pct = (bss_kw / 20.0) * 100.0
                dss.Command(f"Storage.BSS1.%charge={pct}")
            elif bss_state == "DISCHARGING":
                pct = (bss_kw / 20.0) * 100.0
                dss.Command(f"Storage.BSS1.%discharge={pct}")

            # Solve power flow step
            dss.Command("Solve")
            converged = bool(dss.Solution.Converged())
            if not converged:
                raise RuntimeError(f"OpenDSS power flow diverged at simulation step {step}")

            # D. Extract TRUE physical measurements directly from OpenDSS
            true_meter_data = {}
            for meter_name in self.builder.METER_LOCATIONS.keys():
                true_meter_data[meter_name] = self.builder.get_meter_measurements(meter_name)

            # True physical battery SOC (converted to per-unit fraction [0.0, 1.0])
            true_soc_ref = self.builder.get_battery_soc() / 100.0
            true_soc_rep = true_soc_ref

            # True physical power balance verification
            is_balanced, residual_w, breakdown = self.builder.verify_power_balance()

            # E. Deterministic synthetic baseline frequency & realtime ticker
            np.random.seed(random_seed + step)
            freq_noise = float(np.random.normal(0.0, 0.012))
            base_freq = 50.00 + freq_noise

            der_realtime = 17.180131328
            sub_realtime = 886358371.4598651 + float(2 * (step // 2))

            # F. Assemble baseline raw telemetry dictionary
            raw_telemetry: Dict[str, float] = {}
            for meter_name, m_dict in true_meter_data.items():
                for sig, val in m_dict.items():
                    raw_telemetry[f"{meter_name}.{sig}"] = val
                raw_telemetry[f"{meter_name}.freq"] = base_freq
                if meter_name == "Substation-AC-Meter":
                    raw_telemetry[f"{meter_name}.realtime"] = sub_realtime
                else:
                    raw_telemetry[f"{meter_name}.realtime"] = der_realtime

            raw_telemetry["Battery-SOC-Reference.soc_value"] = true_soc_ref
            raw_telemetry["Battery-SOC-Reported.soc_value"] = true_soc_rep

            for prof_col in [
                "Profile.ambient_temperature",
                "Profile.load1_kw",
                "Profile.load2_kw",
                "Profile.pv_active_kw",
                "Profile.pv_input_kw",
                "Profile.pv_reactive_kvar",
                "Profile.total_irradiance",
                "MQTT-Price-Signal.price_eur_per_kwh",
            ]:
                raw_telemetry[prof_col] = float(raw_profiles[prof_col])

            # G. Check telemetry-layer attack injection
            reported_telemetry = dict(raw_telemetry)
            telemetry_modified = False
            if attack is not None:
                reported_telemetry, telemetry_modified = attack.modify_reported_telemetry(
                    reported_telemetry, step, history=telemetry_history
                )

            telemetry_history.append(reported_telemetry)

            # H. Construct 150-feature row strictly conforming to Phase 1 contract
            row_df = pd.DataFrame(
                [[reported_telemetry[f] for f in self.exporter.expected_features]],
                columns=self.exporter.expected_features,
                dtype=np.float64,
            )
            feature_rows.append(row_df)

            # I. Formulate Ground-Truth Metadata (strictly isolated from model inputs)
            attack_active = 0
            attack_family = "benign"
            attack_name = "benign_baseline"
            is_unseen = 0
            target_el = "none"
            target_sigs = "none"

            if attack is not None and attack.is_active(step):
                # Benign disturbances have attack_active = 0
                if attack.attack_type == AttackType.BENIGN_DISTURBANCE:
                    attack_active = 0
                    attack_family = "benign_disturbance"
                    attack_name = attack.name
                else:
                    attack_active = 1
                    attack_family = attack.attack_family
                    attack_name = attack.name
                    is_unseen = 1 if attack.is_unseen else 0
                    target_el = attack.target_element
                    target_sigs = ",".join(attack.target_signals)

            meta_entry = {
                "step": step,
                "timestamp": raw_profiles.get("timestamp", f"2026-03-05T13:19:{step:02d}.000Z"),
                "attack_active": attack_active,
                "attack_family": attack_family,
                "attack_name": attack_name,
                "is_unseen": is_unseen,
                "target_element": target_el,
                "target_signals": target_sigs,
                "control_modified": 1 if control_modified else 0,
                "telemetry_modified": 1 if telemetry_modified else 0,
                "physical_power_balance_converged": converged,
                "physical_power_balance_residual_w": residual_w,
                # True physical values in OpenDSS
                "true_pv_active_w": true_meter_data["PV-Inverter-AC-Meter"]["p_sum3"],
                "true_bss_active_w": true_meter_data["BSS-Inverter-AC-Meter"]["p_sum3"],
                "true_load_a_w": true_meter_data["Load-A-AC-Meter"]["p_sum3"],
                "true_load_b_w": true_meter_data["Load-B-AC-Meter"]["p_sum3"],
                "true_substation_w": true_meter_data["Substation-AC-Meter"]["p_sum3"],
                "true_substation_v_ln_v": true_meter_data["Substation-AC-Meter"]["uln1"],
                "true_soc_ref": true_soc_ref,
                # Reported telemetry (potentially manipulated by cyber layer)
                "reported_pv_active_w": reported_telemetry["PV-Inverter-AC-Meter.p_sum3"],
                "reported_bss_active_w": reported_telemetry["BSS-Inverter-AC-Meter.p_sum3"],
                "reported_load_a_w": reported_telemetry["Load-A-AC-Meter.p_sum3"],
                "reported_load_b_w": reported_telemetry["Load-B-AC-Meter.p_sum3"],
                "reported_substation_w": reported_telemetry["Substation-AC-Meter.p_sum3"],
                "reported_substation_v_ln_v": reported_telemetry["Substation-AC-Meter.uln1"],
                "reported_soc": reported_telemetry["Battery-SOC-Reported.soc_value"],
                # Discrepancy (cyber manipulation magnitude)
                "telemetry_discrepancy_pv_w": (
                    reported_telemetry["PV-Inverter-AC-Meter.p_sum3"]
                    - true_meter_data["PV-Inverter-AC-Meter"]["p_sum3"]
                ),
                "telemetry_discrepancy_bss_w": (
                    reported_telemetry["BSS-Inverter-AC-Meter.p_sum3"]
                    - true_meter_data["BSS-Inverter-AC-Meter"]["p_sum3"]
                ),
            }
            metadata_rows.append(meta_entry)

        features_df = pd.concat(feature_rows, ignore_index=True)
        metadata_df = pd.DataFrame(metadata_rows)
        return features_df, metadata_df

    def run_matched_pair(
        self,
        attack: BaseAttack,
        num_steps: int = 300,
        initial_soc_percent: float = 20.0,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Executes a matched pair:
        1. Benign Control Run (attack = None)
        2. Attack Run (attack = attack)
        Under identical initial conditions, driving profiles, and random seed.
        Computes physical and reported deltas.
        """
        # Run 1: Benign Control
        benign_feats, benign_meta = self.run_scenario(
            num_steps=num_steps,
            attack=None,
            initial_soc_percent=initial_soc_percent,
            random_seed=random_seed,
        )

        # Run 2: Attack Run
        attack_feats, attack_meta = self.run_scenario(
            num_steps=num_steps,
            attack=attack,
            initial_soc_percent=initial_soc_percent,
            random_seed=random_seed,
        )

        # Calculate Physical Deltas: True Physical(Attack) - True Physical(Benign)
        delta_true_pv = attack_meta["true_pv_active_w"] - benign_meta["true_pv_active_w"]
        delta_true_bss = attack_meta["true_bss_active_w"] - benign_meta["true_bss_active_w"]
        delta_true_sub = attack_meta["true_substation_w"] - benign_meta["true_substation_w"]
        delta_true_soc = attack_meta["true_soc_ref"] - benign_meta["true_soc_ref"]

        # Calculate Reported Deltas: Reported(Attack) - Reported(Benign)
        delta_rep_pv = attack_meta["reported_pv_active_w"] - benign_meta["reported_pv_active_w"]
        delta_rep_bss = attack_meta["reported_bss_active_w"] - benign_meta["reported_bss_active_w"]
        delta_rep_sub = attack_meta["reported_substation_w"] - benign_meta["reported_substation_w"]
        delta_rep_soc = attack_meta["reported_soc"] - benign_meta["reported_soc"]

        deltas_df = pd.DataFrame({
            "step": attack_meta["step"],
            "attack_active": attack_meta["attack_active"],
            "delta_true_pv_w": delta_true_pv,
            "delta_true_bss_w": delta_true_bss,
            "delta_true_substation_w": delta_true_sub,
            "delta_true_soc": delta_true_soc,
            "delta_reported_pv_w": delta_rep_pv,
            "delta_reported_bss_w": delta_rep_bss,
            "delta_reported_substation_w": delta_rep_sub,
            "delta_reported_soc": delta_rep_soc,
            "fdi_discrepancy_pv_w": attack_meta["telemetry_discrepancy_pv_w"],
            "fdi_discrepancy_bss_w": attack_meta["telemetry_discrepancy_bss_w"],
        })

        return {
            "attack_name": attack.name,
            "attack_family": attack.attack_family,
            "benign_features": benign_feats,
            "benign_metadata": benign_meta,
            "attack_features": attack_feats,
            "attack_metadata": attack_meta,
            "deltas": deltas_df,
        }
