"""
Telemetry Exporter and Feature Alignment Engine for Smart Grid IDS.
Maps OpenDSS physical circuit outputs and external profiles into the exact 150-feature
tabular telemetry schema expected by Phase 1 frozen machine learning models.

Strict Scientific Constraints:
1. Feature names, count, order, and datatypes match phase_1/results/data4cyber_dataset/binary/models/preprocessor_unscaled.joblib.
2. Timestamps and ground-truth metadata are strictly isolated from the 150-feature input matrix.
3. Every single feature has an explicit provenance classification (A, B, C, D) and physical derivation.
4. Arbitrary zero-filling and unverified substitutions are strictly rejected.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

try:
    from simulation.src.circuit_builder import MicrogridCircuitBuilder
    from simulation.src.profile_loader import ProfileLoader
except ImportError:
    from src.circuit_builder import MicrogridCircuitBuilder
    from src.profile_loader import ProfileLoader


class TelemetryExporter:
    """
    Exports synchronized 150-feature telemetry vectors from OpenDSS simulation
    strictly conforming to the Phase 1 Data4Cyber model contract.
    """

    def __init__(self, preprocessor_path: Optional[str] = None):
        if preprocessor_path is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            preprocessor_path = os.path.join(
                root_dir, "phase_1", "results", "data4cyber_dataset", "binary", "models", "preprocessor_unscaled.joblib"
            )
        self.preprocessor_path = preprocessor_path
        self._expected_features: List[str] = []
        self._provenance_registry: Dict[str, Dict[str, str]] = {}
        self._load_expected_schema()
        self._build_provenance_registry()

    def _load_expected_schema(self):
        """Loads and verifies the frozen preprocessor's exact feature names."""
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Frozen preprocessor not found at {self.preprocessor_path}")
        prep = joblib.load(self.preprocessor_path)
        if not hasattr(prep, "feature_names_in_"):
            raise AttributeError("Loaded preprocessor does not possess feature_names_in_ attribute.")
        self._expected_features = list(prep.feature_names_in_)
        if len(self._expected_features) != 150:
            raise ValueError(f"Expected 150 features, but preprocessor has {len(self._expected_features)}")

    @property
    def expected_features(self) -> List[str]:
        """Returns the list of 150 expected feature names in exact order."""
        return list(self._expected_features)

    def _build_provenance_registry(self):
        """
        Builds the formal provenance and transformation registry for all 150 features:
        A: Directly measured/solved by OpenDSS nodal power flow.
        B: Mathematically derived from genuine OpenDSS complex phasor outputs.
        C: Supplied by explicit synchronized external profile.
        D: Declared synthetic baseline / hardware register.
        """
        registry = {}
        for feat in self._expected_features:
            prefix, signal = feat.split(".", 1)

            # 1. External Profile Features (Category C)
            if prefix == "Profile":
                registry[feat] = {
                    "feature_name": feat,
                    "category": "C",
                    "provenance_description": "Synchronized external environmental/grid profile",
                    "source_element": "Data4Cyber S0 baseline profile",
                    "units": "kW" if "kw" in signal else ("kvar" if "kvar" in signal else ("deg C" if "temp" in signal else "W/m^2")),
                    "transformation": "Direct extraction from profile dataset at current time index",
                }
            elif prefix == "MQTT-Price-Signal":
                registry[feat] = {
                    "feature_name": feat,
                    "category": "C",
                    "provenance_description": "Dynamic electricity market price publication",
                    "source_element": "Data4Cyber S0 dynamic price schedule",
                    "units": "EUR/kWh",
                    "transformation": "Direct extraction from price time-series",
                }

            # 2. Battery State of Charge (Category A / B)
            elif feat == "Battery-SOC-Reference.soc_value":
                registry[feat] = {
                    "feature_name": feat,
                    "category": "A",
                    "provenance_description": "Direct OpenDSS physical battery charge tracking",
                    "source_element": "Storage.BSS1.%stored",
                    "units": "Fractional pu ([0.0, 1.0])",
                    "transformation": "OpenDSS %stored divided by 100.0 (converted to per-unit fraction)",
                }
            elif feat == "Battery-SOC-Reported.soc_value":
                registry[feat] = {
                    "feature_name": feat,
                    "category": "B",
                    "provenance_description": "Reported battery telemetry transmitted to DMS/controller",
                    "source_element": "Storage.BSS1 telemetry channel",
                    "units": "Fractional pu ([0.0, 1.0])",
                    "transformation": "Equals reference SOC in benign mode (fractional pu); manipulated in cyber-layer attacks",
                }

            # 3. AC Electrical Meters (140 features across 5 meters)
            elif prefix in ["BSS-Inverter-AC-Meter", "Load-A-AC-Meter", "Load-B-AC-Meter", "PV-Inverter-AC-Meter", "Substation-AC-Meter"]:
                # Line-Neutral Voltages (Category A)
                if signal in ["uln1", "uln2", "uln3"]:
                    registry[feat] = {
                        "feature_name": feat,
                        "category": "A",
                        "provenance_description": "Direct nodal RMS voltage magnitude",
                        "source_element": f"{prefix} bus nodal voltage",
                        "units": "V (RMS)",
                        "transformation": "Direct bus Line-to-Neutral magnitude from nodal admittance solution",
                    }
                # Line-Line Voltages (Category B)
                elif signal in ["ull12", "ull23", "ull31"]:
                    registry[feat] = {
                        "feature_name": feat,
                        "category": "B",
                        "provenance_description": "Line-to-Line RMS voltage magnitude difference",
                        "source_element": f"{prefix} bus nodal voltage vector difference",
                        "units": "V (RMS)",
                        "transformation": "|V_phase1 - V_phase2|",
                    }
                # Phase RMS Currents (Category A)
                elif signal in ["iln1", "iln2", "iln3"]:
                    registry[feat] = {
                        "feature_name": feat,
                        "category": "A",
                        "provenance_description": "Direct branch terminal phase RMS current",
                        "source_element": f"{prefix} element terminal currents",
                        "units": "A (RMS)",
                        "transformation": "Direct conductor RMS magnitude from OpenDSS CktElement.Currents",
                    }
                # Neutral Vector Sum Current (Category B)
                elif signal == "i_sum3":
                    registry[feat] = {
                        "feature_name": feat,
                        "category": "B",
                        "provenance_description": "Neutral conductor current vector sum (Janitza UMG508 reg 19018)",
                        "source_element": f"{prefix} complex phase currents",
                        "units": "A (Vector)",
                        "transformation": "|I1_complex + I2_complex + I3_complex|",
                    }
                # Active Powers (Category B)
                elif signal in ["pln1", "pln2", "pln3"]:
                    registry[feat] = {
                        "feature_name": feat,
                        "category": "B",
                        "provenance_description": "Phase real active power",
                        "source_element": f"{prefix} element terminal powers",
                        "units": "W (Watts)",
                        "transformation": "P_kW * 1000.0 (converted to Watts, sign adjusted for Substation export)",
                    }
                elif signal == "p_sum3":
                    registry[feat] = {
                        "feature_name": feat,
                        "category": "B",
                        "provenance_description": "Three-phase total real active power (pln1 + pln2 + pln3)",
                        "source_element": f"{prefix} phase powers",
                        "units": "W (Watts)",
                        "transformation": "Sum of per-phase active powers pln1 + pln2 + pln3",
                    }
                # Reactive Powers (Category B)
                elif signal in ["qln1", "qln2", "qln3"]:
                    registry[feat] = {
                        "feature_name": feat,
                        "category": "B",
                        "provenance_description": "Phase fundamental reactive power",
                        "source_element": f"{prefix} element terminal powers",
                        "units": "var",
                        "transformation": "Q_kvar * 1000.0 (converted to var, sign adjusted for Substation export)",
                    }
                elif signal == "q_sum3":
                    registry[feat] = {
                        "feature_name": feat,
                        "category": "B",
                        "provenance_description": "Three-phase total reactive power (qln1 + qln2 + qln3)",
                        "source_element": f"{prefix} phase reactive powers",
                        "units": "var",
                        "transformation": "Sum of per-phase reactive powers qln1 + qln2 + qln3",
                    }
                # Apparent Powers (Category B)
                elif signal in ["sln1", "sln2", "sln3"]:
                    registry[feat] = {
                        "feature_name": feat,
                        "category": "B",
                        "provenance_description": "Phase apparent power",
                        "source_element": f"{prefix} P and Q",
                        "units": "VA",
                        "transformation": "sqrt(P_phase^2 + Q_phase^2)",
                    }
                elif signal == "s_sum3":
                    registry[feat] = {
                        "feature_name": feat,
                        "category": "B",
                        "provenance_description": "Three-phase arithmetic apparent power sum (Janitza UMG508 reg 19034)",
                        "source_element": f"{prefix} phase apparent powers",
                        "units": "VA",
                        "transformation": "Arithmetic sum: sln1 + sln2 + sln3",
                    }
                # Power Factors (Category B)
                elif signal in ["cos_phi1", "cos_phi2", "cos_phi3"]:
                    phase_idx = signal[-1]
                    registry[feat] = {
                        "feature_name": feat,
                        "category": "B",
                        "provenance_description": f"Phase {phase_idx} power factor with sign convention",
                        "source_element": f"{prefix} phase powers",
                        "units": "Dimensionless ([-1.0, 1.0])",
                        "transformation": f"pln{phase_idx} / sln{phase_idx}",
                    }
                # Phase Sequence (Category B)
                elif signal == "phase_seq":
                    registry[feat] = {
                        "feature_name": feat,
                        "category": "B",
                        "provenance_description": "Phase rotation indicator (+1.0 for positive A->B->C)",
                        "source_element": f"{prefix} bus rotation field",
                        "units": "Dimensionless (+1.0)",
                        "transformation": "+1.0 for verified positive phase sequence",
                    }
                # Grid Frequency (Category D)
                elif signal == "freq":
                    registry[feat] = {
                        "feature_name": feat,
                        "category": "D",
                        "provenance_description": "Declared synthetic baseline frequency (OpenDSS steady-state solver)",
                        "source_element": "Synthetic nominal grid baseline",
                        "units": "Hz",
                        "transformation": "50.00 + Gaussian noise (sigma=0.012 Hz) to prevent zero-variance error",
                    }
                # Meter Hardware Clock (Category D)
                elif signal == "realtime":
                    registry[feat] = {
                        "feature_name": feat,
                        "category": "D",
                        "provenance_description": "Janitza UMG508 hardware clock register (reg 0)",
                        "source_element": "Hardware meter clock register",
                        "units": "seconds / hardware tick",
                        "transformation": "Authentic Data4Cyber baseline register values (constant 17.18s for DERs, 2s ticks for substation)",
                    }

        self._provenance_registry = registry

    def get_provenance_registry(self) -> pd.DataFrame:
        """Returns the complete 150-feature provenance audit as a DataFrame."""
        return pd.DataFrame(list(self._provenance_registry.values()))

    def export_telemetry_vector(
        self,
        circuit_builder: MicrogridCircuitBuilder,
        profile_signals: Dict[str, float],
        step: int = 0,
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extracts all measurements from the solved circuit builder, combines with profile signals,
        and constructs a single-row DataFrame matching the exact 150-feature schema.
        Returns:
          (features_df [1, 150], metadata_dict)
        """
        # 1. Query all 5 electrical meters
        meter_data = {}
        for meter_name in circuit_builder.METER_LOCATIONS.keys():
            meter_data[meter_name] = circuit_builder.get_meter_measurements(meter_name)

        # 2. Battery SOC (converted to per-unit fraction [0.0, 1.0] matching Data4Cyber schema)
        battery_soc_ref = circuit_builder.get_battery_soc() / 100.0
        # In benign baseline, reported equals reference
        battery_soc_rep = battery_soc_ref

        # 3. Deterministic synthetic baseline frequency & realtime ticker
        np.random.seed(random_seed + step)
        freq_noise = float(np.random.normal(0.0, 0.012))
        base_freq = 50.00 + freq_noise

        # Realtime hardware registers matching Data4Cyber S0
        # In S0: DER meters have constant 17.180131328; Substation has 886358371.4598651 + 2*(step//2)
        der_realtime = 17.180131328
        sub_realtime = 886358371.4598651 + float(2 * (step // 2))

        # 4. Construct complete 150-feature dictionary
        feature_dict = {}

        # 5 AC meters
        for meter_name, m_dict in meter_data.items():
            for signal_name, val in m_dict.items():
                col_name = f"{meter_name}.{signal_name}"
                feature_dict[col_name] = val
            # Add synthetic baseline freq and realtime
            feature_dict[f"{meter_name}.freq"] = base_freq
            if meter_name == "Substation-AC-Meter":
                feature_dict[f"{meter_name}.realtime"] = sub_realtime
            else:
                feature_dict[f"{meter_name}.realtime"] = der_realtime

        # Battery SOC features
        feature_dict["Battery-SOC-Reference.soc_value"] = battery_soc_ref
        feature_dict["Battery-SOC-Reported.soc_value"] = battery_soc_rep

        # External Profile features
        for prof_col in [
            "Profile.ambient_temperature",
            "Profile.load1_kw",
            "Profile.load2_kw",
            "Profile.pv_active_kw",
            "Profile.pv_input_kw",
            "Profile.pv_reactive_kvar",
            "Profile.total_irradiance",
            "MQTT-Price-Signal.price_eur_per_kwh"
        ]:
            if prof_col in profile_signals:
                feature_dict[prof_col] = float(profile_signals[prof_col])
            else:
                raise KeyError(f"Missing required driving profile signal: {prof_col}")

        # 5. Build ordered DataFrame matching exact preprocessor schema
        missing_in_data = [f for f in self._expected_features if f not in feature_dict]
        if missing_in_data:
            raise KeyError(f"Failed to generate all expected features. Missing: {missing_in_data}")

        # Construct row strictly in expected feature order
        ordered_values = [feature_dict[f] for f in self._expected_features]
        features_df = pd.DataFrame([ordered_values], columns=self._expected_features, dtype=np.float64)

        # 6. Verify zero NaNs and zero missing columns
        if features_df.isnull().sum().sum() > 0:
            raise ValueError("Exported telemetry row contains NaN or null values.")

        # 7. Metadata context (isolated from model input)
        metadata = {
            "step": step,
            "timestamp": profile_signals.get("timestamp", f"2026-03-05T13:19:{step:02d}.000Z"),
            "attack_active": 0,
            "attack_family": "benign",
            "attack_phase": "nominal",
            "provenance": "opendss_sim",
            "power_balance_residual_w": circuit_builder.verify_power_balance()[1],
        }

        return features_df, metadata

    def export_stream(
        self,
        circuit_builder: MicrogridCircuitBuilder,
        profile_loader: ProfileLoader,
        num_steps: int = 10,
        step_seconds: float = 1.0,
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes a multi-step QSTS time-series simulation driven by ProfileLoader
        and returns:
          (features_stream_df [num_steps, 150], metadata_stream_df [num_steps, N])
        """
        feature_rows: List[pd.DataFrame] = []
        metadata_rows: List[Dict[str, Any]] = []

        # Initialize circuit
        circuit_builder.build_circuit()
        import opendssdirect as dss
        dss.Command(f"Set Mode=Daily StepSize={step_seconds}s Number=1")

        for step in range(num_steps):
            # 1. Fetch driving profile at current second
            step_profile = profile_loader.get_step_profile(step)
            setpoints = step_profile["mapped_setpoints"]
            raw_profiles = step_profile["raw_profiles"]

            # 2. Update circuit setpoints
            circuit_builder.set_operating_point(
                pv_kw=setpoints["pv_pmpp_kw"],
                load_a_kw=setpoints["load_a_kw"],
                load_b_kw=setpoints["load_b_kw"],
                bss_state="IDLING"
            )

            # 3. Step simulation
            dss.Command("Solve")
            if not dss.Solution.Converged():
                raise RuntimeError(f"OpenDSS failed to converge at step {step}")

            # 4. Export aligned telemetry row
            row_df, meta = self.export_telemetry_vector(
                circuit_builder=circuit_builder,
                profile_signals=raw_profiles,
                step=step,
                random_seed=random_seed
            )
            feature_rows.append(row_df)
            metadata_rows.append(meta)

        combined_features = pd.concat(feature_rows, ignore_index=True)
        combined_metadata = pd.DataFrame(metadata_rows)
        return combined_features, combined_metadata
