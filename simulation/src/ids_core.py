"""
IDS Core: Unified In-Memory Inference and Evaluation Engine for Smart Grid Microgrid Telemetry.

Integrates the three-stage cyber-physical Intrusion Detection System (IDS):
- Stage 1: Frozen Binary XGBoost Classifier (Attack Detection: Normal vs. Malicious)
  Thresholds: Primary tau = 0.50, Diagnostic tau = 0.05.
- Stage 2: Frozen Multiclass Random Forest Classifier (Attack Identification: Benign, Industroyer, MitM FDI)
- Stage 3: Recovered Deep Autoencoder (Unseen Anomaly Detection via Reconstruction MSE)
  Thresholds: Recovered tau = 0.36402026, Historical tau = 0.42760131.

Strict Scientific Guardrails:
1. Zero model retraining, zero parameter fine-tuning.
2. Exact 150-feature schema matching Phase 1 preprocessors.
3. Strict isolation of ground-truth metadata from model input tensors.
4. Non-destructive: does NOT overwrite canonical milestone baseline files.
"""

import os
import sys
import time
import hashlib
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Root directory resolution
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(ROOT_DIR, "simulation") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT_DIR, "simulation"))

from src.scenario_runner import ScenarioRunner
from src.telemetry_exporter import TelemetryExporter
from src.attack_injector import (
    BaseAttack,
    FalseDataInjectionAttack,
    UnauthorizedControlAttack,
    TelemetryDropoutAttack,
    UnseenStealthyRampAttack,
    BenignDisturbance,
)


# -----------------------------------------------------------------------------
# Deep Autoencoder Architecture (Bit-for-bit matched to Phase 1 PyTorch model)
# -----------------------------------------------------------------------------
class DeepAutoencoder(nn.Module):
    """
    Deep Autoencoder with 150-feature input, 16-dimensional bottleneck,
    and LeakyReLU activations matching Phase 1 Data4Cyber recovery model.
    """
    def __init__(self, in_features: int = 150, bottleneck_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, bottleneck_dim),
            nn.LeakyReLU(0.1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def compute_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Computes per-sample Mean Squared Error (MSE) across all 150 features."""
        self.eval()
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32)
            rec = self.forward(tx)
            mse = torch.mean((tx - rec) ** 2, dim=1).cpu().numpy()
        return mse


# -----------------------------------------------------------------------------
# Scenario Catalog (S0 through S8)
# -----------------------------------------------------------------------------
def _build_scenario_definitions() -> Dict[str, Dict[str, Any]]:
    return {
        "S0_benign_baseline": {
            "code": "S0",
            "name": "Benign Baseline",
            "category": "benign",
            "is_unseen": False,
            "description": "Nominal benign microgrid operation without cyber or physical disturbances.",
            "attack_factory": lambda: None,
        },
        "S1_fdi_bss_scaling_0.5": {
            "code": "S1",
            "name": "FDI Scaling (BSS)",
            "category": "known_attack",
            "is_unseen": False,
            "description": "FDI attack scaling BSS active power and current readings by 0.5 (MitM / ARP spoofing).",
            "attack_factory": lambda: FalseDataInjectionAttack(
                name="fdi_bss_scaling_0.5",
                target_meter="BSS-Inverter-AC-Meter",
                start_step=40,
                end_step=110,
                scale_factor=0.5,
                bias_w=0.0,
            ),
        },
        "S2_fdi_substation_bias_5kw": {
            "code": "S2",
            "name": "FDI Bias (Substation)",
            "category": "known_attack",
            "is_unseen": False,
            "description": "FDI attack injecting +5 kW false import bias on Substation meter.",
            "attack_factory": lambda: FalseDataInjectionAttack(
                name="fdi_substation_bias_5kw",
                target_meter="Substation-AC-Meter",
                start_step=40,
                end_step=110,
                scale_factor=1.0,
                bias_w=5000.0,
            ),
        },
        "S3_industroyer_pv_trip": {
            "code": "S3",
            "name": "Industroyer PV Trip",
            "category": "known_attack",
            "is_unseen": False,
            "description": "Unauthorized actuator command forcing complete PV inverter curtailment (0 kW).",
            "attack_factory": lambda: UnauthorizedControlAttack(
                name="industroyer_pv_trip",
                control_target="PV",
                start_step=40,
                end_step=110,
                forced_pv_kw=0.0,
            ),
        },
        "S4_industroyer_bss_forced_discharge": {
            "code": "S4",
            "name": "Industroyer BSS Discharge",
            "category": "known_attack",
            "is_unseen": False,
            "description": "Unauthorized actuator command forcing battery discharge at 15 kW with spoofed normal SOC telemetry.",
            "attack_factory": lambda: UnauthorizedControlAttack(
                name="industroyer_bss_forced_discharge",
                control_target="BSS",
                start_step=40,
                end_step=110,
                forced_bss_state="DISCHARGING",
                forced_bss_kw=15.0,
                spoof_reported_soc=True,
                spoofed_soc_value=0.20,
            ),
        },
        "S5_load_b_telemetry_freeze": {
            "code": "S5",
            "name": "Load-B Telemetry Freeze",
            "category": "known_attack",
            "is_unseen": False,
            "description": "Telemetry manipulation freezing Load-B meter reading (replay stale value for 70s).",
            "attack_factory": lambda: TelemetryDropoutAttack(
                name="load_b_telemetry_freeze",
                target_meter="Load-B-AC-Meter",
                start_step=40,
                end_step=110,
                dropout_mode="freeze",
            ),
        },
        "S6_unseen_stealthy_ramp_fdi": {
            "code": "S6",
            "name": "Unseen Stealthy Ramp FDI",
            "category": "unseen_attack",
            "is_unseen": True,
            "description": "Unseen gradual linear drift (+25 W/s) on PV telemetry to evade point-anomaly thresholds.",
            "attack_factory": lambda: UnseenStealthyRampAttack(
                name="unseen_stealthy_ramp_fdi",
                stealth_mode="ramp_fdi",
                target_meter="PV-Inverter-AC-Meter",
                start_step=30,
                end_step=120,
                ramp_rate_w_per_sec=25.0,
                max_bias_w=2250.0,
            ),
        },
        "S7_unseen_coordinated_masking": {
            "code": "S7",
            "name": "Unseen Coordinated Masking",
            "category": "unseen_attack",
            "is_unseen": True,
            "description": "Unseen coordinated attack: physical 10 kW battery discharge covertly masked at substation meter.",
            "attack_factory": lambda: UnseenStealthyRampAttack(
                name="unseen_coordinated_masking",
                stealth_mode="coordinated_masking",
                target_meter="Substation-AC-Meter",
                start_step=30,
                end_step=120,
                forced_bss_discharge_kw=10.0,
            ),
        },
        "S8_benign_cloud_cover_shading": {
            "code": "S8",
            "name": "Benign Cloud Shading",
            "category": "benign_disturbance",
            "is_unseen": False,
            "description": "Controlled benign physical transient: 70% irradiance dip. Ground truth is strictly normal (Y=0).",
            "attack_factory": lambda: BenignDisturbance(
                name="benign_cloud_cover_shading",
                disturbance_type="cloud_shading",
                start_step=40,
                end_step=100,
                irradiance_drop_fraction=0.70,
            ),
        },
    }


SCENARIO_CATALOG = _build_scenario_definitions()


# -----------------------------------------------------------------------------
# Core Unified Inference Engine
# -----------------------------------------------------------------------------
class IDSInferenceEngine:
    """
    Unified In-Memory Inference Engine executing Stage 1, 2, and 3 IDS models.
    """

    # Model & Artifact Relative Paths
    PATHS = {
        "preprocessor_unscaled": os.path.join(
            ROOT_DIR, "phase_1", "results", "data4cyber_dataset", "binary", "models", "preprocessor_unscaled.joblib"
        ),
        "preprocessor_scaled": os.path.join(
            ROOT_DIR, "phase_1", "results", "data4cyber_dataset", "binary", "models", "preprocessor_scaled.joblib"
        ),
        "xgb_binary": os.path.join(
            ROOT_DIR, "phase_1", "results", "data4cyber_dataset", "binary", "models", "XGBoost.joblib"
        ),
        "rf_multiclass": os.path.join(
            ROOT_DIR, "phase_1", "results", "data4cyber_dataset", "multiclass", "models", "Random_Forest_multiclass.joblib"
        ),
        "autoencoder_weights": os.path.join(
            ROOT_DIR, "phase_1", "results", "data4cyber_dataset", "anomaly", "recovery_v1", "Autoencoder.pt"
        ),
        "threshold_metadata": os.path.join(
            ROOT_DIR, "phase_1", "results", "data4cyber_dataset", "anomaly", "recovery_v1", "threshold_metadata.json"
        ),
    }

    # Threshold Constants
    STAGE1_PRIMARY_THRESHOLD: float = 0.50
    STAGE1_DIAGNOSTIC_THRESHOLD: float = 0.05
    STAGE3_RECOVERED_THRESHOLD: float = 0.3640202581882477
    STAGE3_HISTORICAL_THRESHOLD: float = 0.42760130763053894

    STAGE2_CLASSES = {
        0: "Benign",
        1: "Industroyer (Modbus)",
        2: "MitM FDI",
        3: "MQTT Price Signal",
    }

    MULTICLASS_FAMILY_MAPPING = {
        "benign": 0,
        "benign_disturbance": 0,
        "unauthorized_pv_trip": 1,
        "unauthorized_bss_discharge": 1,
        "fdi_scaling": 2,
        "fdi_bias": 2,
        # Excluded from Phase 1 training:
        "telemetry_freeze": -1,
        "unseen_stealthy_attack": -1,
    }

    def __init__(self, eager_load: bool = True):
        self.runner = ScenarioRunner()
        self.exporter = TelemetryExporter()

        self._preprocessor_unscaled = None
        self._preprocessor_scaled = None
        self._xgb_binary = None
        self._rf_multiclass = None
        self._autoencoder = None
        self._expected_features = None

        if eager_load:
            self._load_all_models()

    def _load_all_models(self):
        """Loads and verifies all 5 preprocessors and models into memory."""
        # Unscaled preprocessor
        if not os.path.exists(self.PATHS["preprocessor_unscaled"]):
            raise FileNotFoundError(f"Missing unscaled preprocessor: {self.PATHS['preprocessor_unscaled']}")
        self._preprocessor_unscaled = joblib.load(self.PATHS["preprocessor_unscaled"])
        self._expected_features = list(self._preprocessor_unscaled.feature_names_in_)

        # Scaled preprocessor
        if not os.path.exists(self.PATHS["preprocessor_scaled"]):
            raise FileNotFoundError(f"Missing scaled preprocessor: {self.PATHS['preprocessor_scaled']}")
        self._preprocessor_scaled = joblib.load(self.PATHS["preprocessor_scaled"])

        # Stage 1: XGBoost binary
        if not os.path.exists(self.PATHS["xgb_binary"]):
            raise FileNotFoundError(f"Missing XGBoost model: {self.PATHS['xgb_binary']}")
        self._xgb_binary = joblib.load(self.PATHS["xgb_binary"])

        # Stage 2: Random Forest multiclass
        if not os.path.exists(self.PATHS["rf_multiclass"]):
            raise FileNotFoundError(f"Missing Random Forest model: {self.PATHS['rf_multiclass']}")
        self._rf_multiclass = joblib.load(self.PATHS["rf_multiclass"])

        # Stage 3: Deep Autoencoder
        if not os.path.exists(self.PATHS["autoencoder_weights"]):
            raise FileNotFoundError(f"Missing Autoencoder weights: {self.PATHS['autoencoder_weights']}")
        self._autoencoder = DeepAutoencoder(in_features=150, bottleneck_dim=16)
        state_dict = torch.load(self.PATHS["autoencoder_weights"], weights_only=True)
        self._autoencoder.load_state_dict(state_dict)
        self._autoencoder.eval()

    @property
    def expected_features(self) -> List[str]:
        if self._expected_features is None:
            self._load_all_models()
        return list(self._expected_features)

    def get_scenario_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Returns metadata for all available scenarios."""
        return SCENARIO_CATALOG

    def run_scenario_simulation(
        self,
        scenario_id: str,
        num_steps: int = 150,
        initial_soc_percent: float = 20.0,
        random_seed: int = 42,
        slack_pu: Optional[float] = None,
        load_a_kvar: Optional[float] = None,
        load_b_kvar: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes a simulation scenario and returns (features_df [N, 150], metadata_df [N, M]).
        """
        if scenario_id not in SCENARIO_CATALOG:
            raise ValueError(f"Unknown scenario_id '{scenario_id}'. Available: {list(SCENARIO_CATALOG.keys())}")

        sc_info = SCENARIO_CATALOG[scenario_id]
        attack = sc_info["attack_factory"]()

        features_df, metadata_df = self.runner.run_scenario(
            num_steps=num_steps,
            attack=attack,
            initial_soc_percent=initial_soc_percent,
            random_seed=random_seed,
            slack_pu=slack_pu,
            load_a_kvar=load_a_kvar,
            load_b_kvar=load_b_kvar,
        )

        # Append scenario identification to metadata
        metadata_df["scenario_id"] = scenario_id
        metadata_df["scenario_category"] = sc_info["category"]
        metadata_df["is_unseen_scenario"] = 1 if sc_info["is_unseen"] else 0

        return features_df, metadata_df

    def predict_stage1(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Stage 1: Binary Attack Detection (XGBoost).
        Returns attack probabilities and binary predictions at tau=0.50 and tau=0.05.
        """
        if self._xgb_binary is None:
            self._load_all_models()

        X_unscaled = self._preprocessor_unscaled.transform(features_df)
        t_start = time.perf_counter()
        probabilities = self._xgb_binary.predict_proba(X_unscaled)[:, 1]
        elapsed = time.perf_counter() - t_start

        preds_primary = (probabilities >= self.STAGE1_PRIMARY_THRESHOLD).astype(int)
        preds_diagnostic = (probabilities >= self.STAGE1_DIAGNOSTIC_THRESHOLD).astype(int)

        return {
            "probabilities": probabilities,
            "predictions_primary": preds_primary,
            "predictions_diagnostic": preds_diagnostic,
            "threshold_primary": self.STAGE1_PRIMARY_THRESHOLD,
            "threshold_diagnostic": self.STAGE1_DIAGNOSTIC_THRESHOLD,
            "latency_ms_per_sample": (elapsed / max(len(features_df), 1)) * 1000.0,
        }

    def predict_stage2(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Stage 2: Known Attack Multiclass Identification (Random Forest).
        Returns predicted classes, class labels, and 4-class probability distribution.
        """
        if self._rf_multiclass is None:
            self._load_all_models()

        X_unscaled = self._preprocessor_unscaled.transform(features_df)
        t_start = time.perf_counter()
        pred_classes = self._rf_multiclass.predict(X_unscaled)
        class_probs = self._rf_multiclass.predict_proba(X_unscaled)
        elapsed = time.perf_counter() - t_start

        pred_labels = [self.STAGE2_CLASSES.get(c, f"Class {c}") for c in pred_classes]

        return {
            "predicted_classes": pred_classes,
            "predicted_labels": pred_labels,
            "class_probabilities": class_probs,
            "classes": [0, 1, 2, 3],
            "class_names": [self.STAGE2_CLASSES[c] for c in range(4)],
            "latency_ms_per_sample": (elapsed / max(len(features_df), 1)) * 1000.0,
        }

    def predict_stage3(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Stage 3: Unseen Anomaly Detection (Deep Autoencoder).
        Returns reconstruction MSE per sample and anomaly decisions at both thresholds.
        """
        if self._autoencoder is None:
            self._load_all_models()

        X_scaled = self._preprocessor_scaled.transform(features_df)
        t_start = time.perf_counter()
        mse = self._autoencoder.compute_reconstruction_error(X_scaled)
        elapsed = time.perf_counter() - t_start

        preds_recovered = (mse >= self.STAGE3_RECOVERED_THRESHOLD).astype(int)
        preds_historical = (mse >= self.STAGE3_HISTORICAL_THRESHOLD).astype(int)

        return {
            "reconstruction_mse": mse,
            "predictions_recovered": preds_recovered,
            "predictions_historical": preds_historical,
            "threshold_recovered": self.STAGE3_RECOVERED_THRESHOLD,
            "threshold_historical": self.STAGE3_HISTORICAL_THRESHOLD,
            "latency_ms_per_sample": (elapsed / max(len(features_df), 1)) * 1000.0,
        }

    def predict_all_stages(
        self,
        features_df: pd.DataFrame,
        metadata_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Executes unified inference across Stage 1, Stage 2, and Stage 3.
        If metadata_df is provided, computes stage performance metrics.
        """
        s1 = self.predict_stage1(features_df)
        s2 = self.predict_stage2(features_df)
        s3 = self.predict_stage3(features_df)

        results = {
            "num_samples": len(features_df),
            "stage1": s1,
            "stage2": s2,
            "stage3": s3,
            "metrics": None,
        }

        if metadata_df is not None:
            results["metrics"] = self._compute_evaluation_metrics(s1, s2, s3, metadata_df)

        return results

    def _compute_evaluation_metrics(
        self,
        s1: Dict[str, Any],
        s2: Dict[str, Any],
        s3: Dict[str, Any],
        metadata_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Computes rigorous confusion matrix metrics across all 3 stages."""
        y_true_bin = metadata_df["attack_active"].values.astype(int)
        num_attacks = int((y_true_bin == 1).sum())
        num_benign = int((y_true_bin == 0).sum())

        def calc_binary(y_pred):
            acc = float(accuracy_score(y_true_bin, y_pred))
            prec = float(precision_score(y_true_bin, y_pred, zero_division=0))
            rec = float(recall_score(y_true_bin, y_pred, zero_division=0)) if num_attacks > 0 else None
            f1 = float(f1_score(y_true_bin, y_pred, zero_division=0))
            cm = confusion_matrix(y_true_bin, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            tnr = float(tn / (fp + tn)) if (fp + tn) > 0 else 0.0
            return {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "fpr": fpr,
                "tnr": tnr,
                "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            }

        # Stage 1 metrics
        stage1_primary = calc_binary(s1["predictions_primary"])
        stage1_diagnostic = calc_binary(s1["predictions_diagnostic"])

        # Stage 2 metrics (evaluated on supported classes)
        multiclass_targets = []
        is_supported = []
        for _, row in metadata_df.iterrows():
            if row["attack_active"] == 0:
                multiclass_targets.append(0)
                is_supported.append(True)
            else:
                fam = row.get("attack_family", "benign")
                mapped = self.MULTICLASS_FAMILY_MAPPING.get(fam, -1)
                multiclass_targets.append(mapped)
                is_supported.append(mapped != -1)

        multiclass_targets = np.array(multiclass_targets)
        is_supported = np.array(is_supported)

        stage2_metrics = {
            "supported_samples": int(is_supported.sum()),
            "excluded_samples": int((~is_supported).sum()),
            "overall_accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
        }

        if is_supported.sum() > 0:
            y_multi_true = multiclass_targets[is_supported]
            y_multi_pred = s2["predicted_classes"][is_supported]
            stage2_metrics["overall_accuracy"] = float(accuracy_score(y_multi_true, y_multi_pred))
            stage2_metrics["macro_f1"] = float(f1_score(y_multi_true, y_multi_pred, average="macro", zero_division=0))
            stage2_metrics["weighted_f1"] = float(f1_score(y_multi_true, y_multi_pred, average="weighted", zero_division=0))

        # Stage 3 metrics
        stage3_recovered = calc_binary(s3["predictions_recovered"])
        stage3_historical = calc_binary(s3["predictions_historical"])

        return {
            "num_total_samples": len(y_true_bin),
            "num_attack_samples": num_attacks,
            "num_benign_samples": num_benign,
            "stage1_primary": stage1_primary,
            "stage1_diagnostic": stage1_diagnostic,
            "stage2_multiclass": stage2_metrics,
            "stage3_recovered": stage3_recovered,
            "stage3_historical": stage3_historical,
        }

    def run_and_evaluate(
        self,
        scenario_id: str,
        num_steps: int = 150,
        initial_soc_percent: float = 20.0,
        random_seed: int = 42,
        slack_pu: Optional[float] = None,
        load_a_kvar: Optional[float] = None,
        load_b_kvar: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes a single scenario simulation and all three IDS inference stages,
        returning features, metadata, predictions, and metrics in memory.
        """
        features_df, metadata_df = self.run_scenario_simulation(
            scenario_id=scenario_id,
            num_steps=num_steps,
            initial_soc_percent=initial_soc_percent,
            random_seed=random_seed,
            slack_pu=slack_pu,
            load_a_kvar=load_a_kvar,
            load_b_kvar=load_b_kvar,
        )

        results = self.predict_all_stages(features_df, metadata_df=metadata_df)
        results["features_df"] = features_df
        results["metadata_df"] = metadata_df
        results["scenario_id"] = scenario_id
        results["scenario_info"] = SCENARIO_CATALOG[scenario_id]

        return results

    @staticmethod
    def load_canonical_baseline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads the precomputed canonical Milestone 4 baseline files:
        - telemetry_features.csv (1350 x 150)
        - telemetry_metadata.csv (1350 rows)
        - predictions_binary_and_multiclass.csv (1350 rows)
        """
        feat_path = os.path.join(ROOT_DIR, "simulation", "results", "milestone4", "telemetry_features.csv")
        meta_path = os.path.join(ROOT_DIR, "simulation", "results", "milestone4", "telemetry_metadata.csv")
        pred_path = os.path.join(ROOT_DIR, "simulation", "results", "milestone4", "predictions_binary_and_multiclass.csv")

        if not (os.path.exists(feat_path) and os.path.exists(meta_path) and os.path.exists(pred_path)):
            raise FileNotFoundError("Canonical Milestone 4 baseline files are missing from simulation/results/milestone4/")

        features_df = pd.read_csv(feat_path)
        metadata_df = pd.read_csv(meta_path)
        predictions_df = pd.read_csv(pred_path)

        return features_df, metadata_df, predictions_df

