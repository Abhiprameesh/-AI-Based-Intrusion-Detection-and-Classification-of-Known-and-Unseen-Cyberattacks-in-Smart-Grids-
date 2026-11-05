"""
Dashboard Utilities and Inference Engine Cache Management.
Provides safe caching, data loading, and execution helpers for the Smart Grid IDS Streamlit application.
"""

import os
import sys
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import streamlit as st

# Safe path resolution to project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
SIMULATION_DIR = os.path.join(ROOT_DIR, "simulation")
if SIMULATION_DIR not in sys.path:
    sys.path.insert(0, SIMULATION_DIR)

from simulation.src.ids_core import IDSInferenceEngine, SCENARIO_CATALOG


@st.cache_resource(show_spinner="Initializing AI Models and Simulation Engine...")
def get_engine() -> IDSInferenceEngine:
    """
    Cached singleton provider for IDSInferenceEngine.
    Loads Phase 1 frozen preprocessors, XGBoost binary classifier,
    Random Forest multiclass classifier, and Deep Autoencoder once into memory.
    """
    engine = IDSInferenceEngine(eager_load=True)
    return engine


@st.cache_data(show_spinner=False)
def load_canonical_baseline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cached loader for the canonical Milestone 4 baseline dataset (1,350 samples).
    Returns (features_df, metadata_df, predictions_df).
    """
    return IDSInferenceEngine.load_canonical_baseline()


def check_system_health() -> Dict[str, Any]:
    """
    Verifies that OpenDSS, model artifacts, and baseline datasets are intact.
    Returns status indicators and file availability details.
    """
    status = {
        "opendss": False,
        "models": False,
        "baselines": False,
        "details": {},
    }

    # 1. OpenDSS direct check
    try:
        import opendssdirect as dss
        dss.Command("Clear")
        status["opendss"] = True
        status["details"]["opendss_version"] = getattr(dss, "__version__", "Available")
    except Exception as e:
        status["details"]["opendss_error"] = str(e)

    # 2. Check model files existence
    model_paths = IDSInferenceEngine.PATHS
    missing_models = [k for k, p in model_paths.items() if not os.path.exists(p)]
    if not missing_models:
        status["models"] = True
        status["details"]["models_status"] = "All 5 artifacts verified on disk"
    else:
        status["details"]["missing_models"] = missing_models

    # 3. Check baseline dataset existence
    feat_path = os.path.join(ROOT_DIR, "simulation", "results", "milestone4", "telemetry_features.csv")
    meta_path = os.path.join(ROOT_DIR, "simulation", "results", "milestone4", "telemetry_metadata.csv")
    if os.path.exists(feat_path) and os.path.exists(meta_path):
        status["baselines"] = True
    else:
        status["details"]["baseline_missing"] = True

    return status


def run_scenario_execution(
    scenario_id: str,
    num_steps: int = 150,
    initial_soc_percent: float = 20.0,
    random_seed: int = 42,
    slack_pu: Optional[float] = None,
    load_a_kvar: Optional[float] = None,
    load_b_kvar: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Executes a single scenario simulation and passes telemetry through all 3 IDS stages.
    Non-destructive: results are stored strictly in memory without overwriting disk artifacts.
    """
    engine = get_engine()
    results = engine.run_and_evaluate(
        scenario_id=scenario_id,
        num_steps=num_steps,
        initial_soc_percent=initial_soc_percent,
        random_seed=random_seed,
        slack_pu=slack_pu,
        load_a_kvar=load_a_kvar,
        load_b_kvar=load_b_kvar,
    )
    return results


def get_default_scenario_data() -> Dict[str, Any]:
    """
    Loads S0 benign baseline data as default session state on startup.
    Runs in-memory inference on S0 without modifying canonical baseline files.
    """
    engine = get_engine()
    return engine.run_and_evaluate("S0_benign_baseline", num_steps=150)

