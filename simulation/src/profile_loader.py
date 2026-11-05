"""
Profile Loader and Validator for OpenDSS Simulation.
Loads and validates external driving time-series from Data4Cyber S0 baseline:
- Solar irradiance (W/m^2) and ambient temperature (deg C)
- Load profiles (load1_kw, load2_kw)
- PV capability curves (pv_input_kw, pv_active_kw, pv_reactive_kvar)
- Dynamic electricity price signal (MQTT-Price-Signal.price_eur_per_kwh)
"""

import os
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class ProfileLoader:
    """
    Loads, audits, and serves synchronized driving profiles for the microgrid simulator.
    """

    PROFILE_COLUMNS = [
        "Profile.ambient_temperature",
        "Profile.load1_kw",
        "Profile.load2_kw",
        "Profile.pv_active_kw",
        "Profile.pv_input_kw",
        "Profile.pv_reactive_kvar",
        "Profile.total_irradiance",
        "MQTT-Price-Signal.price_eur_per_kwh",
    ]

    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            # Default to S0 dataset.csv
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_path = os.path.join(root_dir, "data4cyber_dataset", "S0_benign_baseline", "dataset.csv")
        self.data_path = data_path
        self._df: Optional[pd.DataFrame] = None
        self._audit_results: Dict[str, Any] = {}
        self.load_and_audit()

    def load_and_audit(self) -> Dict[str, Any]:
        """
        Loads the CSV and audits timestamps, sampling regularity, missing values, and ranges.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset baseline file not found: {self.data_path}")

        df = pd.read_csv(self.data_path)
        self._df = df

        # 1. Timestamp audit
        ts = pd.to_datetime(df["timestamp"])
        deltas = ts.diff().dt.total_seconds().dropna()
        dt_unique = deltas.unique().tolist()
        is_uniform_1s = (len(dt_unique) == 1 and dt_unique[0] == 1.0)

        # 2. Check profile columns
        missing_cols = [c for c in self.PROFILE_COLUMNS if c not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing expected profile columns in {self.data_path}: {missing_cols}")

        column_stats = {}
        for col in self.PROFILE_COLUMNS:
            null_count = int(df[col].isnull().sum())
            nunique = int(df[col].nunique())
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            column_stats[col] = {
                "null_count": null_count,
                "nunique": nunique,
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "is_static": (nunique == 1)
            }

        self._audit_results = {
            "source_file": self.data_path,
            "total_rows": len(df),
            "timestamp_start": str(ts.min()),
            "timestamp_end": str(ts.max()),
            "sampling_interval_seconds": 1.0 if is_uniform_1s else dt_unique,
            "is_uniform_1s": is_uniform_1s,
            "total_nulls": int(df[self.PROFILE_COLUMNS].isnull().sum().sum()),
            "column_stats": column_stats
        }
        return self._audit_results

    @property
    def audit_summary(self) -> Dict[str, Any]:
        """Returns the audit summary dictionary."""
        return self._audit_results

    @property
    def total_steps(self) -> int:
        """Total available rows in the profile dataset."""
        return len(self._df) if self._df is not None else 0

    def get_step_profile(self, step: int) -> Dict[str, Any]:
        """
        Returns the profile values for a given 0-indexed step.
        Wraps around if step >= total_steps.
        Returns:
          Dictionary with profile signals, timestamp, and mapped OpenDSS setpoints.
        """
        if self._df is None:
            self.load_and_audit()

        idx = step % len(self._df)
        row = self._df.iloc[idx]

        # Extract genuine profile values
        p_dict = {col: float(row[col]) for col in self.PROFILE_COLUMNS}
        p_dict["timestamp"] = str(row["timestamp"])

        # Mapped OpenDSS physical setpoints
        # In Data4Cyber, Profile.load1_kw and Profile.load2_kw are positive consumption demand (kW)
        load_a_kw = float(p_dict["Profile.load1_kw"])
        load_b_kw = float(p_dict["Profile.load2_kw"])

        # In Data4Cyber, Profile.pv_active_kw is negative generation (e.g. -20.409 kW)
        # OpenDSS PVSystem pmpp requires positive generation magnitude (kW)
        pv_kw = abs(float(p_dict["Profile.pv_active_kw"]))

        # Irradiance normalized (nominal 1000 W/m^2 = 1.0 pu)
        irr_w_m2 = float(p_dict["Profile.total_irradiance"])

        return {
            "raw_profiles": p_dict,
            "mapped_setpoints": {
                "pv_pmpp_kw": pv_kw,
                "pv_irradiance_pu": irr_w_m2 / 1000.0 if irr_w_m2 > 0 else 1.0,
                "load_a_kw": load_a_kw,
                "load_b_kw": load_b_kw,
                "temperature_deg_c": float(p_dict["Profile.ambient_temperature"]),
            }
        }
