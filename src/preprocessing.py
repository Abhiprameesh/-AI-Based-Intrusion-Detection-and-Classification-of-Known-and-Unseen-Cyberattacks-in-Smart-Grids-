from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
import numpy as np

def clean_inf_and_nan(df, numeric_cols):
    """Replaces positive and negative infinite values with NaN."""
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df

def get_preprocessor(scaling=True, robust=False):
    """
    Returns an sklearn Pipeline with SimpleImputer (median)
    and optional StandardScaler or RobustScaler.
    Fitted strictly on training sets.
    """
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if scaling:
        scaler = RobustScaler() if robust else StandardScaler()
        steps.append(('scaler', scaler))
    return Pipeline(steps)
