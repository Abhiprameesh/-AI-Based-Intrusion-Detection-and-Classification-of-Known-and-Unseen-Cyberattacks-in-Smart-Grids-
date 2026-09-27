import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

def select_top_features_rf(X_train, y_train, feature_names, top_k=30, random_state=42):
    """
    Selects top K features based on Random Forest feature importances fitted on training data.
    """
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=random_state)
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:top_k]

    selected_features = [feature_names[i] for i in sorted_idx]
    selected_indices = sorted_idx.tolist()

    return {
        'selected_features': selected_features,
        'selected_indices': selected_indices,
        'importances': importances[sorted_idx].tolist(),
        'model': rf
    }
