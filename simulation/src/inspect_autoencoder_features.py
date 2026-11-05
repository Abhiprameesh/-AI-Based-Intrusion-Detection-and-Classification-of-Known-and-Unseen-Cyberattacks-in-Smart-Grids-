import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath("phase_1"))
from src.anomaly_detection import Autoencoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inspect_features():
    features_path = "simulation/results/milestone4/telemetry_features.csv"
    scaler_path = "phase_1/results/data4cyber_dataset/binary/models/preprocessor_scaled.joblib"
    weights_path = "phase_1/results/data4cyber_dataset/anomaly/recovery_v1/Autoencoder.pt"

    df_features = pd.read_csv(features_path)
    scaler = joblib.load(scaler_path)
    model = Autoencoder(in_features=150, bottleneck_dim=16).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()

    X_scaled = scaler.transform(df_features)

    with torch.no_grad():
        tx = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
        rec = model(tx)
        feature_mse = ((tx - rec) ** 2).mean(dim=0).cpu().numpy()

    feature_names = list(df_features.columns)
    df_err = pd.DataFrame({
        'feature': feature_names,
        'mean_squared_reconstruction_error': feature_mse,
        'scaled_input_mean': X_scaled.mean(axis=0),
        'scaled_input_std': X_scaled.std(axis=0)
    }).sort_values('mean_squared_reconstruction_error', ascending=False)

    print("--- TOP 15 FEATURES DRIVING RECONSTRUCTION ERROR (MSE) ---")
    for idx, row in df_err.head(15).iterrows():
        print(f"{row['feature']:40s}: MSE={row['mean_squared_reconstruction_error']:10.4f} | InpMean(scaled)={row['scaled_input_mean']:8.2f} | InpStd={row['scaled_input_std']:6.2f}")

    print(f"\nSum of top 15 features MSE: {df_err.head(15)['mean_squared_reconstruction_error'].sum():.4f}")
    print(f"Total MSE across all 150 features (mean): {feature_mse.mean():.4f}")
    print(f"Total MSE sum across 150 features: {feature_mse.sum():.4f}")

if __name__ == '__main__':
    inspect_features()
