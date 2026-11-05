"""
Milestone 4.3 Workstream D: Data4Cyber Autoencoder Recovery Script.
Reconstructs the original documented Phase 1 training procedure with deterministic seed locking.
Trains strictly on benign S0 training partition, calibrates threshold on benign S0 validation partition,
serializes weights to a versioned directory, verifies reload, and records complete provenance manifest.
"""

import os
import sys
import time
import json
import hashlib
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure imports from phase_1
sys.path.append(os.path.abspath("phase_1"))
from src.data_loader import load_data4cyber
from src.anomaly_detection import Autoencoder
from src.unseen_attack import evaluate_unseen_attack_detection
from src.utils import set_seed, ensure_dir, save_json

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def run_autoencoder_recovery():
    print("=" * 80)
    print("STARTING WORKSTREAM D: DATA4CYBER AUTOENCODER RECOVERY")
    print("=" * 80)

    # 1. Environment & Provenance Capture
    env_info = {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'pytorch_version': torch.__version__,
        'numpy_version': np.__version__,
        'sklearn_version': sklearn.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device_used': str(DEVICE)
    }
    print("Environment:", json.dumps(env_info, indent=2))

    # 2. Output Paths (Versioned strictly, avoiding overwriting)
    recovery_dir = os.path.abspath("phase_1/results/data4cyber_dataset/anomaly/recovery_v1")
    ensure_dir(recovery_dir)
    weights_path = os.path.join(recovery_dir, "Autoencoder.pt")
    config_path = os.path.join(recovery_dir, "model_config.json")
    curves_path = os.path.join(recovery_dir, "training_curves.json")
    manifest_path = os.path.join(recovery_dir, "recovery_manifest.json")
    threshold_path = os.path.join(recovery_dir, "threshold_metadata.json")

    # 3. Load Data & Scaler
    print("\n[Step 1] Loading Data4Cyber dataset with random_state=42...")
    data = load_data4cyber(random_state=42)
    le = data['label_encoder']
    benign_idx = list(le.classes_).index('benign')
    mitm_idx = list(le.classes_).index('mitm_false_data_injection')

    scaler_path = "phase_1/results/data4cyber_dataset/binary/models/preprocessor_scaled.joblib"
    print(f"[Step 2] Loading frozen scaler from: {scaler_path}")
    prep = joblib.load(scaler_path)
    scaler_hash = compute_file_hash(scaler_path)

    # Transform partitions using frozen scaler
    print("[Step 3] Transforming train, val, and test partitions with scaler...")
    X_tr_sc = prep.transform(data['X_train'])
    X_val_sc = prep.transform(data['X_val'])
    X_te_sc = prep.transform(data['X_test'])

    # Benign-only subsets for Autoencoder
    X_norm_tr = X_tr_sc[data['y_train_family'] == benign_idx]
    X_norm_val = X_val_sc[data['y_val_family'] == benign_idx]

    # Test subsets for historical validation
    X_te_norm = X_te_sc[data['y_test_family'] == benign_idx]
    X_te_known = X_te_sc[(data['y_test_family'] != benign_idx) & (data['y_test_family'] != mitm_idx)]
    X_te_unseen = X_te_sc[data['y_test_family'] == mitm_idx]

    split_counts = {
        'total_train': int(len(data['X_train'])),
        'total_val': int(len(data['X_val'])),
        'total_test': int(len(data['X_test'])),
        'benign_train_samples': int(len(X_norm_tr)),
        'benign_val_samples': int(len(X_norm_val)),
        'benign_test_samples': int(len(X_te_norm)),
        'known_attack_test_samples': int(len(X_te_known)),
        'unseen_attack_test_samples': int(len(X_te_unseen)),
        'in_features': int(X_norm_tr.shape[1])
    }
    print("Partition sample counts:", json.dumps(split_counts, indent=2))

    # 4. Hyperparameters
    hyperparams = {
        'architecture': 'Autoencoder(150 -> 64 -> 32 -> 16 -> 32 -> 64 -> 150)',
        'in_features': 150,
        'bottleneck_dim': 16,
        'epochs': 25,
        'batch_size': 64,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'optimizer': 'Adam',
        'loss_function': 'MSELoss',
        'random_seed': 42
    }

    # 5. Model Initialization & Training
    print("\n[Step 4] Initializing Autoencoder and setting deterministic seed 42...")
    set_seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    model = Autoencoder(in_features=150, bottleneck_dim=16).to(DEVICE)

    t_train = torch.tensor(X_norm_tr, dtype=torch.float32)
    t_val = torch.tensor(X_norm_val, dtype=torch.float32)

    # Use a fixed generator for DataLoader shuffle
    gen = torch.Generator()
    gen.manual_seed(42)
    loader = DataLoader(TensorDataset(t_train), batch_size=64, shuffle=True, generator=gen)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_weights = None
    epoch_records = []
    t_start = time.time()

    print("\n[Step 5] Commencing training across 25 epochs...")
    for epoch in range(1, 26):
        model.train()
        batch_losses = []
        for (bx,) in loader:
            bx = bx.to(DEVICE)
            optimizer.zero_grad()
            rec = model(bx)
            loss = criterion(rec, bx)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))

        model.eval()
        with torch.no_grad():
            v_rec = model(t_val.to(DEVICE))
            val_loss = float(criterion(v_rec, t_val.to(DEVICE)).item())

        is_best = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            is_best = True

        epoch_records.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'is_best': is_best
        })
        print(f"  Epoch {epoch:2d}/25 | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} {'[BEST]' if is_best else ''}")

    train_duration = time.time() - t_start
    print(f"\nTraining completed in {train_duration:.2f} seconds. Best Val MSE: {best_val_loss:.6f}")

    # Load best checkpoint
    assert best_weights is not None, "Best weights were not captured"
    model.load_state_dict(best_weights)
    model.eval()

    # 6. Threshold Calibration on Benign Validation Data (99th percentile)
    print("\n[Step 6] Calibrating anomaly threshold on benign validation partition...")
    val_errors = model.compute_reconstruction_error(X_norm_val)
    new_threshold = float(np.percentile(val_errors, 99.0))
    val_err_stats = {
        'count': int(len(val_errors)),
        'min': float(np.min(val_errors)),
        'p25': float(np.percentile(val_errors, 25)),
        'median': float(np.median(val_errors)),
        'mean': float(np.mean(val_errors)),
        'p75': float(np.percentile(val_errors, 75)),
        'p95': float(np.percentile(val_errors, 95)),
        'p99': new_threshold,
        'max': float(np.max(val_errors)),
        'std': float(np.std(val_errors))
    }

    # Historical threshold comparison
    historical_threshold = 0.42760130763053894
    diff_abs = abs(new_threshold - historical_threshold)
    diff_pct = (diff_abs / historical_threshold) * 100.0

    print(f"  Calculated Recovery Threshold (99th pct): {new_threshold:.8f}")
    print(f"  Historical Phase 1 Threshold:           {historical_threshold:.8f}")
    print(f"  Absolute Discrepancy:                   {diff_abs:.8f}")
    print(f"  Relative Discrepancy:                   {diff_pct:.4f}%")

    # 7. Evaluate on Historical Data4Cyber Test Partitions (Sanity Check)
    print("\n[Step 7] Evaluating on historical Data4Cyber test set (Sanity Check)...")
    hist_eval_new_thresh = evaluate_unseen_attack_detection(
        model, X_te_norm, X_te_known, X_te_unseen,
        model_type="autoencoder", threshold=new_threshold, results_dir=None, attack_name="MitM_FDI_Recovery"
    )
    hist_eval_old_thresh = evaluate_unseen_attack_detection(
        model, X_te_norm, X_te_known, X_te_unseen,
        model_type="autoencoder", threshold=historical_threshold, results_dir=None, attack_name="MitM_FDI_HistoricalThresh"
    )

    print("  Sanity check with new threshold:", json.dumps(hist_eval_new_thresh, indent=2))

    # 8. Save Weights and Metadata
    print(f"\n[Step 8] Serializing weights to: {weights_path}")
    torch.save(model.state_dict(), weights_path)
    weights_hash = compute_file_hash(weights_path)
    weights_size = os.path.getsize(weights_path)

    # Reload check
    print("[Step 9] Verifying reloadability and finite output validation...")
    reloaded_model = Autoencoder(in_features=150, bottleneck_dim=16).to(DEVICE)
    reloaded_model.load_state_dict(torch.load(weights_path, weights_only=True))
    reloaded_model.eval()

    dummy_input = np.random.randn(5, 150).astype(np.float32)
    dummy_errors = reloaded_model.compute_reconstruction_error(dummy_input)
    assert dummy_errors.shape == (5,), f"Expected shape (5,), got {dummy_errors.shape}"
    assert np.all(np.isfinite(dummy_errors)), "Reloaded model produced non-finite reconstruction error"
    print("  Reload validation PASSED. All outputs finite.")

    # Save artifacts
    save_json(hyperparams, config_path)
    save_json({
        'training_time_seconds': train_duration,
        'best_val_loss': best_val_loss,
        'epochs': epoch_records
    }, curves_path)
    save_json({
        'calculated_threshold': new_threshold,
        'historical_threshold': historical_threshold,
        'absolute_difference': diff_abs,
        'percent_difference': diff_pct,
        'validation_error_distribution': val_err_stats,
        'historical_test_evaluation_with_new_threshold': hist_eval_new_thresh,
        'historical_test_evaluation_with_historical_threshold': hist_eval_old_thresh
    }, threshold_path)

    manifest = {
        'artifact_name': 'Data4Cyber Deep Autoencoder (Recovery v1)',
        'created_at_utc': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'purpose': 'Milestone 4.3 Workstream D Reproducible Recovery',
        'weights_file': {
            'relative_path': 'phase_1/results/data4cyber_dataset/anomaly/recovery_v1/Autoencoder.pt',
            'absolute_path': weights_path,
            'sha256': weights_hash,
            'bytes': weights_size
        },
        'scaler_used': {
            'relative_path': 'phase_1/results/data4cyber_dataset/binary/models/preprocessor_scaled.joblib',
            'sha256': scaler_hash
        },
        'environment': env_info,
        'hyperparameters': hyperparams,
        'data_splits': split_counts,
        'training_summary': {
            'duration_seconds': train_duration,
            'best_val_loss': best_val_loss,
            'final_train_loss': epoch_records[-1]['train_loss']
        },
        'threshold': {
            'calculated_val_p99': new_threshold,
            'historical_p99': historical_threshold,
            'difference_percent': diff_pct
        },
        'reload_validation_passed': True
    }
    save_json(manifest, manifest_path)
    print(f"Manifest written to: {manifest_path}")
    print("\nWORKSTREAM D EXECUTION COMPLETE.")

if __name__ == '__main__':
    run_autoencoder_recovery()
