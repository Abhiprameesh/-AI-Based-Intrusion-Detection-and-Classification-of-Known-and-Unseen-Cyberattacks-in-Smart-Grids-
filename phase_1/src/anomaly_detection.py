import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from src.utils import ensure_dir, save_json, save_model, set_seed

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Autoencoder(nn.Module):
    def __init__(self, in_features, bottleneck_dim=16):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, bottleneck_dim),
            nn.LeakyReLU(0.1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, in_features)
        )

    def forward(self, x):
        code = self.encoder(x)
        rec = self.decoder(code)
        return rec

    def compute_reconstruction_error(self, X):
        self.eval()
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            rec = self.forward(tx)
            mse = torch.mean((tx - rec) ** 2, dim=1).cpu().numpy()
        return mse

def train_autoencoder(X_normal_train, X_normal_val, epochs=30, batch_size=128, lr=1e-3, random_state=42):
    set_seed(random_state)
    in_features = X_normal_train.shape[1]
    model = Autoencoder(in_features=in_features).to(DEVICE)

    t_train = torch.tensor(X_normal_train, dtype=torch.float32)
    t_val = torch.tensor(X_normal_val, dtype=torch.float32)

    loader = DataLoader(TensorDataset(t_train), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_weights = None
    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        for (bx,) in loader:
            bx = bx.to(DEVICE)
            optimizer.zero_grad()
            rec = model(bx)
            loss = criterion(rec, bx)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            v_rec = model(t_val.to(DEVICE))
            val_loss = criterion(v_rec, t_val.to(DEVICE)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - t_start
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()

    # Calibrate threshold on normal validation data: 99th percentile
    val_errors = model.compute_reconstruction_error(X_normal_val)
    threshold = float(np.percentile(val_errors, 99.0))

    return model, threshold, train_time

def evaluate_anomaly_detector(name, score_func, X_test, y_test, threshold=None, is_distance=True, train_time=0.0):
    """
    Evaluates an anomaly detector.
    y_test: 0 = Normal, 1 = Anomaly/Attack.
    is_distance: True if higher score = anomaly (e.g. Autoencoder MSE),
                 False if lower score = anomaly (e.g. Isolation Forest).
    """
    start_inf = time.time()
    scores = score_func(X_test)
    inf_time = time.time() - start_inf
    latency = (inf_time / max(len(X_test), 1)) * 1000.0

    if threshold is None:
        # For sklearn models with -1 as anomaly, 1 as normal
        y_pred = (scores == -1).astype(int) if not is_distance else (scores >= np.median(scores)).astype(int)
    else:
        if is_distance:
            y_pred = (scores >= threshold).astype(int)
        else:
            y_pred = (scores <= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    else:
        tn = fp = fn = tp = fpr = fnr = 0.0

    roc_auc = None
    try:
        norm_scores = scores if is_distance else -scores
        roc_auc = roc_auc_score(y_test, norm_scores)
    except Exception:
        pass

    return {
        'model': name,
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'macro_f1': float(macro_f1),
        'fpr': float(fpr),
        'fnr': float(fnr),
        'roc_auc': float(roc_auc) if roc_auc is not None else None,
        'threshold': float(threshold) if threshold is not None else None,
        'confusion_matrix': cm.tolist(),
        'training_time_sec': float(train_time),
        'latency_per_sample_ms': float(latency),
        'num_test_samples': int(len(y_test))
    }
