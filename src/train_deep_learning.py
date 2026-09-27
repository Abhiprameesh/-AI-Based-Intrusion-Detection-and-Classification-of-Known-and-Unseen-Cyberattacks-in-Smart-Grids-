import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.evaluate import evaluate_binary_classifier, evaluate_multiclass_classifier
from src.utils import ensure_dir, save_json, set_seed

# Device configuration (CPU on local system)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TabularMLP(nn.Module):
    def __init__(self, in_features, num_classes=1, hidden_dims=[128, 64, 32], dropout=0.2):
        super(TabularMLP, self).__init__()
        layers = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
        self.num_classes = num_classes

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            logits = self.forward(tensor_x)
            if self.num_classes == 1:
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                return np.vstack([1.0 - probs, probs]).T
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                return probs

    def predict(self, X):
        probas = self.predict_proba(X)
        if self.num_classes == 1:
            return (probas[:, 1] >= 0.5).astype(int)
        else:
            return np.argmax(probas, axis=1)

def train_tabular_mlp(X_train, y_train, X_val, y_val, num_classes=1, epochs=30, batch_size=256, lr=1e-3, random_state=42):
    set_seed(random_state)
    in_features = X_train.shape[1]
    model = TabularMLP(in_features=in_features, num_classes=num_classes).to(DEVICE)

    t_train_x = torch.tensor(X_train, dtype=torch.float32)
    t_val_x = torch.tensor(X_val, dtype=torch.float32)

    if num_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
        t_train_y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        t_val_y = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    else:
        criterion = nn.CrossEntropyLoss()
        t_train_y = torch.tensor(y_train, dtype=torch.long)
        t_val_y = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(t_train_x, t_train_y), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    best_weights = None

    t_start = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(t_train_x)

        model.eval()
        with torch.no_grad():
            val_out = model(t_val_x.to(DEVICE))
            val_loss = criterion(val_out, t_val_y.to(DEVICE)).item()

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - t_start
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    return model, train_time

class Conv1DTemporalNet(nn.Module):
    def __init__(self, in_channels, seq_len, num_classes=1):
        super(Conv1DTemporalNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        # x shape: (batch, seq_len, features) -> transpose to (batch, features, seq_len)
        x = x.transpose(1, 2)
        feat = self.conv(x).squeeze(-1)
        return self.fc(feat)

    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            logits = self.forward(tx)
            if self.num_classes == 1:
                p = torch.sigmoid(logits).cpu().numpy().flatten()
                return np.vstack([1.0 - p, p]).T
            else:
                return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X):
        probas = self.predict_proba(X)
        if self.num_classes == 1:
            return (probas[:, 1] >= 0.5).astype(int)
        else:
            return np.argmax(probas, axis=1)

class LSTMTemporalNet(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_layers=2, num_classes=1):
        super(LSTMTemporalNet, self).__init__()
        self.lstm = nn.LSTM(in_features, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.fc(last_step)

    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            logits = self.forward(tx)
            if self.num_classes == 1:
                p = torch.sigmoid(logits).cpu().numpy().flatten()
                return np.vstack([1.0 - p, p]).T
            else:
                return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X):
        probas = self.predict_proba(X)
        if self.num_classes == 1:
            return (probas[:, 1] >= 0.5).astype(int)
        else:
            return np.argmax(probas, axis=1)

class CNNLSTMTemporalNet(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_classes=1):
        super(CNNLSTMTemporalNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(32, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        # x: (batch, seq_len, in_features) -> conv expects (batch, in_features, seq_len)
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(x_conv)
        return self.fc(out[:, -1, :])

    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            tx = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            logits = self.forward(tx)
            if self.num_classes == 1:
                p = torch.sigmoid(logits).cpu().numpy().flatten()
                return np.vstack([1.0 - p, p]).T
            else:
                return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X):
        probas = self.predict_proba(X)
        if self.num_classes == 1:
            return (probas[:, 1] >= 0.5).astype(int)
        else:
            return np.argmax(probas, axis=1)
