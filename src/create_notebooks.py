import os
import json

WORKSPACE = r"c:\ALLMLPROJ\MAINMLPROJS\MAJOR PROJECT"
NOTEBOOKS_DIR = os.path.join(WORKSPACE, "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def create_notebook(filename, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.10"
            },
            "colab": {
                "provenance": []
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    path = os.path.join(NOTEBOOKS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {path}")

def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    }

def code(code_str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code_str.split("\n")]
    }

# ==============================================================================
# 01_dataset_discovery.ipynb
# ==============================================================================
nb1 = [
    md("""# Phase 1: Automated Smart Grid Dataset Discovery
### Final-Year Major Project: AI-Based Intrusion Detection in Smart Grids

This notebook automatically inspects the project directory, discovers all smart grid datasets, parses metadata, reads testbed documentation (`.docx`, `.xlsx`, `.md`), and compiles the Master Dataset Registry.

**Google Colab Instructions:**
Uncomment the Colab setup cell below if running on Google Colab."""),
    code("""# --- Google Colab Setup (Uncomment if running on Colab) ---
# !pip install -q pandas numpy scikit-learn openpyxl python-docx pyyaml
# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/MAJOR PROJECT"""),
    code("""import os
import sys
import glob
import pandas as pd
import numpy as np

# Working directory resolution (works locally and in Colab)
if os.path.exists('src'):
    PROJECT_ROOT = os.path.abspath('.')
elif os.path.exists('../src'):
    PROJECT_ROOT = os.path.abspath('..')
else:
    PROJECT_ROOT = os.path.abspath('.')
sys.path.insert(0, PROJECT_ROOT)

print(f"Project root set to: {PROJECT_ROOT}")"""),
    md("""## 1. Directory Tree & Dataset File Discovery
Search recursively for dataset files (CSV, DOCX, XLSX, JSON) across the project folder."""),
    code("""found_files = []
for root, dirs, files in os.walk(PROJECT_ROOT):
    # Skip virtual environments and hidden git folders
    if any(p in root for p in ['.venv', '.git', '__pycache__', '.system_generated']):
        continue
    for f in files:
        if f.endswith(('.csv', '.docx', '.xlsx', '.json')):
            rel_p = os.path.relpath(os.path.join(root, f), PROJECT_ROOT)
            size_mb = os.path.getsize(os.path.join(root, f)) / (1024 * 1024)
            found_files.append({'Relative_Path': rel_p, 'Size_MB': round(size_mb, 2)})

df_files = pd.DataFrame(found_files)
print(f"Total relevant dataset/doc files discovered: {len(df_files)}")
df_files.head(15)"""),
    md("""## 2. Load & Display the Master Dataset Registry
The automated registry categorizes all 6 discovered datasets across domain features (physical vs. cyber), rows, columns, and task feasibility."""),
    code("""reg_path = os.path.join(PROJECT_ROOT, "reports", "dataset_registry.csv")
if os.path.exists(reg_path):
    df_reg = pd.read_csv(reg_path)
    display(df_reg[['Dataset_ID', 'Dataset_Name', 'Rows', 'Columns', 'Number_of_Classes', 'Temporal_Data', 'Binary_Task_Available', 'Multiclass_Task_Available', 'Anomaly_Task_Available', 'Unseen_Attack_Task_Available']])
else:
    print("Run src/generate_registry_and_reports.py to generate dataset_registry.csv")"""),
    md("""## 3. Formatted Markdown Registry Details"""),
    code("""from IPython.display import Markdown, display
md_path = os.path.join(PROJECT_ROOT, "reports", "dataset_registry.md")
if os.path.exists(md_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        display(Markdown(f.read()))""")
]
create_notebook("01_dataset_discovery.ipynb", nb1)

# ==============================================================================
# 02_dataset_analysis.ipynb
# ==============================================================================
nb2 = [
    md("""# Phase 3: Comprehensive Dataset Audit & Data Quality Analysis
### Smart Grid Cyber-Physical Intrusion Detection

This notebook audits:
1. Data shapes, numerical/categorical data types
2. Missing values and infinity occurrences ($Z = V/I$ in transmission relays)
3. Duplicate records
4. Class balance and distribution
5. Critical data leakage vectors (`Flow ID`, IP addresses, `Attacker.*` logs)"""),
    code("""# --- Environment Setup ---
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if os.path.exists('src'):
    PROJECT_ROOT = os.path.abspath('.')
elif os.path.exists('../src'):
    PROJECT_ROOT = os.path.abspath('..')
else:
    PROJECT_ROOT = os.path.abspath('.')
sys.path.insert(0, PROJECT_ROOT)

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.size': 11})"""),
    md("""## 1. Inspecting Dataset 1: `binaryAllNaturalPlusNormalVsAttacks`
Checking impedance infinity values and class imbalance."""),
    code("""p1 = os.path.join(PROJECT_ROOT, "binaryAllNaturalPlusNormalVsAttacks", "data1.csv")
df1 = pd.read_csv(p1)
print(f"data1.csv Shape: {df1.shape}")
print(f"Target distribution:\\n{df1['marker'].value_counts(normalize=True)}")

# Check for infinity values in impedance columns
inf_counts = np.isinf(df1.select_dtypes(include=[np.number])).sum()
print(f"Columns with infinity values:\\n{inf_counts[inf_counts > 0]}")"""),
    md("""## 2. Inspecting Dataset 4: `CICFlowMeter_Balanced_DNP3_SCADA`
Checking direct leakage vectors (`Flow ID`, `Src IP`, `Dst IP`, `Timestamp`)."""),
    code("""train_p = os.path.join(PROJECT_ROOT, "CICFlowMeter_Training_Balanced.csv")
df_scada = pd.read_csv(train_p)
print(f"DNP3 SCADA Train Shape: {df_scada.shape}")
print(f"Classes ({df_scada['Label'].nunique()}):\\n{df_scada['Label'].value_counts()}")

leakage_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Timestamp']
print(f"Direct leakage columns present: {[c for c in leakage_cols if c in df_scada.columns]}")"""),
    md("""## 3. Inspecting Dataset 6: `data4cyber_dataset`
Multi-modal DER telemetry (PV solar, battery storage, AC meters, Modbus, MQTT)."""),
    code("""s0_p = os.path.join(PROJECT_ROOT, "data4cyber_dataset", "S0_benign_baseline", "dataset.csv")
df_s0 = pd.read_csv(s0_p)
print(f"Scenario S0 Benign Shape: {df_s0.shape}")
print(f"Sample electrical columns: {[c for c in df_s0.columns if 'p_sum3' in c or 'soc' in c.lower()][:6]}")
print(f"Sample cyber/protocol columns: {[c for c in df_s0.columns if 'modbus' in c.lower() or 'price' in c.lower()][:6]}")"""),
    md("""## 4. Visualizing Class Distributions Across Datasets"""),
    code("""from IPython.display import Image, display

# Display generated distribution plots
plots = [
    "reports/plots/data4cyber_dataset/family_distribution.png",
    "reports/plots/CICFlowMeter_Balanced_DNP3_SCADA/class_distribution.png",
    "reports/plots/binaryAllNaturalPlusNormalVsAttacks/class_distribution.png"
]
for p in plots:
    full_p = os.path.join(PROJECT_ROOT, p)
    if os.path.exists(full_p):
        print(f"Plot: {p}")
        display(Image(filename=full_p))""")
]
create_notebook("02_dataset_analysis.ipynb", nb2)

# ==============================================================================
# 03_binary_classification.ipynb
# ==============================================================================
nb3 = [
    md("""# Phase 7: Binary Attack Detection Experiment
### Smart Grid Cyber-Physical Intrusion Detection

This notebook trains, evaluates, and compares standard and deep learning models for binary intrusion detection (`Normal` vs. `Attack`):
1. **Logistic Regression** (Linear baseline with class weighting)
2. **Decision Tree** (Interpretable tree baseline)
3. **Random Forest** (Ensemble bagging)
4. **XGBoost** (Gradient boosted trees)
5. **Support Vector Machine (SVM)** (Margin classifier with probability calibration)
6. **Deep Tabular MLP** (PyTorch Neural Network with BatchNorm, LeakyReLU, and Dropout)

**Target Metrics:** Accuracy, Attack Recall, Attack Precision, Macro F1, False Positive Rate (FPR), False Negative Rate (FNR), Inference Latency."""),
    code("""# --- Colab / Local Setup ---
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score

if os.path.exists('src'):
    PROJECT_ROOT = os.path.abspath('.')
elif os.path.exists('../src'):
    PROJECT_ROOT = os.path.abspath('..')
else:
    PROJECT_ROOT = os.path.abspath('.')
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_data4cyber, load_cicflowmeter_dnp3, get_preprocessor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on Device: {DEVICE}")"""),
    md("""## 1. Load Dataset (`data4cyber_dataset` as Primary Cyber-Physical Benchmark)
Strictly cleans ground-truth attacker logs to eliminate target leakage."""),
    code("""print("Loading data4cyber_dataset...")
data = load_data4cyber(random_state=42)

X_train, y_train = data['X_train'], data['y_train_binary']
X_val, y_val = data['X_val'], data['y_val_binary']
X_test, y_test = data['X_test'], data['y_test_binary']
features = data['feature_names']

print(f"Training samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")
print(f"Feature count: {len(features)}")
print(f"Class counts in Test: {np.bincount(y_test)} (0: Benign, 1: Attack)")"""),
    md("""## 2. Preprocessing Pipeline
Fitted **strictly on training data only** to prevent data snooping."""),
    code("""prep_scaled = get_preprocessor(scaling=True)
prep_unscaled = get_preprocessor(scaling=False)

# Fit on Train only
X_tr_sc = prep_scaled.fit_transform(X_train)
X_va_sc = prep_scaled.transform(X_val)
X_te_sc = prep_scaled.transform(X_test)

X_tr_raw = prep_unscaled.fit_transform(X_train)
X_te_raw = prep_unscaled.transform(X_test)"""),
    md("""## 3. Train Classical Machine Learning Models"""),
    code("""models = {
    "Logistic_Regression": (LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42), True),
    "Decision_Tree": (DecisionTreeClassifier(max_depth=15, class_weight='balanced', random_state=42), False),
    "Random_Forest": (RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced_subsample', n_jobs=-1, random_state=42), False),
    "XGBoost": (XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric='logloss', random_state=42), False),
    "SVM": (SGDClassifier(loss='modified_huber', max_iter=1000, class_weight='balanced', random_state=42), True)
}

results = []
trained_models = {}

for name, (clf, scaled) in models.items():
    print(f"Training {name}...")
    X_tr = X_tr_sc if scaled else X_tr_raw
    X_te = X_te_sc if scaled else X_te_raw
    
    t0 = time.time()
    clf.fit(X_tr, y_train)
    tr_time = time.time() - t0
    trained_models[name] = clf
    
    t0 = time.time()
    y_pred = clf.predict(X_te)
    inf_time_ms = ((time.time() - t0) / len(X_te)) * 1000.0
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Attack_Recall': round(rec, 4),
        'Attack_Precision': round(prec, 4),
        'Macro_F1': round(macro_f1, 4),
        'FPR': round(fpr, 4),
        'FNR': round(fnr, 4),
        'Latency_ms': round(inf_time_ms, 4),
        'Training_Time_s': round(tr_time, 2)
    })

df_res = pd.DataFrame(results)
display(df_res)"""),
    md("""## 4. Train Deep Tabular Neural Network (PyTorch MLP)"""),
    code("""class TabularMLP(nn.Module):
    def __init__(self, in_features):
        super(TabularMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

mlp = TabularMLP(X_tr_sc.shape[1]).to(DEVICE)
crit = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

tx_tr = torch.tensor(X_tr_sc, dtype=torch.float32)
ty_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx_tr, ty_tr), batch_size=128, shuffle=True)

print("Training PyTorch Tabular MLP...")
mlp.train()
for ep in range(15):
    for bx, by in loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        opt.zero_grad()
        loss = crit(mlp(bx), by)
        loss.backward()
        opt.step()

mlp.eval()
with torch.no_grad():
    tx_te = torch.tensor(X_te_sc, dtype=torch.float32).to(DEVICE)
    logits = mlp(tx_te)
    probs = torch.sigmoid(logits).cpu().numpy().flatten()
    mlp_preds = (probs >= 0.5).astype(int)

mlp_f1 = f1_score(y_test, mlp_preds, average='macro')
mlp_rec = recall_score(y_test, mlp_preds)
print(f"Deep MLP Macro F1: {mlp_f1:.4f} | Recall: {mlp_rec:.4f}")"""),
    md("""## 5. Visualizing Confusion Matrices & Feature Importances"""),
    code("""plt.figure(figsize=(12, 4.5))

plt.subplot(1, 2, 1)
cm_xgb = confusion_matrix(y_test, trained_models['XGBoost'].predict(X_te_raw))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.subplot(1, 2, 2)
xgb = trained_models['XGBoost']
imp = pd.Series(xgb.feature_importances_, index=features).sort_values(ascending=False).head(10)
imp.plot(kind='barh', color='#2b5c8f')
plt.title("Top 10 Most Important Features (XGBoost)")
plt.xlabel("Relative Importance")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()""")
]
create_notebook("03_binary_classification.ipynb", nb3)

# ==============================================================================
# 04_multiclass_classification.ipynb
# ==============================================================================
nb4 = [
    md("""# Phase 8: Multiclass Known-Attack Classification
### Smart Grid Cyber-Physical Intrusion Detection

This notebook evaluates multiclass models to attribute network traffic and power telemetry to specific attack types:
- **`data4cyber_dataset`**: 4 DER attack families (`benign`, `industroyer_modbus`, `mitm_false_data_injection`, `mqtt_price_signal_manipulation`).
- **`CICFlowMeter_Balanced_DNP3_SCADA`**: 11 DNP3 SCADA attack classes."""),
    code("""import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

if os.path.exists('src'):
    PROJECT_ROOT = os.path.abspath('.')
elif os.path.exists('../src'):
    PROJECT_ROOT = os.path.abspath('..')
else:
    PROJECT_ROOT = os.path.abspath('.')
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_data4cyber, get_preprocessor

sns.set_theme(style="whitegrid")"""),
    md("""## 1. Load Multiclass DER Attack Campaigns (`data4cyber_dataset`)"""),
    code("""data = load_data4cyber(random_state=42)
X_train, y_train = data['X_train'], data['y_train_family']
X_test, y_test = data['X_test'], data['y_test_family']
class_names = data['class_names_family']
features = data['feature_names']

print("Class Names:", class_names)
print("Test distribution across families:\\n", pd.Series(y_test).value_counts())"""),
    md("""## 2. Train Multiclass Classifiers"""),
    code("""prep = get_preprocessor(scaling=False)
X_tr = prep.fit_transform(X_train)
X_te = prep.transform(X_test)

models = {
    "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='mlogloss')
}

for name, clf in models.items():
    print(f"Training {name}...")
    t0 = time.time()
    clf.fit(X_tr, y_train)
    tr_time = time.time() - t0
    
    y_pred = clf.predict(X_te)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"  -> {name} Macro F1: {macro_f1:.4f} (Trained in {tr_time:.2f}s)")
    print(f"\\nClassification Report for {name}:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))"""),
    md("""## 3. Multiclass Confusion Matrix Heatmap"""),
    code("""plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, models['Random_Forest'].predict(X_te))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.title("Random Forest Multiclass Attack Family Attribution", fontweight='bold')
plt.xlabel("Predicted Attack Family")
plt.ylabel("Actual Attack Family")
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.show()""")
]
create_notebook("04_multiclass_classification.ipynb", nb4)

# ==============================================================================
# 05_anomaly_detection.ipynb
# ==============================================================================
nb5 = [
    md("""# Phase 13: Anomaly Detection via Deep Autoencoder & Isolation Forest
### Smart Grid Cyber-Physical Intrusion Detection

In real-world smart grids, attackers constantly invent novel techniques that have no prior training signatures.
**Semi-supervised anomaly detection** models normal operational telemetry (`Normal` baseline only) and flags any statistical or physical deviations."""),
    code("""import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

if os.path.exists('src'):
    PROJECT_ROOT = os.path.abspath('.')
elif os.path.exists('../src'):
    PROJECT_ROOT = os.path.abspath('..')
else:
    PROJECT_ROOT = os.path.abspath('.')
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_data4cyber, get_preprocessor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on Device: {DEVICE}")"""),
    md("""## 1. Load Data & Isolate Pure Normal Baseline
Scenario S0 represents uncontaminated benign DER operations."""),
    code("""data = load_data4cyber(random_state=42)
prep = get_preprocessor(scaling=True)

X_tr_sc = prep.fit_transform(data['X_train'])
X_te_sc = prep.transform(data['X_test'])

benign_idx = list(data['label_encoder'].classes_).index('benign')

# Training set: ONLY benign normal data
X_norm_train = X_tr_sc[data['y_train_family'] == benign_idx]
print(f"Pure Normal Training Samples: {len(X_norm_train)}")

# Test set: Both normal and attack data
y_test_binary = data['y_test_binary']
print(f"Test Set: {len(X_te_sc)} (Normal: {np.sum(y_test_binary == 0)}, Attack: {np.sum(y_test_binary == 1)})")"""),
    md("""## 2. Define & Train Deep Autoencoder"""),
    code("""class GridAutoencoder(nn.Module):
    def __init__(self, in_features, latent_dim=16):
        super(GridAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, latent_dim),
            nn.LeakyReLU(0.1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, in_features)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

ae = GridAutoencoder(X_norm_train.shape[1], latent_dim=16).to(DEVICE)
crit = nn.MSELoss()
opt = torch.optim.Adam(ae.parameters(), lr=1e-3)

tx_norm = torch.tensor(X_norm_train, dtype=torch.float32)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx_norm), batch_size=64, shuffle=True)

print("Training Autoencoder on Normal Baseline...")
ae.train()
for ep in range(20):
    for (bx,) in loader:
        bx = bx.to(DEVICE)
        opt.zero_grad()
        loss = crit(ae(bx), bx)
        loss.backward()
        opt.step()

# Compute reconstruction errors on Normal Training to set threshold (95th percentile)
ae.eval()
with torch.no_grad():
    recon_tr = ae(tx_norm.to(DEVICE))
    err_tr = torch.mean((tx_norm.to(DEVICE) - recon_tr)**2, dim=1).cpu().numpy()

threshold = np.percentile(err_tr, 95.0)
print(f"Calibrated Anomaly Threshold (95th percentile of normal): {threshold:.5f}")"""),
    md("""## 3. Evaluate Anomaly Detection on Test Set"""),
    code("""with torch.no_grad():
    tx_te = torch.tensor(X_te_sc, dtype=torch.float32).to(DEVICE)
    recon_te = ae(tx_te)
    err_te = torch.mean((tx_te - recon_te)**2, dim=1).cpu().numpy()

preds = (err_te >= threshold).astype(int)

acc = accuracy_score(y_test_binary, preds)
rec = recall_score(y_test_binary, preds)
f1 = f1_score(y_test_binary, preds, average='macro')
fpr = np.mean(preds[y_test_binary == 0])

print(f"Anomaly Detection Performance:")
print(f"  Accuracy: {acc:.4f} | Macro F1: {f1:.4f} | Attack Recall: {rec:.4f} | False Alarm Rate (FPR): {fpr:.4f}")"""),
    md("""## 4. Plot Reconstruction Error Distribution"""),
    code("""plt.figure(figsize=(10, 5))
sns.kdeplot(err_te[y_test_binary == 0], label='Normal Operational Baseline', fill=True, color='green', common_norm=False)
sns.kdeplot(err_te[y_test_binary == 1], label='Cyberattack Deviations', fill=True, color='red', common_norm=False)
plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
plt.title("Autoencoder Reconstruction Residual Distribution", fontweight='bold')
plt.xlabel("Mean Squared Reconstruction Error")
plt.ylabel("Density")
plt.xlim(0, np.percentile(err_te, 99))
plt.legend()
plt.tight_layout()
plt.show()""")
]
create_notebook("05_anomaly_detection.ipynb", nb5)

# ==============================================================================
# 06_unseen_attack_detection.ipynb
# ==============================================================================
nb6 = [
    md("""# Phase 12: Unseen / Zero-Day Cyberattack Detection
### Smart Grid Cyber-Physical Intrusion Detection

**The Research Goal:**
Determine whether an Intrusion Detection System can recognize a completely unseen attack class as anomalous rather than incorrectly classifying it as normal or assigning it to a known class.

**Experimental Setup:**
- **Training Set**: Benign + Known Attacks (`industroyer_modbus`, `mqtt_price_signal_manipulation`).
- **Held-Out Zero-Day Attack**: `mitm_false_data_injection` (MitM FDI).
- **Rule Enforcement**: The held-out attack is NEVER used during training, scaling, feature selection, or threshold tuning!"""),
    code("""import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score

if os.path.exists('src'):
    PROJECT_ROOT = os.path.abspath('.')
elif os.path.exists('../src'):
    PROJECT_ROOT = os.path.abspath('..')
else:
    PROJECT_ROOT = os.path.abspath('.')
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_data4cyber, get_preprocessor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sns.set_theme(style="whitegrid")"""),
    md("""## 1. Zero-Day Partitioning"""),
    code("""data = load_data4cyber(random_state=42)
prep = get_preprocessor(scaling=True)

X_tr_sc = prep.fit_transform(data['X_train'])
X_te_sc = prep.transform(data['X_test'])

le = data['label_encoder']
unseen_name = 'mitm_false_data_injection'
unseen_idx = list(le.classes_).index(unseen_name)
benign_idx = list(le.classes_).index('benign')

print(f"Held-out Unseen Zero-Day Attack: {unseen_name}")

# Normal training data
X_norm_tr = X_tr_sc[data['y_train_family'] == benign_idx]

# Split test sets
X_te_norm = X_te_sc[data['y_test_family'] == benign_idx]
X_te_known = X_te_sc[(data['y_test_family'] != benign_idx) & (data['y_test_family'] != unseen_idx)]
X_te_unseen = X_te_sc[data['y_test_family'] == unseen_idx]

print(f"Test Normal Samples: {len(X_te_norm)}")
print(f"Test Known Attack Samples: {len(X_te_known)}")
print(f"Test Held-Out Zero-Day Attack Samples: {len(X_te_unseen)}")"""),
    md("""## 2. Train Autoencoder on Normal Baseline"""),
    code("""class GridAutoencoder(nn.Module):
    def __init__(self, in_features, latent_dim=16):
        super(GridAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, latent_dim),
            nn.LeakyReLU(0.1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, in_features)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

ae = GridAutoencoder(X_norm_tr.shape[1], latent_dim=16).to(DEVICE)
crit = nn.MSELoss()
opt = torch.optim.Adam(ae.parameters(), lr=1e-3)

tx_tr = torch.tensor(X_norm_tr, dtype=torch.float32)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx_tr), batch_size=64, shuffle=True)

ae.train()
for ep in range(20):
    for (bx,) in loader:
        bx = bx.to(DEVICE)
        opt.zero_grad()
        loss = crit(ae(bx), bx)
        loss.backward()
        opt.step()

ae.eval()
with torch.no_grad():
    recon_tr = ae(tx_tr.to(DEVICE))
    err_tr = torch.mean((tx_tr.to(DEVICE) - recon_tr)**2, dim=1).cpu().numpy()

threshold = np.percentile(err_tr, 95.0)
print(f"Anomaly Detection Threshold: {threshold:.5f}")"""),
    md("""## 3. Zero-Day Evaluation: Can the Autoencoder Detect the Unseen Attack?"""),
    code("""with torch.no_grad():
    def get_err(X):
        tx = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        return torch.mean((tx - ae(tx))**2, dim=1).cpu().numpy()

err_norm = get_err(X_te_norm)
err_known = get_err(X_te_known)
err_unseen = get_err(X_te_unseen)

unseen_det_rate = np.mean(err_unseen >= threshold)
known_det_rate = np.mean(err_known >= threshold)
false_alarm_rate = np.mean(err_norm >= threshold)

print("="*60)
print(f"ZERO-DAY ATTACK EVALUATION ({unseen_name}):")
print(f"  Unseen Attack Detection Rate: {unseen_det_rate * 100:.2f}%")
print(f"  Known Attack Detection Rate:  {known_det_rate * 100:.2f}%")
print(f"  False Alarm Rate on Normal:   {false_alarm_rate * 100:.2f}%")
print("="*60)"""),
    md("""## 4. Visualizing Zero-Day Separation vs. Normal Operations"""),
    code("""plt.figure(figsize=(10, 5))
sns.kdeplot(err_norm, label='Normal Telemetry', color='green', fill=True, common_norm=False)
sns.kdeplot(err_known, label='Known Attacks (Modbus/MQTT)', color='orange', fill=True, common_norm=False)
sns.kdeplot(err_unseen, label=f'Unseen Zero-Day ({unseen_name})', color='red', fill=True, common_norm=False)
plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
plt.title(f"Zero-Day Detection: Reconstruction Deviation on Unseen {unseen_name}", fontweight='bold')
plt.xlabel("Reconstruction Residual Error")
plt.ylabel("Density")
plt.xlim(0, np.percentile(err_unseen, 95))
plt.legend()
plt.tight_layout()
plt.show()""")
]
create_notebook("06_unseen_attack_detection.ipynb", nb6)

# ==============================================================================
# 07_final_comparison.ipynb
# ==============================================================================
nb7 = [
    md("""# Phase 16-20: Master Benchmark Comparison, Dataset Selection & Final Architecture
### Smart Grid Cyber-Physical Intrusion Detection Major Project

This notebook synthesizes all experimental results across the 5 suitable smart grid datasets:
1. **Master Model Comparison table** (54 experimental configurations across Binary, Multiclass, Anomaly, and Temporal tasks)
2. **Dataset-level comparison matrix**
3. **Performance vs. Latency trade-off analysis**
4. **Final dataset selection rationale**
5. **Final multi-tier Smart Grid IDS architecture**

**Google Colab Instructions:**
- **Option 1 (Direct Upload)**: Upload this notebook to Google Colab, click the folder icon on the left panel, and upload `final_model_comparison.csv` and `dataset_comparison.csv` (from your `reports/` folder).
- **Option 2 (Google Drive)**: Upload your project folder to Google Drive and uncomment the Drive mount cell below."""),
    code("""# --- Google Colab Setup (Uncomment if using Google Drive) ---
# from google.colab import drive
# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/MAJOR PROJECT"""),
    code("""import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display

# Automatic project root / file finder (works on Local & Colab)
def resolve_file(filename, subfolder="reports"):
    candidates = [
        os.path.join(".", subfolder, filename),
        os.path.join("..", subfolder, filename),
        os.path.join(".", filename),
        os.path.join("/content", subfolder, filename),
        os.path.join("/content", filename)
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return filename

sns.set_theme(style="whitegrid")
print("Environment initialized.")"""),
    md("""## 1. Master Model Comparison Table (54 Experiments Across All Datasets)"""),
    code("""models_csv = resolve_file("final_model_comparison.csv", "reports")
if os.path.exists(models_csv):
    models_df = pd.read_csv(models_csv)
    print(f"Total benchmarked models: {len(models_df)}")
    display(models_df)
else:
    print(f"File not found: {models_csv}. Please upload final_model_comparison.csv to Colab.")"""),
    md("""## 2. Master Dataset Comparison Matrix"""),
    code("""ds_csv = resolve_file("dataset_comparison.csv", "reports")
if os.path.exists(ds_csv):
    ds_df = pd.read_csv(ds_csv)
    display(ds_df)
else:
    print(f"File not found: {ds_csv}. Please upload dataset_comparison.csv to Colab.")"""),
    md("""## 3. Visualizing Detection Performance vs. Inference Latency Trade-Off"""),
    code("""if os.path.exists(models_csv):
    bin_df = models_df[models_df['Task'] == 'binary'].dropna(subset=['Inference_Time', 'Macro_F1'])
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=bin_df, x='Inference_Time', y='Macro_F1', hue='Model', style='Dataset_Name', s=140)
    plt.xscale('log')
    plt.title("Detection Performance vs. Inference Latency Trade-Off", fontsize=14, fontweight='bold')
    plt.xlabel("Inference Latency per Sample (ms, Log Scale)", fontsize=12)
    plt.ylabel("Macro F1 Score", fontsize=12)
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()"""),
    md("""## 4. Final Dataset Selection Report"""),
    code("""sel_md = resolve_file("final_dataset_selection.md", "reports")
if os.path.exists(sel_md):
    with open(sel_md, 'r', encoding='utf-8') as f:
        display(Markdown(f.read()))
else:
    print(f"Optional markdown report not found: {sel_md}")"""),
    md("""## 5. Final Experimental Recommendation Report"""),
    code("""rep_md = resolve_file("final_experimental_report.md", "reports")
if os.path.exists(rep_md):
    with open(rep_md, 'r', encoding='utf-8') as f:
        display(Markdown(f.read()))
else:
    print(f"Optional markdown report not found: {rep_md}")""")
]
create_notebook("07_final_comparison.ipynb", nb7)

print("\nAll 7 comprehensive, Colab-ready notebooks created successfully!")
