# AI-Based Intrusion Detection and Classification of Known and Unseen Cyberattacks in Smart Grids

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4-red.svg)](https://xgboost.ai/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📌 Executive Summary
This repository houses the end-to-end experimental benchmarking and development framework for a multi-tier **Smart Grid Intrusion Detection System (IDS)**. The project systematically evaluates **54 machine learning and deep learning configurations** across 5 distinct smart-grid cybersecurity datasets to determine the optimal detection architecture for:
1. **Line-Rate Binary Attack Detection** (Normal vs. Malicious Activity)
2. **Fine-Grained Multiclass Attack Attribution** (Known Attack Families & Protocol Commands)
3. **Zero-Day / Unseen Cyberattack Detection** (Novelty & Anomaly Detection)
4. **Temporal Waveform Dynamics** (Continuous 30 Hz synchrophasor PMU measurements)

---

## 🏆 Key Experimental Results

| Task | Selected Best Model | Macro F1 | Attack Recall | FPR | Latency per Sample | Key Strengths & Selection Justification |
|---|---|---|---|---|---|---|
| **Binary Attack Detection** | **XGBoost** | **0.9986** | **99.92%** | **0.22%** | **0.0023 ms** | State-of-the-art recall; 13x faster inference than Random Forest; low false positive burden on grid operators. |
| **Multiclass Attribution** | **Random Forest** | **1.0000** | **100.0%** | N/A | **0.0080 ms** | Robust multi-scale feature handling; perfectly separates complex DER exploit campaigns; calibrated class probabilities. |
| **Zero-Day / Unseen Attack** | **Deep Autoencoder** | **0.9520** | **94.8%** | **4.20%** | **0.0018 ms** | Trained exclusively on uncorrupted benign baseline (Scenario S0); flags zero-day False Data Injection without prior signatures. |
| **Temporal Waveform Dynamics** | **CNN-LSTM** | **0.9398** | **93.85%** | **5.72%** | **0.0142 ms** | Spatial convolutional feature extraction combined with recurrent memory captures transient 30 Hz synchrophasor waveform faults. |

---

## 📊 Dataset Evaluation & Primary Selection

Following a comprehensive audit across all available testbeds, **`data4cyber_dataset`** was selected as the **Primary Dataset** for the final project lifecycle:

- **Primary Selected Dataset:** **`data4cyber_dataset`** (DER Multi-Modal Telemetry)
  - **Why:** The only dataset featuring **true Cyber-Physical multi-modality**: combines cyber telemetry (Modbus register read/write traces, MQTT dynamic price signals, raw PCAP network captures) with physical grid telemetry (PV solar generation, battery State of Charge, AC active/reactive power, voltage, and grid frequency).
  - Directly supports upcoming phases: **Threat Hunting, Risk Assessment, and Safe Reinforcement Learning Autonomous Defense**.
- **Secondary Waveform Benchmark:** **`PMU_Data_with_Anomalies_and_Events`** (IEEE SGSMA 2024 New England 39-bus system with 8 PMUs at 30 Hz).

---

## 🏗️ Recommended Multi-Tier System Architecture

```
Incoming Smart Grid Telemetry (Data4Cyber Multi-Modal Stream)
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Tier 1: Line-Rate Binary Detection (XGBoost)           │
│ Latency: 0.0023 ms | Macro F1: 99.86% | FPR: 0.22%     │
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
       [Normal: Pass]             [Attack Flagged]
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        ▼                                                                 ▼
┌───────────────────────────────────┐             ┌────────────────────────────────────────┐
│ Tier 2: Attack Attribution        │             │ Tier 3: Zero-Day Anomaly Safeguard     │
│ (Random Forest - Multiclass)      │             │ (Deep Autoencoder Reconstruction)      │
│ Classifies Known Threat Families  │             │ Catches Unseen Exploits (94.8% Rate)   │
└─────────────────┬─────────────────┘             └───────────────────┬────────────────────┘
                  │                                                   │
                  └─────────────────────────┬─────────────────────────┘
                                            ▼
                       [Threat Hunting, Risk Assessment &
                         Safe RL Autonomous Defense]
```

---

## 📁 Repository Directory Structure

```text
.
├── phase_1/                                # Phase 1: Model Benchmarking & Dataset Selection
│   ├── README.md                           # Phase 1 documentation & executive findings
│   ├── experiment_config.yaml              # Reproducibility configuration & parameters
│   ├── models/                             # Saved models registry (.gitkeep tracked)
│   ├── notebooks/                          # 7 Interactive, Colab-ready Jupyter Notebooks
│   │   ├── 01_dataset_discovery.ipynb      # Automated dataset discovery & registration
│   │   ├── 02_dataset_analysis.ipynb       # Data audit, leakage checks & distributions
│   │   ├── 03_binary_classification.ipynb  # Binary attack detection training suite
│   │   ├── 04_multiclass_classification.ipynb # Multiclass known-attack attribution
│   │   ├── 05_anomaly_detection.ipynb      # Deep Autoencoder & Isolation Forest
│   │   ├── 06_unseen_attack_detection.ipynb # Zero-day / held-out attack evaluation
│   │   └── 07_final_comparison.ipynb       # Master benchmark comparison & selection
│   │
│   ├── reports/                            # Generated reports, matrices, and plots
│   │   ├── dataset_registry.csv            # Automated dataset catalog
│   │   ├── dataset_registry.md             # Detailed dataset documentation
│   │   ├── dataset_comparison.csv          # Cross-dataset comparative matrix
│   │   ├── final_model_comparison.csv      # Master 54-model benchmark results
│   │   ├── final_dataset_selection.md      # Multi-criteria dataset selection report
│   │   ├── final_experimental_report.md    # Complete academic experimental report
│   │   └── plots/                          # Confusion matrices, ROC/PR curves & trade-offs
│   │
│   ├── results/                            # JSON evaluation metrics & CSV logs
│   └── src/                                # Modular, reusable Python package
│       ├── data_loader.py                  # Robust data loaders with leakage prevention
│       ├── preprocessing.py                # Preprocessing pipelines fitted strictly on train
│       ├── feature_selection.py            # Feature importance filtering
│       ├── train_classical.py              # Classical ML training suite
│       ├── train_deep_learning.py          # PyTorch MLP, 1D-CNN, LSTM, CNN-LSTM
│       ├── anomaly_detection.py            # PyTorch Deep Autoencoders & Isolation Forest
│       ├── unseen_attack.py                # Zero-day evaluation & open-set rejection
│       ├── evaluate.py                     # Security-critical metric computation
│       └── visualization.py                # Publication-grade plotting utilities
│
├── requirements.txt                        # Locked Python virtual environment dependencies
└── .gitignore                              # Clean Git filter (excludes heavy binaries/data)
```

---

## 🚀 Getting Started

### 1. Local Python Environment Setup

```bash
# Clone the repository
git clone https://github.com/Abhiprameesh/-AI-Based-Intrusion-Detection-and-Classification-of-Known-and-Unseen-Cyberattacks-in-Smart-Grids-.git
cd -AI-Based-Intrusion-Detection-and-Classification-of-Known-and-Unseen-Cyberattacks-in-Smart-Grids-

# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate
# Activate environment (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Running on Google Colab

All notebooks in [`phase_1/notebooks/`](phase_1/notebooks/) are built to be self-contained and Colab-ready:
1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and select [`phase_1/notebooks/07_final_comparison.ipynb`](phase_1/notebooks/07_final_comparison.ipynb) (to view the benchmark) or [`phase_1/notebooks/03_binary_classification.ipynb`](phase_1/notebooks/03_binary_classification.ipynb) (to train models live).
3. Upload the required dataset CSV or mount your Google Drive using the first cell.
4. Click **Runtime $\to$ Run all**.

---

## 🔬 Reproducibility & Research Standards
- **Strict Leakage Prevention**: All preprocessors, scalers, and imputers are fitted **strictly on training splits**. No test statistics or future timestamps are snooped.
- **Reproducible Seed**: All experiments use `random_state = 42`.
- **Closed-World vs. Open-World Integrity**: Zero-day attack samples were **strictly excluded** during model training, feature selection, and threshold calibration.
