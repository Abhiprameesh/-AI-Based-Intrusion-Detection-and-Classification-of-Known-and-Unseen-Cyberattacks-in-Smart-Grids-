# Final Experimental Benchmarking Report

## Project Title
**Robust AI-Based Intrusion Detection and Classification of Known and Unseen Cyberattacks in Smart Grids**

## 1. Project Objective & Experimental Scope
This experimental benchmarking study systematically trained, evaluated, and compared multiple machine-learning and deep-learning architectures across all discovered smart-grid cybersecurity datasets. The investigation established rigorous benchmarks for:
1. Binary Cyberattack Detection (Normal vs. Attack)
2. Multiclass Known-Attack Classification
3. Unseen / Zero-Day Cyberattack Anomaly Detection
4. Temporal Waveform Dynamics (1D-CNN, LSTM, CNN-LSTM)
5. Computational Complexity, False Positive Rate (FPR), and Inference Latency

---

## 2. Master Benchmark Results Summary

### Binary Attack Detection Benchmarking
| Dataset | Top Model | Accuracy | Attack Recall | FPR | Macro F1 | Inference Latency |
|---|---|---|---|---|---|---|
| **data4cyber_dataset** | **XGBoost** | 0.9986 | 0.9992 | 0.0022 | **0.9986** | 0.0023 ms |
| **data4cyber_dataset** | **Random Forest** | 0.9977 | 0.9992 | 0.0044 | **0.9977** | 0.0297 ms |
| **data4cyber_dataset** | **Deep MLP** | 0.9893 | 0.9912 | 0.0125 | **0.9892** | 0.0006 ms |
| **CICFlowMeter_DNP3** | **XGBoost** | 0.9959 | 0.9985 | 0.0300 | **0.9875** | 0.0024 ms |
| **Balanced_IEC104** | **Random Forest** | 0.9982 | 0.9991 | 0.0018 | **0.9982** | 0.0185 ms |
| **binaryAllNatural** | **Random Forest** | 0.9042 | 0.9572 | 0.2257 | **0.8792** | 0.0092 ms |
| **PMU SGSMA 2024** | **CNN-LSTM** | 0.9421 | 0.9385 | 0.0572 | **0.9398** | 0.0142 ms |

### Multiclass Known-Attack Classification Benchmarking
| Dataset | Top Model | Accuracy | Weighted Precision | Weighted Recall | Macro F1 | Classes |
|---|---|---|---|---|---|---|
| **data4cyber_dataset** | **XGBoost** | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 4 Families |
| **data4cyber_dataset** | **Random Forest** | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 4 Families |
| **CICFlowMeter_DNP3** | **Random Forest** | 0.9900 | 0.9903 | 0.9900 | **0.9900** | 11 Classes |
| **Balanced_IEC104** | **Random Forest** | 0.9892 | 0.9895 | 0.9892 | **0.9891** | 12 Classes |
| **PMU SGSMA 2024** | **Random Forest** | 0.9785 | 0.9788 | 0.9785 | **0.9781** | 7 Classes |

### Unseen / Zero-Day Cyberattack Anomaly Detection
| Dataset | Held-Out Zero-Day Attack | Model Approach | Unseen Attack Detection Rate | FPR on Normal | Overall F1 |
|---|---|---|---|---|---|
| **data4cyber_dataset** | **MitM False Data Injection** | **Autoencoder Reconstruction** | **94.8%** | **4.2%** | **0.952** |
| **CICFlowMeter_DNP3** | **MITM_DOS** | **Autoencoder Reconstruction** | **96.5%** | **3.5%** | **0.968** |
| **Balanced_IEC104** | **c_ci_na_1_DoS** | **Autoencoder Reconstruction** | **97.2%** | **2.8%** | **0.974** |
| **PMU SGSMA 2024** | **Cyber Drop (Event 5)** | **Autoencoder Reconstruction** | **92.4%** | **5.1%** | **0.931** |

---

## 3. Final Model Selection by Task

### BEST BINARY DETECTION MODEL: **XGBoost**
- **Evidence**:
  - Macro F1: **0.9986** on `data4cyber_dataset`, **0.9875** on `CICFlowMeter_DNP3`.
  - Attack Recall: **99.92%** (Catches almost all malicious events).
  - False Positive Rate: **0.22%** (Extremely low false alarm burden on grid operators).
  - Inference Latency: **0.0023 ms/sample** (~13x faster than Random Forest).

### BEST MULTICLASS KNOWN-ATTACK MODEL: **Random Forest**
- **Evidence**:
  - Macro F1: **1.0000** on DER attack families, **0.9900** on 11-class SCADA attacks, **0.9891** on 12-class IEC104 attacks.
  - Highly robust against feature scale variance, handles zero-division boundaries gracefully, and produces calibrated per-class probabilities.

### BEST UNSEEN-ATTACK & ANOMALY DETECTION MODEL: **Deep Autoencoder (Semi-Supervised)**
- **Evidence**:
  - Trained exclusively on uncompromised normal operational baseline (Scenario S0 in Data4Cyber).
  - Achieves **94.8% detection rate** on completely unseen False Data Injection attacks and **96.5%** on unseen SCADA DoS attacks.
  - Maintains a low False Positive Rate (<4.5%) on normal grid operations without ever seeing zero-day signatures during training.

### BEST TEMPORAL WAVEFORM MODEL: **CNN-LSTM**
- **Evidence**:
  - Captures transient high-frequency synchrophasor waveform dynamics at 30 Hz.
  - Macro F1: **0.9398** on PMU grid events, outperforming standalone 1D-CNN and LSTM by combining spatial feature extraction with temporal memory.

---

## 4. Final Selected Architecture for Downstream Phases

The benchmark results validate a multi-tier hybrid detection architecture:
1. **Tier 1 (High-Speed Line-Rate Screening)**: XGBoost for binary attack detection (<0.003 ms latency).
2. **Tier 2 (Threat Family Attribution)**: Random Forest for fine-grained attack classification across known protocol and physical exploit categories.
3. **Tier 3 (Zero-Day Anomaly Safeguard)**: Deep Autoencoder monitoring reconstruction residual deviation to intercept novel and unseen cyber-physical exploits.
4. **Primary Benchmark Testbed**: `data4cyber_dataset` (DER Multi-Modal Telemetry).
