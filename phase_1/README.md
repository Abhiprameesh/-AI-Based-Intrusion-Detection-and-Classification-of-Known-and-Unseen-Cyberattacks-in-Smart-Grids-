# Phase 1: Multi-Dataset Experimental Benchmarking Suite

> **Project:** Robust AI-Based Intrusion Detection and Classification of Known and Unseen Cyberattacks in Smart Grids  
> **Phase Objective:** Systematically evaluate, benchmark, and compare Machine Learning and Deep Learning architectures across all candidate smart-grid datasets to establish the definitive experimental baseline and dataset selection for downstream defense stages.

---

## 1. Directory Structure

```text
phase_1/
├── experiment_config.yaml          # Standardized hyperparameters, seeds, and split ratios
├── models/                         # Model registry directory (weights gitignored)
│   └── .gitkeep
├── notebooks/                      # 7 Interactive, Colab-ready Jupyter Notebooks
│   ├── 01_dataset_discovery.ipynb
│   ├── 02_dataset_analysis.ipynb
│   ├── 03_binary_classification.ipynb
│   ├── 04_multiclass_classification.ipynb
│   ├── 05_anomaly_detection.ipynb
│   ├── 06_unseen_attack_detection.ipynb
│   └── 07_final_comparison.ipynb
├── reports/                        # Decision documents, summaries, and plots
│   ├── data_quality_report.md
│   ├── dataset_audit_raw.json
│   ├── dataset_comparison.csv
│   ├── dataset_registry.csv
│   ├── dataset_registry.md
│   ├── feature_selection_summary.json
│   ├── final_dataset_selection.md
│   ├── final_experimental_report.md
│   ├── final_model_comparison.csv
│   ├── hyperparameter_tuning_summary.json
│   └── plots/                      # Publication-grade figures across all datasets
├── results/                        # JSON metrics & comparison tables
│   ├── Balanced_IEC104_Train_Test_CSV_Files/
│   ├── binaryAllNaturalPlusNormalVsAttacks/
│   ├── CICFlowMeter_Balanced_DNP3_SCADA/
│   ├── data4cyber_dataset/
│   └── PMU_Data_with_Anomalies_and_Events/
└── src/                            # Production modular Python engine
    ├── __init__.py
    ├── anomaly_detection.py
    ├── create_notebooks.py
    ├── data_loader.py
    ├── dataset_audit.py
    ├── evaluate.py
    ├── execute_experimental_suite.py
    ├── experiment_tracker.py
    ├── feature_selection.py
    ├── generate_registry_and_reports.py
    ├── preprocessing.py
    ├── run_all_experiments.py
    ├── train_classical.py
    ├── train_deep_learning.py
    ├── unseen_attack.py
    ├── utils.py
    └── visualization.py
```

---

## 2. Executive Benchmarking Conclusions

### Question 1: Which dataset is most suitable for our final Smart Grid IDS?
- **Primary Selection:** **`data4cyber_dataset` (DER Multi-Modal Telemetry)**
  - Only testbed capturing multi-modal cyber-physical telemetry: PV power, Battery Energy Storage (BESS) SoC, inverter currents, Modbus registers, MQTT price broadcasts, and raw PCAP captures.
  - Features a pristine baseline scenario (`S0`), 4 structured attack campaigns (DoS, Replay, MitM FDI, Malicious Commands), and zero missing values.
- **Secondary Selection:** **`PMU_Data_with_Anomalies_and_Events` (IEEE SGSMA 2024 PMU)**
  - Provides high-frequency (30 Hz) synchrophasor waveform dynamics across the New England 39-bus system for sub-second transient anomaly and temporal classification.

### Question 2: Which model performs best for binary attack detection?
- **Winner:** **`XGBoost`**
  - **Macro F1:** **0.9986** | **Attack Recall:** **99.92%** | **FPR:** **0.22%** | **Inference Latency:** **0.0023 ms/sample**

### Question 3: Which model performs best for multiclass known-attack classification?
- **Winner:** **`Random Forest`**
  - **Macro F1:** **1.0000** (Data4Cyber campaigns), **0.9900** (MSU/ORNL SCADA attacks)
  - Exceptional robustness against feature collinearity without requiring heavy GPU acceleration.

### Question 4: Which model is most suitable for unseen / zero-day attack detection?
- **Winner:** **`Deep Autoencoder (PyTorch)`**
  - Trained purely on pristine normal operations (`S0` in Data4Cyber; Natural in MSU/ORNL).
  - Achieved **94.8% detection rate** on held-out MitM False Data Injection attacks and **96.5% detection rate** on unseen SCADA DoS with False Positive Rate < 4.5%.

### Question 5: Which model provides the optimal operational trade-off?
- **Winner:** **`XGBoost`**
  - Sub-microsecond latency (0.0023 ms/sample), ultra-low false positive rate (0.22%), zero deep learning warm-up overhead, and native explainability (SHAP / Tree Feature Importances).

### Question 6: Best temporal sequence model for PMU waveform dynamics?
- **Winner:** **`CNN-LSTM`**
  - **Macro F1:** **0.9398** on 30 Hz synchrophasor sequences, capturing local transient frequency fluctuations via 1D convolutions and long-range inter-area oscillations via recurrent LSTM memory.

---

## 3. Interactive Notebook Execution

Each notebook in `phase_1/notebooks/` is self-contained with automatic virtual environment and Google Colab initialization blocks:

| Notebook | Focus Area |
| :--- | :--- |
| [`01_dataset_discovery.ipynb`](file:///c:/ALLMLPROJ/MAINMLPROJS/MAJOR%20PROJECT/phase_1/notebooks/01_dataset_discovery.ipynb) | Automated directory scanning, integrity checks, and schema validation |
| [`02_dataset_analysis.ipynb`](file:///c:/ALLMLPROJ/MAINMLPROJS/MAJOR%20PROJECT/phase_1/notebooks/02_dataset_analysis.ipynb) | Exploratory data analysis, class distributions, and impedance/inf cleaning |
| [`03_binary_classification.ipynb`](file:///c:/ALLMLPROJ/MAINMLPROJS/MAJOR%20PROJECT/phase_1/notebooks/03_binary_classification.ipynb) | Classical ML + Tabular MLP binary benchmarking across all datasets |
| [`04_multiclass_classification.ipynb`](file:///c:/ALLMLPROJ/MAINMLPROJS/MAJOR%20PROJECT/phase_1/notebooks/04_multiclass_classification.ipynb) | Attack family attribution and confusion matrix generation |
| [`05_anomaly_detection.ipynb`](file:///c:/ALLMLPROJ/MAINMLPROJS/MAJOR%20PROJECT/phase_1/notebooks/05_anomaly_detection.ipynb) | Unsupervised baseline training (Autoencoders, Isolation Forest) |
| [`06_unseen_attack_detection.ipynb`](file:///c:/ALLMLPROJ/MAINMLPROJS/MAJOR%20PROJECT/phase_1/notebooks/06_unseen_attack_detection.ipynb) | Zero-day attack evaluation on strictly held-out cyber campaigns |
| [`07_final_comparison.ipynb`](file:///c:/ALLMLPROJ/MAINMLPROJS/MAJOR%20PROJECT/phase_1/notebooks/07_final_comparison.ipynb) | Multi-criteria radar chart analysis, latency Pareto frontier, and decision report |
