# Smart Grid Cybersecurity Initial Data Quality Report

## Executive Summary
This report documents the exhaustive data audit conducted across all discovered smart-grid datasets in the workspace. The audit evaluates shape, data integrity, class balance, temporal structure, domain features (cyber vs. physical), missing values, duplicates, infinity values, and critical data leakage vectors.

---

## 1. Dataset 1: `binaryAllNaturalPlusNormalVsAttacks`

- **Exact Directory**: `binaryAllNaturalPlusNormalVsAttacks/`
- **Origin**: Mississippi State University / Oak Ridge National Laboratory (MSU/ORNL) Power System Cyber Attack Dataset.
- **Files**: 15 CSV files (`data1.csv` through `data15.csv`).
- **Total Records**: 78,377 rows across all 15 files.
- **Dimensionality**: 129 columns.
- **Target Variable**: `marker` (Binary).
  - `Attack`: 55,663 samples (71.02%)
  - `Natural`: 22,714 samples (28.98%)
  - Class Ratio: ~2.45:1 (Moderate imbalance).
- **Physical Features (116 columns)**:
  - Measurements from 4 Intelligent Electronic Devices (IEDs) / Protective Relays (R1, R2, R3, R4) monitoring transmission lines.
  - Per relay: 3-phase Voltage Phase Angles (`PA1:VH`, `PA2:VH`, `PA3:VH`), Magnitudes (`PM1:V`, `PM2:V`, `PM3:V`), Current Phase Angles (`PA4:IH`, `PA5:IH`, `PA6:IH`), Current Magnitudes (`PM4:I`, `PM5:I`, `PM6:I`), Zero/Negative sequence components (`PA7-PA12`), Frequency (`F`), Frequency derivative/ROCOF (`DF`), Apparent Impedance (`PA:Z`, `PA:ZH`), and Relay Status (`S`).
- **Cyber Features (12 columns)**:
  - System logs: `control_panel_log1` - `control_panel_log4`, `relay1_log` - `relay4_log`, and Snort IDS alerts: `snort_log1` - `snort_log4`.
- **Data Quality Findings**:
  - Missing Values (NaN): 0.
  - Exact Duplicates: 8 rows across all 15 files.
  - **Infinity Values (`inf`)**: **10,906 infinity occurrences** detected across apparent impedance columns (`R1-PA:Z`, `R2-PA:Z`, `R3-PA:Z`, `R4-PA:Z`). These occur when current approaches zero during line trips/faults ($Z = V/I$). Must be replaced with NaN/capped during preprocessing.
  - Constant Features: `R1:S`, `R2:S`, `R3:S`, `R4:S` have variance 0 in several files.
- **Leakage Prevention**: No direct ID or label-leaking columns; cyber alarm logs (`snort_log*`) correlate strongly with attack state and must be evaluated both with and without cyber logs.
- **Supported Tasks**:
  - Binary Classification: Yes (`Natural` vs `Attack`).
  - Multiclass Classification: No (attack subtypes are collapsed into `Attack`).
  - Anomaly Detection: Yes (Autoencoder / One-Class SVM trained on `Natural`).
  - Temporal Models: Not recommended (data is stored as static event snapshots rather than uniform time series).

---

## 2. Dataset 2: `PMU_Data_with_Anomalies and Events` (IEEE SGSMA 2024)

- **Exact Directory**: `SGSMA_Competiton 2024_PMU_DATA/PMU_Data_with_Anomalies and Events/`
- **Origin**: IEEE SGSMA 2024 Competition - Realistic Labelled PMU Data for Cyber-Power Anomaly Detection Using Real-Time Synchrophasor Testbed.
- **Physical Testbed**: IEEE 39-bus New England power system with 8 Phasor Measurement Units (PMUs).
- **Files**: 8 bus CSV files: `Bus10`, `Bus19`, `Bus22`, `Bus29`, `Bus2`, `Bus39`, `Bus5`, `Bus6_Competition_Data.csv`.
- **Total Records**: 161,379 time steps per bus (approx. 90 minutes duration) = 1,291,032 total measurements.
- **Sampling Rate**: Continuous 30 frames/second (reporting interval $\Delta t \approx 0.033\text{ s}$).
- **Dimensionality**: 16 columns per bus file:
  - `TIMESTAMP`
  - 3-phase Voltage Phasor Angles: `VA_ANG`, `VB_ANG`, `VC_ANG`
  - 3-phase Voltage Phasor Magnitudes: `VA_MAG`, `VB_MAG`, `VC_MAG`
  - 3-phase Current Phasor Angles: `IA_ANG`, `IB_ANG`, `IC_ANG`
  - 3-phase Current Phasor Magnitudes: `IA_MAG`, `IB_MAG`, `IC_MAG`
  - Grid Frequency: `Freq`
  - Rate of Change of Frequency: `ROCOF`
  - Label: `Event`
- **Target Variable**: `Event` (Multiclass: 8 distinct classes):
  - `0`: Normal operation (~133,798 to 142,444 samples, ~83-88%)
  - `1`: Fault (149 samples, ~0.09%)
  - `2`: Line outage (126 samples, ~0.08%)
  - `3`: Generation change/outage (8,974 - 9,528 samples, ~5.5-5.9%)
  - `4`: Load change/drop (9,132 samples, ~5.6%)
  - `5`: Missing data / Cyber communication link failure (8,646 samples at Bus 29, ~5.3%)
  - `6`: Missing data with physical event / Concurrent Cyber-Power event (554 samples at Bus 29, ~0.34%)
  - `7`: Bad data (22 samples at Bus 2, 55 samples at Bus 39)
- **Data Quality Findings**:
  - Missing Values (NaN): 0.
  - Duplicate Rows: 0.
  - Infinity Values: 0.
  - Class Imbalance: Severe for rare transient faults (Event 1: 149 samples; Event 2: 126 samples).
- **Leakage Prevention**: `TIMESTAMP` must not be used as a numerical feature. Chronological splitting (Past $\to$ Train, Later $\to$ Val, Future $\to$ Test) is strictly required to prevent temporal leakage.
- **Supported Tasks**:
  - Binary Classification: Yes (Normal 0 vs Event >0).
  - Multiclass Classification: Yes (8 classes).
  - Temporal / Sequential Models: **Yes (Ideal for 1D-CNN, LSTM, CNN-LSTM due to 30 Hz continuous synchrophasor waveform dynamics)**.
  - Anomaly Detection: Yes (Train Autoencoder on Event 0).
  - Unseen Attack Detection: **Yes (Hold out Event 5 cyber drop or Event 6 concurrent cyber-physical attack as unseen anomalies)**.

---

## 3. Dataset 3: `SGSMA_Competition Day_Testdata`

- **Exact Directory**: `SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competition Day_Testdata/`
- **Subdirectories**: `Competition_Testing Data Set 1/` (8 bus CSVs, 43,472 rows), `Competition_Testing Data Set 2/` (8 bus CSVs, 42,750 rows).
- **Dimensionality**: 15 columns (`TIMESTAMP` + 14 PMU channels).
- **Ground Truth Labels**: **NONE (Unlabelled)**. The `Event` column is entirely absent.
- **Critical Policy Ruling**: These files are competition blind test traces without answer keys. In strict accordance with Project Rule 19 ("Do not use competition test data for training") and Rule 7 ("NEVER invent labels"), **these files are excluded from model training and internal validation**.

---

## 4. Dataset 4: `CICFlowMeter_Balanced_DNP3_SCADA`

- **Exact Files**: `CICFlowMeter_Training_Balanced.csv` and `CICFlowMeter_Testing_Balanced.csv` (Workspace Root).
- **Protocol & Context**: DNP3 / SCADA network traffic captured and processed via CICFlowMeter into bidirectional network flow statistics.
- **Records**: Training: 5,126 rows; Testing: 2,200 rows (Total: 7,326 rows).
- **Dimensionality**: 86 columns.
- **Target Variable**: `Label` (11 perfectly balanced classes):
  - `NORMAL`: 466 train / 200 test
  - `INIT_DATA`: 466 train / 200 test
  - `DISABLE_UNSOLICITED`: 466 train / 200 test
  - `WARM_RESTART`: 466 train / 200 test
  - `MITM_DOS`: 466 train / 200 test
  - `ARP_POISONING`: 466 train / 200 test
  - `DNP3_ENUMERATE`: 466 train / 200 test
  - `COLD_RESTART`: 466 train / 200 test
  - `DNP3_INFO`: 466 train / 200 test
  - `REPLAY`: 466 train / 200 test
  - `STOP_APP`: 466 train / 200 test
- **Cyber vs. Physical**: 100% Cyber OT/SCADA network layer. 0 Physical grid measurements.
- **Data Quality Findings**:
  - Missing Values: 0.
  - Infinity Values: 2 `inf` occurrences in `Flow Byts/s` / `Flow Pkts/s` in training set due to zero flow duration.
  - Duplicates: 340 duplicated rows in train, 99 in test (flows with identical packet metrics).
- **Critical Data Leakage Vectors**:
  - Columns `Unnamed: 0`, `Flow ID`, `Src IP`, `Dst IP`, `Src Port`, `Dst Port`, `Timestamp` **directly leak flow identities and IP addressing**. These must be removed from the feature set prior to training.
- **Supported Tasks**:
  - Binary Classification: Yes (`NORMAL` vs Attacks).
  - Multiclass Classification: Yes (11 classes).
  - Unseen Attack Detection: **Yes (Hold out `MITM_DOS`, `REPLAY`, or `ARP_POISONING`)**.
  - Anomaly Detection: Yes (Train on `NORMAL`).

---

## 5. Dataset 5: `Balanced_IEC104_Train_Test_CSV_Files`

- **Exact Directory**: `Balanced_IEC104_Train_Test_CSV_Files/iec104_train_test_csvs/`
- **Subdirectories**: 12 experimental window configurations:
  - CICFlowMeter features: `tests_cic_15`, `tests_cic_30`, `tests_cic_60`, `tests_cic_90`, `tests_cic_120`, `tests_cic_180`.
  - Custom IEC 104 Script features: `tests_custom_15`, `tests_custom_30`, `tests_custom_60`, `tests_custom_90`, `tests_custom_120`, `tests_custom_180`.
- **Total Records**: 54,408 across all windows.
- **Dimensionality**: CICFlowMeter: 84 columns; Custom Script: 112 columns.
- **Protocol Features**: Deep IEC 60870-5-104 SCADA protocol semantics:
  - Cause of Transmission (`cot=7` through `cot=20`).
  - ASDU Type IDs (`type_id_process_information_in_monitor_direction`, `type_id_process_information_in_control_direction`, etc.).
- **Target Variable**: `Label` (12 balanced classes per window):
  - `NORMAL`, `c_ci_na_1`, `c_ci_na_1_DoS`, `c_rd_na_1`, `c_rd_na_1_DoS`, `c_rp_na_1`, `c_rp_na_1_DoS`, `c_sc_na_1`, `c_sc_na_1_DoS`, `c_se_na_1`, `c_se_na_1_DoS`, `m_sp_na_1_DoS`.
- **Supported Tasks**:
  - Binary Classification: Yes (`NORMAL` vs 11 IEC 104 attack classes).
  - Multiclass Classification: Yes (12 classes).
  - Unseen Attack Detection: Yes (Hold out specific ASDU command injections or DoS variants).

---

## 6. Dataset 6: `data4cyber_dataset`

- **Exact Directory**: `data4cyber_dataset/`
- **Origin**: Cyber-physical security testbed for Distributed Energy Resource (DER) distribution management systems.
- **Scenarios**: 8 scenario runs (`S0` to `S6` + `S1_alt`):
  - `S0_benign_baseline` (Benign operation: 1,798 rows)
  - `S1_industroyer_pv` (Industroyer Modbus PV tampering: 1,798 rows)
  - `S1_industroyer_pv_alt` (Industroyer Modbus alternate: 1,792 rows)
  - `S2_industroyer_bss` (Industroyer Modbus Battery tampering: 1,768 rows)
  - `S3_arp_spoof_bss_meter_half_values` (MitM False Data Injection on meter: 1,800 rows)
  - `S4_arp_spoof_loads_pv_two_phase` (MitM FDI 2-phase PV/load falsification: 1,798 rows)
  - `S5_arp_spoof_loads_pv_bss_two_phase` (Extended MitM FDI: 1,800 rows)
  - `S6_mqtt_supply_chain_compromise` (MQTT price signal manipulation: 1,800 rows)
- **Total Records**: 14,354 rows across all 8 scenarios (Primary set S0-S6: 12,562 rows).
- **Time Grid**: Aligned 1.0-second time series grid.
- **Dimensionality**: 154 to 158 columns per scenario.
- **Cyber-Physical Multi-Modal Telemetry**:
  - Physical layer: AC meter active/reactive power (`*.p_sum3`), current (`*.i_sum3`), voltage, frequency, battery SoC reference vs reported.
  - Cyber layer: Modbus protocol variables, MQTT price signals, IP/MAC telemetry.
- **Labels**:
  - Binary: `attack_active` (5,880 Benign / 8,474 Attack).
  - Multiclass: `attack_phase` (15 phase tokens) and `attack_family` (4 families: `benign`, `industroyer_modbus`, `mitm_false_data_injection`, `mqtt_price_signal_manipulation`).
- **Data Quality & Leakage Findings**:
  - Missing Values: 0.
  - Duplicates: 0.
  - **Severe Data Leakage Risk**: Scenarios S1-S5 contain columns `Attacker.event`, `Attacker.old_value`, and `Attacker.new_value`. As documented in the README, these are attacker event ground-truth logs. **They must be stripped immediately before training to prevent 100% artificial label leakage**.
- **Supported Tasks**:
  - Binary Classification: Yes.
  - Multiclass Classification: Yes (Attack Family: 4 classes; Attack Phase: 15 classes).
  - Anomaly Detection: **Superior (S0 is an uncontaminated benign operational baseline)**.
  - Unseen Attack Detection: **State-of-the-Art (Train on S0 + S1-S2 Modbus, test generalization on unseen S3-S5 MitM FDI or S6 MQTT supply-chain attack)**.
