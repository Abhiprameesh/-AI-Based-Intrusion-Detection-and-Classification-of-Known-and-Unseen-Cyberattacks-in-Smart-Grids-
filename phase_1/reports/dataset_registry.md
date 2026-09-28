# Smart Grid Cybersecurity Dataset Registry

Generated automatically during Phase 1 project directory inspection.

| ID | Dataset Name | Path | Files | Format | Rows | Cols | Target Column | Classes | Temporal | Cyber | Physical | Binary | Multi | Anomaly | Unseen |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DS-01 | **binaryAllNaturalPlusNormalVsAttacks** | `binaryAllNaturalPlusNormalVsAttacks` | 15 | CSV | 78,377 | 129 | `marker` | 2 | No (Tabula | Yes | Yes | Yes | No | Yes | No |
| DS-02 | **PMU_Data_with_Anomalies and Events** | `SGSMA_Competiton 2024_PMU_DATA/PMU_Data_with_Anomalies and Events` | 8 | CSV | 161,379 | 16 | `Event` | 8 | Yes (30 Hz | Yes | Yes | Yes | Yes | Yes | Yes |
| DS-03 | **SGSMA_Competition Day_Testdata** | `SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competition Day_Testdata` | 16 | CSV | 86,222 | 15 | `None (Unlabelled)` | 0 | Yes (30 Hz | No | Yes | No (Unlabelled) | No (Unlabelled) | No | No |
| DS-04 | **CICFlowMeter_Balanced_DNP3_SCADA** | `CICFlowMeter_Training_Balanced.csv & CICFlowMeter_Testing_Balanced.csv` | 2 | CSV | 7,326 | 86 | `Label` | 11 | No (Flow s | Yes | No | Yes | Yes | Yes | Yes |
| DS-05 | **Balanced_IEC104_Train_Test_CSV_Files** | `Balanced_IEC104_Train_Test_CSV_Files/iec104_train_test_csvs` | 24 | CSV | 54,408 | 112 | `Label` | 12 | Windowed f | Yes | No | Yes | Yes | Yes | Yes |
| DS-06 | **data4cyber_dataset** | `data4cyber_dataset` | 8 | CSV, JSON, PCAP | 14,354 | 158 | `attack_active (Binary) / attack_phase (Multiclass)` | 4 | Yes (1.0 H | Yes | Yes | Yes | Yes | Yes | Yes |

## Detailed Registry Information

### DS-01: binaryAllNaturalPlusNormalVsAttacks
- **Path**: `binaryAllNaturalPlusNormalVsAttacks`
- **Files**: data1.csv - data15.csv (15 files)
- **Dimensions**: 78,377 rows × 129 columns
- **Target Column**: `marker`
- **Class Names**: Attack, Natural
- **Temporal Nature**: No (Tabular event snapshots)
- **Cyber Features**: Yes (12 log/alert features: Snort, relay logs, control panel logs)
- **Physical Features**: Yes (116 PMU phasor features across 4 relays: V/I phasors, freq, ROCOF, Z)
- **Train/Test Structure**: 15 individual CSV batches (Stratified 70/15/15 recommended)
- **Supported Tasks**: Binary (Yes), Multiclass (No), Anomaly (Yes (Train Autoencoder / Isolation Forest on Natural baseline)), Unseen Attack (No)

### DS-02: PMU_Data_with_Anomalies and Events
- **Path**: `SGSMA_Competiton 2024_PMU_DATA/PMU_Data_with_Anomalies and Events`
- **Files**: Bus10, Bus19, Bus22, Bus29, Bus2, Bus39, Bus5, Bus6 Competition_Data.csv (8 files)
- **Dimensions**: 161,379 rows × 16 columns
- **Target Column**: `Event`
- **Class Names**: 0: Normal, 1: Fault, 2: Line Outage, 3: Gen Change, 4: Load Change, 5: Missing Data, 6: Missing+Physical, 7: Bad Data
- **Temporal Nature**: Yes (30 Hz continuous synchrophasor time series, ~90 mins)
- **Cyber Features**: Yes (Communication drop events: missing data Event 5 & 6)
- **Physical Features**: Yes (14 PMU channels per bus: 3-phase V/I phasors, frequency, ROCOF)
- **Train/Test Structure**: Continuous time series (Chronological 70% Train, 15% Val, 15% Test)
- **Supported Tasks**: Binary (Yes), Multiclass (Yes), Anomaly (Yes (Train Autoencoder on Event 0 Normal baseline)), Unseen Attack (Yes (Hold-out Event 5, 6, or 7))

### DS-03: SGSMA_Competition Day_Testdata
- **Path**: `SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competition Day_Testdata`
- **Files**: Set 1 (8 bus CSVs), Set 2 (8 bus CSVs) (16 files)
- **Dimensions**: 86,222 rows × 15 columns
- **Target Column**: `None (Unlabelled)`
- **Class Names**: N/A
- **Temporal Nature**: Yes (30 Hz continuous synchrophasor time series)
- **Cyber Features**: Unknown (Unlabelled evaluation traces)
- **Physical Features**: Yes (14 PMU channels per bus)
- **Train/Test Structure**: Official blind competition test traces (No ground truth labels)
- **Supported Tasks**: Binary (No (Unlabelled)), Multiclass (No (Unlabelled)), Anomaly (No (Unlabelled blind evaluation set)), Unseen Attack (No (Unlabelled))

### DS-04: CICFlowMeter_Balanced_DNP3_SCADA
- **Path**: `CICFlowMeter_Training_Balanced.csv & CICFlowMeter_Testing_Balanced.csv`
- **Files**: CICFlowMeter_Training_Balanced.csv, CICFlowMeter_Testing_Balanced.csv (2 files)
- **Dimensions**: 7,326 rows × 86 columns
- **Target Column**: `Label`
- **Class Names**: NORMAL, INIT_DATA, DISABLE_UNSOLICITED, WARM_RESTART, MITM_DOS, ARP_POISONING, DNP3_ENUMERATE, COLD_RESTART, DNP3_INFO, REPLAY, STOP_APP
- **Temporal Nature**: No (Flow summary statistics with timestamp string)
- **Cyber Features**: Yes (80+ network flow features: packet counts, lengths, IAT, TCP flags)
- **Physical Features**: No (Pure cyber network layer)
- **Train/Test Structure**: Official balanced train (5126 rows) and test (2200 rows) split
- **Supported Tasks**: Binary (Yes), Multiclass (Yes), Anomaly (Yes (Train Autoencoder on NORMAL flow baseline)), Unseen Attack (Yes (Hold-out MITM_DOS, REPLAY, etc.))

### DS-05: Balanced_IEC104_Train_Test_CSV_Files
- **Path**: `Balanced_IEC104_Train_Test_CSV_Files/iec104_train_test_csvs`
- **Files**: 12 windowed subfolders (tests_cic_15 to 180, tests_custom_15 to 180) (24 files)
- **Dimensions**: 54,408 rows × 112 columns
- **Target Column**: `Label`
- **Class Names**: NORMAL, c_ci_na_1, c_ci_na_1_DoS, c_rd_na_1, c_rd_na_1_DoS, c_rp_na_1, c_rp_na_1_DoS, c_sc_na_1, c_sc_na_1_DoS, c_se_na_1, c_se_na_1_DoS, m_sp_na_1_DoS
- **Temporal Nature**: Windowed flow aggregations (15s to 180s windows)
- **Cyber Features**: Yes (Network flow + IEC 60870-5-104 ASDU/COT protocol features)
- **Physical Features**: No (Pure industrial SCADA protocol telemetry)
- **Train/Test Structure**: Official train/test splits per aggregation window
- **Supported Tasks**: Binary (Yes), Multiclass (Yes), Anomaly (Yes (Train Autoencoder on NORMAL protocol baseline)), Unseen Attack (Yes (Hold-out specific command injection or DoS types))

### DS-06: data4cyber_dataset
- **Path**: `data4cyber_dataset`
- **Files**: S0_benign_baseline to S6_mqtt_supply_chain_compromise (8 files)
- **Dimensions**: 14,354 rows × 158 columns
- **Target Column**: `attack_active (Binary) / attack_phase (Multiclass)`
- **Class Names**: Families: benign, industroyer_modbus, mitm_false_data_injection, mqtt_price_signal_manipulation; 15 attack phases
- **Temporal Nature**: Yes (1.0 Hz strictly synchronized time-series grid)
- **Cyber Features**: Yes (Modbus variables, MQTT price signal, network taps, reconnaissance phases)
- **Physical Features**: Yes (DER smart grid: PV solar power, battery BSS SoC, AC meters, voltages, frequency)
- **Train/Test Structure**: Scenario-based time series (S0 pure baseline, S1-S6 campaigns)
- **Supported Tasks**: Binary (Yes), Multiclass (Yes), Anomaly (Yes (Train Autoencoder on S0 benign baseline)), Unseen Attack (Yes (Hold-out entire scenario families: e.g. MitM FDI or MQTT price attack))

