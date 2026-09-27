import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

WORKSPACE = r"c:\ALLMLPROJ\MAINMLPROJS\MAJOR PROJECT"
REPORTS_DIR = os.path.join(WORKSPACE, "reports")
PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.size': 11})

# Load raw audit data
with open(os.path.join(REPORTS_DIR, "dataset_audit_raw.json"), 'r') as f:
    raw_data = json.load(f)

# Build Registry Dataframe
registry_rows = [
    {
        "Dataset_ID": "DS-01",
        "Dataset_Name": "binaryAllNaturalPlusNormalVsAttacks",
        "Dataset_Path": "binaryAllNaturalPlusNormalVsAttacks",
        "File_Count": 15,
        "File_Names": "data1.csv - data15.csv",
        "File_Format": "CSV",
        "Rows": 78377,
        "Columns": 129,
        "Target_Column": "marker",
        "Number_of_Classes": 2,
        "Class_Names": "Attack, Natural",
        "Temporal_Data": "No (Tabular event snapshots)",
        "Cyber_Features": "Yes (12 log/alert features: Snort, relay logs, control panel logs)",
        "Physical_Features": "Yes (116 PMU phasor features across 4 relays: V/I phasors, freq, ROCOF, Z)",
        "Train_Test_Structure": "15 individual CSV batches (Stratified 70/15/15 recommended)",
        "Binary_Task_Available": "Yes",
        "Multiclass_Task_Available": "No",
        "Anomaly_Task_Available": "Yes (Train Autoencoder / Isolation Forest on Natural baseline)",
        "Unseen_Attack_Task_Available": "No"
    },
    {
        "Dataset_ID": "DS-02",
        "Dataset_Name": "PMU_Data_with_Anomalies and Events",
        "Dataset_Path": "SGSMA_Competiton 2024_PMU_DATA/PMU_Data_with_Anomalies and Events",
        "File_Count": 8,
        "File_Names": "Bus10, Bus19, Bus22, Bus29, Bus2, Bus39, Bus5, Bus6 Competition_Data.csv",
        "File_Format": "CSV",
        "Rows": 161379,
        "Columns": 16,
        "Target_Column": "Event",
        "Number_of_Classes": 8,
        "Class_Names": "0: Normal, 1: Fault, 2: Line Outage, 3: Gen Change, 4: Load Change, 5: Missing Data, 6: Missing+Physical, 7: Bad Data",
        "Temporal_Data": "Yes (30 Hz continuous synchrophasor time series, ~90 mins)",
        "Cyber_Features": "Yes (Communication drop events: missing data Event 5 & 6)",
        "Physical_Features": "Yes (14 PMU channels per bus: 3-phase V/I phasors, frequency, ROCOF)",
        "Train_Test_Structure": "Continuous time series (Chronological 70% Train, 15% Val, 15% Test)",
        "Binary_Task_Available": "Yes",
        "Multiclass_Task_Available": "Yes",
        "Anomaly_Task_Available": "Yes (Train Autoencoder on Event 0 Normal baseline)",
        "Unseen_Attack_Task_Available": "Yes (Hold-out Event 5, 6, or 7)"
    },
    {
        "Dataset_ID": "DS-03",
        "Dataset_Name": "SGSMA_Competition Day_Testdata",
        "Dataset_Path": "SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competition Day_Testdata",
        "File_Count": 16,
        "File_Names": "Set 1 (8 bus CSVs), Set 2 (8 bus CSVs)",
        "File_Format": "CSV",
        "Rows": 86222,
        "Columns": 15,
        "Target_Column": "None (Unlabelled)",
        "Number_of_Classes": 0,
        "Class_Names": "N/A",
        "Temporal_Data": "Yes (30 Hz continuous synchrophasor time series)",
        "Cyber_Features": "Unknown (Unlabelled evaluation traces)",
        "Physical_Features": "Yes (14 PMU channels per bus)",
        "Train_Test_Structure": "Official blind competition test traces (No ground truth labels)",
        "Binary_Task_Available": "No (Unlabelled)",
        "Multiclass_Task_Available": "No (Unlabelled)",
        "Anomaly_Task_Available": "No (Unlabelled blind evaluation set)",
        "Unseen_Attack_Task_Available": "No (Unlabelled)"
    },
    {
        "Dataset_ID": "DS-04",
        "Dataset_Name": "CICFlowMeter_Balanced_DNP3_SCADA",
        "Dataset_Path": "CICFlowMeter_Training_Balanced.csv & CICFlowMeter_Testing_Balanced.csv",
        "File_Count": 2,
        "File_Names": "CICFlowMeter_Training_Balanced.csv, CICFlowMeter_Testing_Balanced.csv",
        "File_Format": "CSV",
        "Rows": 7326,
        "Columns": 86,
        "Target_Column": "Label",
        "Number_of_Classes": 11,
        "Class_Names": "NORMAL, INIT_DATA, DISABLE_UNSOLICITED, WARM_RESTART, MITM_DOS, ARP_POISONING, DNP3_ENUMERATE, COLD_RESTART, DNP3_INFO, REPLAY, STOP_APP",
        "Temporal_Data": "No (Flow summary statistics with timestamp string)",
        "Cyber_Features": "Yes (80+ network flow features: packet counts, lengths, IAT, TCP flags)",
        "Physical_Features": "No (Pure cyber network layer)",
        "Train_Test_Structure": "Official balanced train (5126 rows) and test (2200 rows) split",
        "Binary_Task_Available": "Yes",
        "Multiclass_Task_Available": "Yes",
        "Anomaly_Task_Available": "Yes (Train Autoencoder on NORMAL flow baseline)",
        "Unseen_Attack_Task_Available": "Yes (Hold-out MITM_DOS, REPLAY, etc.)"
    },
    {
        "Dataset_ID": "DS-05",
        "Dataset_Name": "Balanced_IEC104_Train_Test_CSV_Files",
        "Dataset_Path": "Balanced_IEC104_Train_Test_CSV_Files/iec104_train_test_csvs",
        "File_Count": 24,
        "File_Names": "12 windowed subfolders (tests_cic_15 to 180, tests_custom_15 to 180)",
        "File_Format": "CSV",
        "Rows": 54408,
        "Columns": 112,
        "Target_Column": "Label",
        "Number_of_Classes": 12,
        "Class_Names": "NORMAL, c_ci_na_1, c_ci_na_1_DoS, c_rd_na_1, c_rd_na_1_DoS, c_rp_na_1, c_rp_na_1_DoS, c_sc_na_1, c_sc_na_1_DoS, c_se_na_1, c_se_na_1_DoS, m_sp_na_1_DoS",
        "Temporal_Data": "Windowed flow aggregations (15s to 180s windows)",
        "Cyber_Features": "Yes (Network flow + IEC 60870-5-104 ASDU/COT protocol features)",
        "Physical_Features": "No (Pure industrial SCADA protocol telemetry)",
        "Train_Test_Structure": "Official train/test splits per aggregation window",
        "Binary_Task_Available": "Yes",
        "Multiclass_Task_Available": "Yes",
        "Anomaly_Task_Available": "Yes (Train Autoencoder on NORMAL protocol baseline)",
        "Unseen_Attack_Task_Available": "Yes (Hold-out specific command injection or DoS types)"
    },
    {
        "Dataset_ID": "DS-06",
        "Dataset_Name": "data4cyber_dataset",
        "Dataset_Path": "data4cyber_dataset",
        "File_Count": 8,
        "File_Names": "S0_benign_baseline to S6_mqtt_supply_chain_compromise",
        "File_Format": "CSV, JSON, PCAP",
        "Rows": 14354,
        "Columns": 158,
        "Target_Column": "attack_active (Binary) / attack_phase (Multiclass)",
        "Number_of_Classes": 4,
        "Class_Names": "Families: benign, industroyer_modbus, mitm_false_data_injection, mqtt_price_signal_manipulation; 15 attack phases",
        "Temporal_Data": "Yes (1.0 Hz strictly synchronized time-series grid)",
        "Cyber_Features": "Yes (Modbus variables, MQTT price signal, network taps, reconnaissance phases)",
        "Physical_Features": "Yes (DER smart grid: PV solar power, battery BSS SoC, AC meters, voltages, frequency)",
        "Train_Test_Structure": "Scenario-based time series (S0 pure baseline, S1-S6 campaigns)",
        "Binary_Task_Available": "Yes",
        "Multiclass_Task_Available": "Yes",
        "Anomaly_Task_Available": "Yes (Train Autoencoder on S0 benign baseline)",
        "Unseen_Attack_Task_Available": "Yes (Hold-out entire scenario families: e.g. MitM FDI or MQTT price attack)"
    }
]

df_registry = pd.DataFrame(registry_rows)
# Ensure exact requested column order
cols_order = [
    "Dataset_ID", "Dataset_Name", "Dataset_Path", "File_Count", "File_Names", "File_Format",
    "Rows", "Columns", "Target_Column", "Number_of_Classes", "Class_Names", "Temporal_Data",
    "Cyber_Features", "Physical_Features", "Train_Test_Structure", "Binary_Task_Available",
    "Multiclass_Task_Available", "Anomaly_Task_Available", "Unseen_Attack_Task_Available"
]
df_registry = df_registry[cols_order]
csv_registry_path = os.path.join(REPORTS_DIR, "dataset_registry.csv")
df_registry.to_csv(csv_registry_path, index=False)
print(f"Saved {csv_registry_path}")

# Build Markdown Registry
md_registry_path = os.path.join(REPORTS_DIR, "dataset_registry.md")
with open(md_registry_path, "w") as f:
    f.write("# Smart Grid Cybersecurity Dataset Registry\n\n")
    f.write("Generated automatically during Phase 1 project directory inspection.\n\n")
    f.write("| ID | Dataset Name | Path | Files | Format | Rows | Cols | Target Column | Classes | Temporal | Cyber | Physical | Binary | Multi | Anomaly | Unseen |\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in registry_rows:
        f.write(f"| {r['Dataset_ID']} | **{r['Dataset_Name']}** | `{r['Dataset_Path']}` | {r['File_Count']} | {r['File_Format']} | {r['Rows']:,} | {r['Columns']} | `{r['Target_Column']}` | {r['Number_of_Classes']} | {r['Temporal_Data'][:10]} | {'Yes' if 'Yes' in r['Cyber_Features'] else 'No'} | {'Yes' if 'Yes' in r['Physical_Features'] else 'No'} | {r['Binary_Task_Available']} | {r['Multiclass_Task_Available']} | {r['Anomaly_Task_Available'].split()[0]} | {r['Unseen_Attack_Task_Available'].split()[0]} |\n")
    f.write("\n## Detailed Registry Information\n\n")
    for r in registry_rows:
        f.write(f"### {r['Dataset_ID']}: {r['Dataset_Name']}\n")
        f.write(f"- **Path**: `{r['Dataset_Path']}`\n")
        f.write(f"- **Files**: {r['File_Names']} ({r['File_Count']} files)\n")
        f.write(f"- **Dimensions**: {r['Rows']:,} rows × {r['Columns']} columns\n")
        f.write(f"- **Target Column**: `{r['Target_Column']}`\n")
        f.write(f"- **Class Names**: {r['Class_Names']}\n")
        f.write(f"- **Temporal Nature**: {r['Temporal_Data']}\n")
        f.write(f"- **Cyber Features**: {r['Cyber_Features']}\n")
        f.write(f"- **Physical Features**: {r['Physical_Features']}\n")
        f.write(f"- **Train/Test Structure**: {r['Train_Test_Structure']}\n")
        f.write(f"- **Supported Tasks**: Binary ({r['Binary_Task_Available']}), Multiclass ({r['Multiclass_Task_Available']}), Anomaly ({r['Anomaly_Task_Available']}), Unseen Attack ({r['Unseen_Attack_Task_Available']})\n\n")

print(f"Saved {md_registry_path}")

# Generate Plots for Each Dataset
print("\nGenerating audit plots...")

# 1. binaryAllNaturalPlusNormalVsAttacks Plot
p1_dir = os.path.join(PLOTS_DIR, "binaryAllNaturalPlusNormalVsAttacks")
os.makedirs(p1_dir, exist_ok=True)
plt.figure(figsize=(7, 4.5))
classes_b = raw_data['binaryAllNaturalPlusNormalVsAttacks']['classes']
bars = plt.bar(classes_b.keys(), classes_b.values(), color=['#d9534f', '#5cb85c'], width=0.5)
plt.title("binaryAllNaturalPlusNormalVsAttacks: Class Distribution", fontweight='bold')
plt.ylabel("Number of Samples")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1000, f'{yval:,} ({yval/sum(classes_b.values()):.1%})', ha='center', va='bottom', fontsize=10)
plt.ylim(0, max(classes_b.values()) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(p1_dir, "class_distribution.png"), dpi=300)
plt.close()

# 2. PMU_Data_with_Anomalies and Events Plot
p2_dir = os.path.join(PLOTS_DIR, "PMU_Data_with_Anomalies_and_Events")
os.makedirs(p2_dir, exist_ok=True)
pmu_events = {
    '0: Normal': 133798,
    '1: Fault': 149,
    '2: Line Outage': 126,
    '3: Gen Change': 8974,
    '4: Load Change': 9132,
    '5: Missing (Cyber)': 8646,
    '6: Missing+Phys': 554,
    '7: Bad Data (B2/B39)': 77
}
plt.figure(figsize=(10, 5))
colors = ['#428bca', '#d9534f', '#f0ad4e', '#5bc0de', '#5cb85c', '#9370db', '#e75480', '#f4a460']
bars = plt.bar(pmu_events.keys(), pmu_events.values(), color=colors)
plt.yscale('log')
plt.title("PMU_Data_with_Anomalies and Events (Bus 29): Class Distribution (Log Scale)", fontweight='bold')
plt.ylabel("Sample Count (Log Scale)")
plt.xticks(rotation=30, ha='right')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval * 1.15, f'{yval:,}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(p2_dir, "class_distribution_log.png"), dpi=300)
plt.close()

# 3. CICFlowMeter_Balanced_DNP3_SCADA Plot
p3_dir = os.path.join(PLOTS_DIR, "CICFlowMeter_Balanced_DNP3_SCADA")
os.makedirs(p3_dir, exist_ok=True)
train_c = raw_data['Balanced_IEC104_Train_Test_CSV_Files']['root_train']['classes']
plt.figure(figsize=(11, 5))
bars = plt.bar(train_c.keys(), train_c.values(), color='#337ab7')
plt.title("CICFlowMeter Training Dataset: Balanced Class Distribution", fontweight='bold')
plt.ylabel("Number of Samples")
plt.xticks(rotation=40, ha='right')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f'{yval}', ha='center', va='bottom', fontsize=9)
plt.ylim(0, max(train_c.values()) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(p3_dir, "class_distribution.png"), dpi=300)
plt.close()

# 4. data4cyber_dataset Plot
p4_dir = os.path.join(PLOTS_DIR, "data4cyber_dataset")
os.makedirs(p4_dir, exist_ok=True)
fams = raw_data['data4cyber_dataset']['families']
plt.figure(figsize=(8, 4.5))
bars = plt.bar(fams.keys(), fams.values(), color=['#5cb85c', '#d9534f', '#f0ad4e', '#5bc0de'])
plt.title("data4cyber_dataset: Attack Family Distribution", fontweight='bold')
plt.ylabel("Number of Rows (1.0s steps)")
plt.xticks(rotation=20, ha='right')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 100, f'{yval:,} ({yval/sum(fams.values()):.1%})', ha='center', va='bottom', fontsize=9)
plt.ylim(0, max(fams.values()) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(p4_dir, "family_distribution.png"), dpi=300)
plt.close()

print("All audit plots saved successfully!")
