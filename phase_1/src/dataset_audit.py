import os
import glob
import json
import pandas as pd
PHASE_1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(PHASE_1_DIR, ".."))
WORKSPACE = os.environ.get("MAJOR_PROJECT_ROOT", ROOT_DIR)
REPORTS_DIR = os.path.join(PHASE_1_DIR, "reports")

def audit_binary_all_natural():
    print("\n" + "="*80)
    print("AUDITING: binaryAllNaturalPlusNormalVsAttacks")
    print("="*80)
    folder = os.path.join(WORKSPACE, "binaryAllNaturalPlusNormalVsAttacks")
    files = sorted(glob.glob(os.path.join(folder, "data*.csv")))
    print(f"Discovered {len(files)} files: {[os.path.basename(f) for f in files]}")
    
    total_rows = 0
    all_class_counts = {}
    col_names = None
    all_nulls = 0
    all_dupes = 0
    
    file_summaries = []
    
    for f in files:
        df = pd.read_csv(f)
        if col_names is None:
            col_names = list(df.columns)
        total_rows += len(df)
        counts = df['marker'].value_counts().to_dict()
        for k, v in counts.items():
            all_class_counts[k] = all_class_counts.get(k, 0) + v
        null_count = df.isnull().sum().sum()
        all_nulls += null_count
        dupes = df.duplicated().sum()
        all_dupes += dupes
        file_summaries.append({
            'file': os.path.basename(f),
            'rows': len(df),
            'cols': len(df.columns),
            'nulls': null_count,
            'dupes': dupes,
            'classes': counts
        })
    
    print(f"Total Rows across 15 files: {total_rows}")
    print(f"Total Columns: {len(col_names)}")
    print(f"Target Column: 'marker'")
    print(f"Overall Class Distribution: {all_class_counts}")
    print(f"Total Null Values: {all_nulls}")
    print(f"Total Duplicate Rows: {all_dupes}")
    print(f"Columns Sample (first 10): {col_names[:10]}")
    print(f"Columns Sample (last 15): {col_names[-15:]}")
    
    # Check data types and constant features on data1
    df1 = pd.read_csv(files[0])
    const_cols = [c for c in df1.columns if df1[c].nunique() <= 1]
    print(f"Constant/Near-constant features in data1: {const_cols}")
    
    return {
        'name': 'binaryAllNaturalPlusNormalVsAttacks',
        'path': folder,
        'file_count': len(files),
        'files': [os.path.basename(f) for f in files],
        'total_rows': total_rows,
        'cols': len(col_names),
        'target': 'marker',
        'classes': all_class_counts,
        'nulls': all_nulls,
        'dupes': all_dupes,
        'features': col_names,
        'file_summaries': file_summaries
    }

def audit_sgsma_pmu():
    print("\n" + "="*80)
    print("AUDITING: PMU_Data_with_Anomalies and Events (SGSMA 2024)")
    print("="*80)
    folder = os.path.join(WORKSPACE, "SGSMA_Competiton 2024_PMU_DATA", "PMU_Data_with_Anomalies and Events")
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    print(f"Discovered {len(files)} files: {[os.path.basename(f) for f in files]}")
    
    summaries = []
    for f in files:
        df = pd.read_csv(f)
        event_counts = df['Event'].value_counts().to_dict()
        nulls = df.isnull().sum().to_dict()
        active_nulls = {k: v for k, v in nulls.items() if v > 0}
        dupes = df.duplicated().sum()
        time_diff = df['TIMESTAMP'].diff().dropna()
        sampling_rate = 1.0 / time_diff.median() if time_diff.median() > 0 else 0
        
        summaries.append({
            'file': os.path.basename(f),
            'rows': len(df),
            'cols': len(df.columns),
            'columns': list(df.columns),
            'target': 'Event',
            'event_counts': event_counts,
            'nulls': active_nulls,
            'dupes': dupes,
            'time_min': df['TIMESTAMP'].min(),
            'time_max': df['TIMESTAMP'].max(),
            'sampling_rate_hz': round(sampling_rate, 2)
        })
        print(f"\n{os.path.basename(f)}: Rows={len(df)}, Cols={len(df.columns)}, Nulls={sum(active_nulls.values())}, Dupes={dupes}, Sampling={round(sampling_rate, 2)} Hz")
        print(f"  Event distribution: {event_counts}")
        if active_nulls:
            print(f"  Null columns: {active_nulls}")
            
    return {
        'name': 'PMU_Data_with_Anomalies and Events',
        'path': folder,
        'file_count': len(files),
        'summaries': summaries
    }

def audit_sgsma_testdata():
    print("\n" + "="*80)
    print("AUDITING: SGSMA_Competition Day_Testdata")
    print("="*80)
    base_folder = os.path.join(WORKSPACE, "SGSMA_Competiton 2024_PMU_DATA", "SGSMA_Competition Day_Testdata")
    set1_files = sorted(glob.glob(os.path.join(base_folder, "Competition_Testing Data Set 1", "*.csv")))
    set2_files = sorted(glob.glob(os.path.join(base_folder, "Competition_Testing Data Set 2", "*.csv")))
    print(f"Set 1 files ({len(set1_files)}): {[os.path.basename(f) for f in set1_files]}")
    print(f"Set 2 files ({len(set2_files)}): {[os.path.basename(f) for f in set2_files]}")
    
    df1 = pd.read_csv(set1_files[0])
    df2 = pd.read_csv(set2_files[0])
    print(f"Set 1 sample ({os.path.basename(set1_files[0])}): Rows={len(df1)}, Cols={len(df1.columns)}, Has 'Event'={'Event' in df1.columns}")
    print(f"  Columns: {list(df1.columns)}")
    print(f"Set 2 sample ({os.path.basename(set2_files[0])}): Rows={len(df2)}, Cols={len(df2.columns)}, Has 'Event'={'Event' in df2.columns}")
    print(f"  Columns: {list(df2.columns)}")
    
    return {
        'name': 'SGSMA_Competition Day_Testdata',
        'path': base_folder,
        'set1_count': len(set1_files),
        'set2_count': len(set2_files),
        'has_labels': False
    }

def audit_iec104_datasets():
    print("\n" + "="*80)
    print("AUDITING: Balanced_IEC104_Train_Test_CSV_Files & Root CICFlowMeter Files")
    print("="*80)
    
    # 1. Root files
    root_train = os.path.join(WORKSPACE, "CICFlowMeter_Training_Balanced.csv")
    root_test = os.path.join(WORKSPACE, "CICFlowMeter_Testing_Balanced.csv")
    
    df_train = pd.read_csv(root_train)
    df_test = pd.read_csv(root_test)
    
    print(f"Root CICFlowMeter_Training_Balanced.csv: Rows={len(df_train)}, Cols={len(df_train.columns)}")
    print(f"  Target 'Label' distribution:\n{df_train['Label'].value_counts()}")
    print(f"  Null count: {df_train.isnull().sum().sum()}")
    print(f"  Duplicates: {df_train.duplicated().sum()}")
    
    print(f"\nRoot CICFlowMeter_Testing_Balanced.csv: Rows={len(df_test)}, Cols={len(df_test.columns)}")
    print(f"  Target 'Label' distribution:\n{df_test['Label'].value_counts()}")
    print(f"  Null count: {df_test.isnull().sum().sum()}")
    print(f"  Duplicates: {df_test.duplicated().sum()}")
    
    # 2. Subfolders in Balanced_IEC104_Train_Test_CSV_Files/iec104_train_test_csvs
    sub_dir = os.path.join(WORKSPACE, "Balanced_IEC104_Train_Test_CSV_Files", "iec104_train_test_csvs")
    subfolders = sorted([d for d in os.listdir(sub_dir) if os.path.isdir(os.path.join(sub_dir, d))])
    print(f"\nSubfolders in iec104_train_test_csvs: {subfolders}")
    
    sub_summaries = {}
    for sf in subfolders:
        s_path = os.path.join(sub_dir, sf)
        csvs = glob.glob(os.path.join(s_path, "*.csv"))
        sub_summaries[sf] = []
        for c in csvs:
            df_c = pd.read_csv(c)
            label_col = 'Label' if 'Label' in df_c.columns else ('label' if 'label' in df_c.columns else None)
            counts = df_c[label_col].value_counts().to_dict() if label_col else {}
            sub_summaries[sf].append({
                'file': os.path.basename(c),
                'rows': len(df_c),
                'cols': len(df_c.columns),
                'label_col': label_col,
                'class_counts': counts,
                'nulls': int(df_c.isnull().sum().sum()),
                'dupes': int(df_c.duplicated().sum())
            })
            print(f"  [{sf}] {os.path.basename(c)}: Rows={len(df_c)}, Cols={len(df_c.columns)}, Label={label_col}, Classes={len(counts)}")
            
    return {
        'name': 'Balanced_IEC104_Train_Test_CSV_Files',
        'root_train': {'rows': len(df_train), 'cols': len(df_train.columns), 'classes': df_train['Label'].value_counts().to_dict()},
        'root_test': {'rows': len(df_test), 'cols': len(df_test.columns), 'classes': df_test['Label'].value_counts().to_dict()},
        'subfolders': sub_summaries
    }

def audit_data4cyber():
    print("\n" + "="*80)
    print("AUDITING: data4cyber_dataset")
    print("="*80)
    base_folder = os.path.join(WORKSPACE, "data4cyber_dataset")
    scenarios = sorted([d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d)) and d.startswith("S")])
    print(f"Scenarios: {scenarios}")
    
    scenario_audits = {}
    total_rows = 0
    all_families = {}
    all_phases = {}
    
    for sc in scenarios:
        csv_path = os.path.join(base_folder, sc, "dataset.csv")
        sc_def_path = os.path.join(base_folder, sc, "scenario_definition.json")
        
        family = "unknown"
        if os.path.exists(sc_def_path):
            with open(sc_def_path, 'r') as f:
                sc_def = json.load(f)
                family = sc_def.get('attack_family', sc_def.get('family', 'unknown'))
        
        df = pd.read_csv(csv_path)
        total_rows += len(df)
        active_counts = df['attack_active'].value_counts().to_dict()
        phase_counts = df['attack_phase'].value_counts().to_dict()
        nulls = int(df.isnull().sum().sum())
        dupes = int(df.duplicated().sum())
        
        all_families[family] = all_families.get(family, 0) + len(df)
        for p, count in phase_counts.items():
            all_phases[p] = all_phases.get(p, 0) + count
            
        leakage_cols = [c for c in df.columns if 'Attacker' in c]
        
        scenario_audits[sc] = {
            'family': family,
            'rows': len(df),
            'cols': len(df.columns),
            'attack_active': active_counts,
            'attack_phases': phase_counts,
            'nulls': nulls,
            'dupes': dupes,
            'leakage_columns': leakage_cols,
            'start_time': df['timestamp'].iloc[0] if 'timestamp' in df.columns else None,
            'end_time': df['timestamp'].iloc[-1] if 'timestamp' in df.columns else None
        }
        print(f"\nScenario {sc} ({family}): Rows={len(df)}, Cols={len(df.columns)}, Nulls={nulls}, Dupes={dupes}")
        print(f"  attack_active: {active_counts}")
        print(f"  attack_phase: {phase_counts}")
        print(f"  Potential leakage cols: {leakage_cols}")
        
    print(f"\nTotal data4cyber rows across all scenarios: {total_rows}")
    print(f"All attack families: {all_families}")
    print(f"All phases: {all_phases}")
    
    return {
        'name': 'data4cyber_dataset',
        'path': base_folder,
        'scenarios': scenario_audits,
        'total_rows': total_rows,
        'families': all_families,
        'phases': all_phases
    }

if __name__ == '__main__':
    res_binary = audit_binary_all_natural()
    res_sgsma = audit_sgsma_pmu()
    res_sgsma_test = audit_sgsma_testdata()
    res_iec104 = audit_iec104_datasets()
    res_data4cyber = audit_data4cyber()
    
    audit_summary = {
        'binaryAllNaturalPlusNormalVsAttacks': res_binary,
        'PMU_Data_with_Anomalies and Events': res_sgsma,
        'SGSMA_Competition Day_Testdata': res_sgsma_test,
        'Balanced_IEC104_Train_Test_CSV_Files': res_iec104,
        'data4cyber_dataset': res_data4cyber
    }
    
    with open(os.path.join(REPORTS_DIR, "dataset_audit_raw.json"), 'w') as f:
        json.dump(audit_summary, f, indent=2, default=str)
    print(f"\nSaved raw audit data to {os.path.join(REPORTS_DIR, 'dataset_audit_raw.json')}")
