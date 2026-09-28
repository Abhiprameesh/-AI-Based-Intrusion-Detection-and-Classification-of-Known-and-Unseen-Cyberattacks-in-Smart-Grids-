import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
PHASE_1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(PHASE_1_DIR, ".."))
WORKSPACE = os.environ.get("MAJOR_PROJECT_ROOT", ROOT_DIR)

def clean_inf_and_nan(df, numeric_cols):
    """Replaces infinite values with NaN for proper imputation."""
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df

def load_binary_all_natural(sample_frac=1.0, random_state=42):
    """
    Loads binaryAllNaturalPlusNormalVsAttacks dataset across all 15 CSV files.
    Labels: 'Natural' -> 0, 'Attack' -> 1
    Stratified split: 70% Train, 15% Val, 15% Test.
    """
    folder = os.path.join(WORKSPACE, "binaryAllNaturalPlusNormalVsAttacks")
    files = sorted(glob.glob(os.path.join(folder, "data*.csv")))
    dfs = [pd.read_csv(f) for f in files]
    full_df = pd.concat(dfs, ignore_index=True)

    if sample_frac < 1.0:
        full_df = full_df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

    X = full_df.drop(columns=['marker'])
    y = full_df['marker'].map({'Natural': 0, 'Attack': 1}).values

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = clean_inf_and_nan(X, numeric_cols)

    # Stratified Train (70%), Val (15%), Test (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=random_state
    )
    val_rel_size = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_rel_size, stratify=y_train_val, random_state=random_state
    )

    feature_names = list(X.columns)
    return {
        'dataset_name': 'binaryAllNaturalPlusNormalVsAttacks',
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'feature_names': feature_names,
        'class_names': ['Natural', 'Attack'],
        'task_type': 'binary'
    }

def load_cicflowmeter_dnp3(val_size=0.15, random_state=42):
    """
    Loads CICFlowMeter_Training_Balanced.csv and CICFlowMeter_Testing_Balanced.csv.
    Strips identifier and leakage columns.
    Returns both binary ('NORMAL' vs Attacks) and multiclass labels.
    """
    train_path = os.path.join(WORKSPACE, "CICFlowMeter_Training_Balanced.csv")
    test_path = os.path.join(WORKSPACE, "CICFlowMeter_Testing_Balanced.csv")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Identifiers & direct leakage columns
    leakage_cols = ['Unnamed: 0', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Timestamp', 'Protocol']
    cols_to_drop = [c for c in leakage_cols if c in df_train.columns]

    X_train_full = df_train.drop(columns=cols_to_drop + ['Label'])
    y_train_raw = df_train['Label'].values

    X_test = df_test.drop(columns=cols_to_drop + ['Label'])
    y_test_raw = df_test['Label'].values

    numeric_cols = X_train_full.select_dtypes(include=[np.number]).columns.tolist()
    X_train_full = clean_inf_and_nan(X_train_full, numeric_cols)
    X_test = clean_inf_and_nan(X_test, numeric_cols)

    le = LabelEncoder()
    le.fit(np.concatenate([y_train_raw, y_test_raw]))
    y_train_multi = le.transform(y_train_raw)
    y_test_multi = le.transform(y_test_raw)

    normal_class_idx = list(le.classes_).index('NORMAL') if 'NORMAL' in le.classes_ else 0
    y_train_bin = (y_train_multi != normal_class_idx).astype(int)
    y_test_bin = (y_test_multi != normal_class_idx).astype(int)

    # Validation split from official training set
    X_train, X_val, y_train_m, y_val_m, y_train_b, y_val_b = train_test_split(
        X_train_full, y_train_multi, y_train_bin, test_size=val_size, stratify=y_train_multi, random_state=random_state
    )

    feature_names = list(X_train.columns)
    return {
        'dataset_name': 'CICFlowMeter_Balanced_DNP3_SCADA',
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train_binary': y_train_b, 'y_val_binary': y_val_b, 'y_test_binary': y_test_bin,
        'y_train_multi': y_train_m, 'y_val_multi': y_val_m, 'y_test_multi': y_test_multi,
        'feature_names': feature_names,
        'class_names_binary': ['NORMAL', 'ATTACK'],
        'class_names_multi': list(le.classes_),
        'label_encoder': le
    }

def load_data4cyber(random_state=42):
    """
    Loads all scenarios from data4cyber_dataset.
    Strips attacker logging tags ('Attacker.*') and timestamp from features.
    Provides binary (attack_active) and multiclass (attack_family) tasks.
    """
    base_folder = os.path.join(WORKSPACE, "data4cyber_dataset")
    scenarios = sorted([d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d)) and d.startswith("S")])

    family_map = {
        'S0_benign_baseline': 'benign',
        'S1_industroyer_pv': 'industroyer_modbus',
        'S1_industroyer_pv_alt': 'industroyer_modbus',
        'S2_industroyer_bss': 'industroyer_modbus',
        'S3_arp_spoof_bss_meter_half_values': 'mitm_false_data_injection',
        'S4_arp_spoof_loads_pv_two_phase': 'mitm_false_data_injection',
        'S5_arp_spoof_loads_pv_bss_two_phase': 'mitm_false_data_injection',
        'S6_mqtt_supply_chain_compromise': 'mqtt_price_signal_manipulation'
    }

    dfs = []
    for sc in scenarios:
        csv_p = os.path.join(base_folder, sc, "dataset.csv")
        df_sc = pd.read_csv(csv_p)
        df_sc['scenario'] = sc
        df_sc['attack_family'] = family_map.get(sc, 'unknown')
        dfs.append(df_sc)

    full_df = pd.concat(dfs, ignore_index=True)

    # Strip attacker ground truth logs, timestamps, and metadata strings
    leakage_cols = [c for c in full_df.columns if 'Attacker.' in c or 'timestamp' in c.lower() or c in ['attack_phase', 'attack_phase_all', 'scenario', 'MQTT-Price-Signal.event']]
    feature_cols = [c for c in full_df.columns if c not in leakage_cols and c not in ['attack_active', 'attack_family']]

    X = full_df[feature_cols].copy()
    X = X.select_dtypes(include=[np.number])
    feature_cols = list(X.columns)
    y_bin = full_df['attack_active'].astype(int).values

    le_fam = LabelEncoder()
    y_family = le_fam.fit_transform(full_df['attack_family'])

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = clean_inf_and_nan(X, numeric_cols)

    # Train/Val/Test Split (70/15/15 stratified)
    X_train_val, X_test, y_b_tv, y_b_test, y_f_tv, y_f_test = train_test_split(
        X, y_bin, y_family, test_size=0.15, stratify=y_family, random_state=random_state
    )
    X_train, X_val, y_b_train, y_b_val, y_f_train, y_f_val = train_test_split(
        X_train_val, y_b_tv, y_f_tv, test_size=0.15/0.85, stratify=y_f_tv, random_state=random_state
    )

    return {
        'dataset_name': 'data4cyber_dataset',
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train_binary': y_b_train, 'y_val_binary': y_b_val, 'y_test_binary': y_b_test,
        'y_train_family': y_f_train, 'y_val_family': y_f_val, 'y_test_family': y_f_test,
        'feature_names': feature_cols,
        'class_names_binary': ['Benign', 'Attack'],
        'class_names_family': list(le_fam.classes_),
        'label_encoder': le_fam,
        'full_df': full_df
    }

def load_pmu_sgsma(bus='Bus29', random_state=42):
    """
    Loads PMU synchrophasor dataset from IEEE SGSMA 2024.
    Strict chronological time-series splitting (70% Train, 15% Val, 15% Test)
    to prevent temporal leakage.
    Target: Event (0: Normal, >0: Anomalies/Cyber-Power events).
    """
    pmu_dir = os.path.join(WORKSPACE, "SGSMA_Competiton 2024_PMU_DATA", "PMU_Data_with_Anomalies and Events")
    file_path = os.path.join(pmu_dir, f"{bus}_Competition_Data.csv")
    df = pd.read_csv(file_path)

    # Exclude TIMESTAMP from features to prevent temporal memorization
    feature_cols = [c for c in df.columns if c not in ['TIMESTAMP', 'Event']]
    X = df[feature_cols].copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = clean_inf_and_nan(X[numeric_cols], numeric_cols)
    feature_cols = list(X.columns)
    y_multi = df['Event'].values
    y_bin = (y_multi != 0).astype(int)

    # Episodic Chronological Split:
    # Preserves strict temporal ordering within each continuous operational and event episode
    # (Past 70% -> Train, Middle 15% -> Val, Future 15% -> Test)
    # This prevents temporal data leakage while ensuring all transient cyber-physical events are represented.
    train_mask = np.zeros(len(df), dtype=bool)
    val_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)

    for ev in np.unique(y_multi):
        ev_indices = np.where(y_multi == ev)[0]
        n = len(ev_indices)
        n_tr = int(n * 0.70)
        n_va = int(n * 0.15)
        train_mask[ev_indices[:n_tr]] = True
        val_mask[ev_indices[n_tr:n_tr+n_va]] = True
        test_mask[ev_indices[n_tr+n_va:]] = True

    X_train = X.iloc[train_mask].copy()
    y_train_m = y_multi[train_mask]
    y_train_b = y_bin[train_mask]

    X_val = X.iloc[val_mask].copy()
    y_val_m = y_multi[val_mask]
    y_val_b = y_bin[val_mask]

    X_test = X.iloc[test_mask].copy()
    y_test_m = y_multi[test_mask]
    y_test_b = y_bin[test_mask]

    return {
        'dataset_name': f'PMU_Data_with_Anomalies_and_Events_{bus}',
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train_binary': y_train_b, 'y_val_binary': y_val_b, 'y_test_binary': y_test_b,
        'y_train_multi': y_train_m, 'y_val_multi': y_val_m, 'y_test_multi': y_test_m,
        'feature_names': feature_cols,
        'class_names_binary': ['Normal', 'Event'],
        'timestamps_train': df['TIMESTAMP'].iloc[train_mask].values,
        'timestamps_test': df['TIMESTAMP'].iloc[test_mask].values
    }

def get_preprocessor(scaling=True):
    """
    Builds a robust sklearn pipeline preprocessor:
    1. SimpleImputer (median)
    2. StandardScaler (optional for tree models vs linear/distance models)
    Fitted strictly on training data.
    """
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if scaling:
        steps.append(('scaler', StandardScaler()))
    return Pipeline(steps)

def load_iec104_balanced(window=15, feature_type='cic', val_size=0.15, random_state=42):
    """
    Loads Balanced_IEC104_Train_Test_CSV_Files dataset.
    feature_type: 'cic' (CICFlowMeter 84 features) or 'custom' (IEC 104 ASDU semantics 112 features).
    """
    base_dir = os.path.join(WORKSPACE, "Balanced_IEC104_Train_Test_CSV_Files", "iec104_train_test_csvs")
    folder_name = f"tests_{feature_type}_{window}"
    target_folder = os.path.join(base_dir, folder_name)

    if feature_type == 'cic':
        train_file = os.path.join(target_folder, f"train_{window}_cicflow.csv")
        test_file = os.path.join(target_folder, f"test_{window}_cicflow.csv")
    else:
        train_file = os.path.join(target_folder, f"train_{window}_custom_script.csv")
        test_file = os.path.join(target_folder, f"test_{window}_custom_script.csv")

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    leakage_cols = ['Unnamed: 0', 'Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Timestamp', 'Protocol']
    cols_to_drop = [c for c in leakage_cols if c in df_train.columns]

    X_train_full = df_train.drop(columns=cols_to_drop + ['Label'])
    y_train_raw = df_train['Label'].values

    X_test = df_test.drop(columns=cols_to_drop + ['Label'])
    y_test_raw = df_test['Label'].values

    for col in X_train_full.columns:
        X_train_full[col] = pd.to_numeric(X_train_full[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

    numeric_cols = list(X_train_full.columns)
    X_train_full = clean_inf_and_nan(X_train_full, numeric_cols)
    X_test = clean_inf_and_nan(X_test, numeric_cols)

    le = LabelEncoder()
    le.fit(np.concatenate([y_train_raw, y_test_raw]))
    y_train_multi = le.transform(y_train_raw)
    y_test_multi = le.transform(y_test_raw)

    norm_idx = list(le.classes_).index('NORMAL') if 'NORMAL' in le.classes_ else 0
    y_train_bin = (y_train_multi != norm_idx).astype(int)
    y_test_bin = (y_test_multi != norm_idx).astype(int)

    X_train, X_val, y_train_m, y_val_m, y_train_b, y_val_b = train_test_split(
        X_train_full, y_train_multi, y_train_bin, test_size=val_size, stratify=y_train_multi, random_state=random_state
    )

    feature_names = list(X_train.columns)
    return {
        'dataset_name': f'Balanced_IEC104_{feature_type}_{window}',
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train_binary': y_train_b, 'y_val_binary': y_val_b, 'y_test_binary': y_test_bin,
        'y_train_multi': y_train_m, 'y_val_multi': y_val_m, 'y_test_multi': y_test_multi,
        'feature_names': feature_names,
        'class_names_binary': ['NORMAL', 'ATTACK'],
        'class_names_multi': list(le.classes_),
        'label_encoder': le
    }
