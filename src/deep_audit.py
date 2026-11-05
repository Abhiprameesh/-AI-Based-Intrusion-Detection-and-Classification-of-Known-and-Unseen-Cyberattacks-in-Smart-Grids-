import os
import glob
import json
import pandas as pd
import numpy as np

DATASET_DIR = r"c:\ALLMLPROJ\MAINMLPROJS\MAJOR PROJECT\data4cyber_dataset"
scenarios = [
    "S0_benign_baseline",
    "S1_industroyer_pv",
    "S1_industroyer_pv_alt",
    "S2_industroyer_bss",
    "S3_arp_spoof_bss_meter_half_values",
    "S4_arp_spoof_loads_pv_two_phase",
    "S5_arp_spoof_loads_pv_bss_two_phase",
    "S6_mqtt_supply_chain_compromise"
]

audit_results = {}
all_columns = {}
common_columns = None

for sc in scenarios:
    sc_path = os.path.join(DATASET_DIR, sc)
    csv_path = os.path.join(sc_path, "dataset.csv")
    meta_path = os.path.join(sc_path, "metadata.json")
    phases_path = os.path.join(sc_path, "attack-phases.json")
    sc_def_path = os.path.join(sc_path, "scenario_definition.json")
    
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
    phases = []
    if os.path.exists(phases_path):
        with open(phases_path, 'r') as f:
            phases = json.load(f)
            
    sc_def = {}
    if os.path.exists(sc_def_path):
        with open(sc_def_path, 'r') as f:
            sc_def = json.load(f)
            
    df = pd.read_csv(csv_path)
    
    # Analyze columns
    cols = df.columns.tolist()
    all_columns[sc] = set(cols)
    if common_columns is None:
        common_columns = set(cols)
    else:
        common_columns = common_columns.intersection(set(cols))
        
    # Attack columns
    attack_active_counts = df['attack_active'].value_counts(dropna=False).to_dict()
    attack_phase_counts = df['attack_phase'].value_counts(dropna=False).to_dict() if 'attack_phase' in df.columns else {}
    attack_phase_all_counts = df['attack_phase_all'].value_counts(dropna=False).head(10).to_dict() if 'attack_phase_all' in df.columns else {}
    
    # Nulls & dupes
    null_counts = df.isnull().sum().to_dict()
    total_nulls = df.isnull().sum().sum()
    dupes = df.duplicated().sum()
    
    # Dtypes
    dtypes = {k: str(v) for k, v in df.dtypes.to_dict().items()}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    ts_dt = pd.to_datetime(df['timestamp'], errors='coerce') if 'timestamp' in df.columns else None
    ts_min = str(ts_dt.min()) if ts_dt is not None else None
    ts_max = str(ts_dt.max()) if ts_dt is not None else None
    ts_step = (ts_dt.diff().dt.total_seconds().mode()[0]) if ts_dt is not None and len(df) > 1 else None
    
    # Attacker features
    attacker_cols = [c for c in cols if c.startswith('Attacker.')]
    
    # Cyber vs Physical
    # Cyber: MQTT, Modbus, Attacker
    cyber_cols = [c for c in cols if 'MQTT' in c or 'Modbus' in c or 'Attacker' in c or 'ip' in c.lower()]
    physical_cols = [c for c in cols if c not in cyber_cols and c not in ['timestamp', 'attack_active', 'attack_phase', 'attack_phase_all']]
    
    audit_results[sc] = {
        'scenario_definition': sc_def,
        'metadata': meta,
        'phases': phases,
        'rows': len(df),
        'columns': len(cols),
        'numeric_columns_count': len(numeric_cols),
        'object_columns_count': len(object_cols),
        'total_nulls': int(total_nulls),
        'duplicates': int(dupes),
        'timestamp_min': ts_min,
        'timestamp_max': ts_max,
        'timestamp_step': float(ts_step) if ts_step is not None else None,
        'attack_active_counts': {str(k): int(v) for k, v in attack_active_counts.items()},
        'attack_phase_counts': {str(k): int(v) for k, v in attack_phase_counts.items()},
        'attacker_columns': attacker_cols,
        'cyber_cols_sample': cyber_cols[:10],
        'cyber_cols_count': len(cyber_cols),
        'physical_cols_sample': physical_cols[:10],
        'physical_cols_count': len(physical_cols),
    }

summary = {
    'audit_by_scenario': audit_results,
    'common_columns_count': len(common_columns),
    'all_scenarios_columns': {sc: len(cols) for sc, cols in all_columns.items()}
}

out_path = r"c:\ALLMLPROJ\MAINMLPROJS\MAJOR PROJECT\reports\data4cyber_audit_raw.json"
with open(out_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Data audit complete. Wrote raw summary to {out_path}")
print(f"Common columns across all 8 scenarios: {len(common_columns)}")
for sc, res in audit_results.items():
    fam = res['scenario_definition'].get('attack_family', 'unknown')
    print(f"[{sc}] family: {fam}, rows: {res['rows']}, benign: {res['attack_active_counts'].get('False', 0)}, attack: {res['attack_active_counts'].get('True', 0)}, attacker_cols: {res['attacker_columns']}")
