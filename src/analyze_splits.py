import os
import pandas as pd
import numpy as np

DATASET_DIR = r"c:\ALLMLPROJ\MAINMLPROJS\MAJOR PROJECT\data4cyber_dataset"

primary_scenarios = [
    "S0_benign_baseline",
    "S1_industroyer_pv",
    "S2_industroyer_bss",
    "S4_arp_spoof_loads_pv_two_phase",
    "S5_arp_spoof_loads_pv_bss_two_phase",
    "S6_mqtt_supply_chain_compromise",
    "S3_arp_spoof_bss_meter_half_values"
]

dfs = {}
for sc in primary_scenarios:
    df = pd.read_csv(os.path.join(DATASET_DIR, sc, "dataset.csv"))
    df['scenario'] = sc
    # map attack family
    if sc == "S0_benign_baseline":
        df['attack_family'] = 'benign'
    elif sc in ["S1_industroyer_pv", "S2_industroyer_bss"]:
        df['attack_family'] = df['attack_active'].apply(lambda a: 'industroyer_modbus' if a else 'benign')
    elif sc in ["S3_arp_spoof_bss_meter_half_values", "S4_arp_spoof_loads_pv_two_phase", "S5_arp_spoof_loads_pv_bss_two_phase"]:
        df['attack_family'] = df['attack_active'].apply(lambda a: 'mitm_false_data_injection' if a else 'benign')
    elif sc == "S6_mqtt_supply_chain_compromise":
        df['attack_family'] = df['attack_active'].apply(lambda a: 'mqtt_price_signal_manipulation' if a else 'benign')
    dfs[sc] = df

print("Scenario row breakdown:")
for sc, df in dfs.items():
    print(f"{sc:36} | Total: {len(df):4} | Benign: {(df['attack_active']==False).sum():4} | Attack: {(df['attack_active']==True).sum():4} | Families: {dict(df['attack_family'].value_counts())}")

