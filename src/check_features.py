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
    "S6_mqtt_supply_chain_compromise"
]

dfs_train = []
for sc in primary_scenarios:
    df = pd.read_csv(os.path.join(DATASET_DIR, sc, "dataset.csv"))
    # 70% chronological train split
    n_train = int(len(df) * 0.70)
    dfs_train.append(df.iloc[:n_train])

train_df = pd.concat(dfs_train, ignore_index=True)
print(f"Total training samples across 6 known scenarios: {len(train_df)}")

# Check constant columns
constant_cols = [c for c in train_df.columns if train_df[c].nunique() <= 1]
print(f"Constant columns in train ({len(constant_cols)}):", constant_cols)

# Check attacker columns
attacker_cols = [c for c in train_df.columns if c.startswith("Attacker.")]
print(f"Attacker columns ({len(attacker_cols)}):", attacker_cols)

# Non-numeric columns
non_num = train_df.select_dtypes(exclude=[np.number]).columns.tolist()
print("Non-numeric columns:", non_num)

