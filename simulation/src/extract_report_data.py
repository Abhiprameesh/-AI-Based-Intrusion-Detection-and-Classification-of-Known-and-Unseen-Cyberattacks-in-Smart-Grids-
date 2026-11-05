import json
import pandas as pd
import numpy as np

conditions = ['C0_baseline', 'C1_voltage_alignment', 'C2_reactive_alignment', 'C3_combined_alignment']

print("================================================================================")
print("STAGE 1 PROBABILITY DISTRIBUTIONS")
print("================================================================================")
for cid in conditions:
    d = json.load(open(f"simulation/results/milestone4_5/candidates/{cid}/metrics_all_stages.json"))
    s1 = d['stage_1_binary_xgboost']
    p_all = s1['probability_distributions']['all']
    p_att = s1['probability_distributions']['attack']
    p_ben = s1['probability_distributions']['benign']
    print(f"\n--- {cid} ---")
    print("ALL:    mean={:.4f}, std={:.4f}, median={:.4f}, min={:.4f}, max={:.4f}, p05={:.4f}, p25={:.4f}, p75={:.4f}, p95={:.4f}".format(
        p_all['mean'], p_all['std'], p_all['median'], p_all['min'], p_all['max'], p_all['p05'], p_all['p25'], p_all['p75'], p_all['p95']))
    print("ATTACK: mean={:.4f}, std={:.4f}, median={:.4f}, min={:.4f}, max={:.4f}".format(
        p_att['mean'], p_att['std'], p_att['median'], p_att['min'], p_att['max']))
    print("BENIGN: mean={:.4f}, std={:.4f}, median={:.4f}, min={:.4f}, max={:.4f}".format(
        p_ben['mean'], p_ben['std'], p_ben['median'], p_ben['min'], p_ben['max']))
    print("Scenario breakdown:")
    for sc, sc_d in s1['per_scenario'].items():
        rec_str = f"{sc_d['diagnostic_recall']:.4f}" if 'diagnostic_recall' in sc_d else "N/A"
        print(f"  {sc:<35}: mean={sc_d['prob_mean']:.4f}, max={sc_d['prob_max']:.4f}, tau005_rec={rec_str}")

print("\n================================================================================")
print("STAGE 3 RECONSTRUCTION MSE STATS")
print("================================================================================")
for cid in conditions:
    d = json.load(open(f"simulation/results/milestone4_5/candidates/{cid}/metrics_all_stages.json"))
    s3 = d['stage_3_unseen_autoencoder']
    m_all = s3['reconstruction_mse_distributions']['all']
    m_att = s3['reconstruction_mse_distributions']['attack']
    m_ben = s3['reconstruction_mse_distributions']['benign']
    print(f"\n--- {cid} ---")
    print("ALL:    mean={:.4f}, std={:.4f}, median={:.4f}, min={:.4f}, max={:.4f}, p05={:.4f}, p25={:.4f}, p75={:.4f}, p95={:.4f}".format(
        m_all['mean'], m_all['std'], m_all['median'], m_all['min'], m_all['max'], m_all['p05'], m_all['p25'], m_all['p75'], m_all['p95']))
    print("ATTACK: mean={:.4f}, std={:.4f}, median={:.4f}, min={:.4f}, max={:.4f}".format(
        m_att['mean'], m_att['std'], m_att['median'], m_att['min'], m_att['max']))
    print("BENIGN: mean={:.4f}, std={:.4f}, median={:.4f}, min={:.4f}, max={:.4f}".format(
        m_ben['mean'], m_ben['std'], m_ben['median'], m_ben['min'], m_ben['max']))

