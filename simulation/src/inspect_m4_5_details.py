import json
import pandas as pd
import numpy as np

conditions = ['C0_baseline', 'C1_voltage_alignment', 'C2_reactive_alignment', 'C3_combined_alignment']

print("================================================================================")
print("MILESTONE 4.5 DETAILED RESULTS SUMMARY")
print("================================================================================")

for cid in conditions:
    path = f"simulation/results/milestone4_5/candidates/{cid}/metrics_all_stages.json"
    with open(path) as f:
        data = json.load(f)
    
    label = data['label']
    s1 = data['stage_1_binary_xgboost']
    s2 = data['stage_2_known_attack_multiclass']
    s3 = data['stage_3_unseen_autoencoder']
    val = data['validation']
    
    val_meas = val['achieved_physical_measurements']
    print(f"\nCONDITION: {cid} ({label})")
    print("-" * 60)
    print("Physical Telemetry:")
    print(f"  Substation V_LN:   {val_meas['substation_v_ln']['mean']:.2f} V")
    print(f"  Load A V_LN:       {val_meas['load_a_v_ln']['mean']:.2f} V")
    print(f"  Load A Q:          {val_meas['load_a_q_var']['mean']:.2f} var")
    print(f"  Load B Q:          {val_meas['load_b_q_var']['mean']:.2f} var")
    print(f"  Load A PF:         {val_meas['load_a_power_factor']['mean']:.4f}")
    
    print("Stage 1 (Binary XGBoost):")
    print(f"  Primary (tau=0.50): Acc={s1['primary_threshold_0_50']['accuracy']:.4f}, Recall={s1['primary_threshold_0_50']['recall']:.4f}, FPR={s1['primary_threshold_0_50']['false_positive_rate']:.4f}")
    print(f"  Diagnostic (tau=0.05): Recall={s1['secondary_diagnostic_threshold_0_05']['recall']:.4f}, Prec={s1['secondary_diagnostic_threshold_0_05']['precision']:.4f}, F1={s1['secondary_diagnostic_threshold_0_05']['f1_score']:.4f}, FPR={s1['secondary_diagnostic_threshold_0_05']['false_positive_rate']:.4f}")
    p_all = s1['probability_distributions']['all']
    print(f"  Attack Probabilities: Mean={p_all['mean']:.4f}, Max={p_all['max']:.4f}, Median={p_all['median']:.4f}")
    print(f"  ROC-AUC: {s1['ranking_metrics']['roc_auc']:.4f}, PR-AUC: {s1['ranking_metrics']['pr_auc']:.4f}")
    
    print("Stage 2 (Multiclass Random Forest):")
    print(f"  Accuracy: {s2['overall_accuracy']:.4f}, Macro-F1: {s2['macro_f1']:.4f}, Weighted-F1: {s2['weighted_f1']:.4f}")
    print(f"  Confusion Matrix 4x4:")
    print(f"    {s2['confusion_matrix_4x4']}")
    print(f"  Classification Report:")
    for cls_name, cls_m in s2['classification_report'].items():
        if isinstance(cls_m, dict):
            print(f"    {cls_name}: Precision={cls_m['precision']:.4f}, Recall={cls_m['recall']:.4f}, F1={cls_m['f1-score']:.4f}, Support={cls_m['support']}")
        else:
            print(f"    {cls_name}: {cls_m}")
        
    print("Stage 3 (Deep Autoencoder):")
    mse_dist = s3['reconstruction_mse_distributions']
    print(f"  Mean MSE: All={mse_dist['all']['mean']:.4f}, Benign={mse_dist['benign']['mean']:.4f}, Attack={mse_dist['attack']['mean']:.4f}")
    print(f"  Median MSE: All={mse_dist['all']['median']:.4f}, Benign={mse_dist['benign']['median']:.4f}, Attack={mse_dist['attack']['median']:.4f}")
    print(f"  Evaluation at tau={s3['threshold_recovered']:.4f}: Recall={s3['evaluation_recovered_threshold']['recall']:.4f}, Benign FPR={s3['evaluation_recovered_threshold']['false_positive_rate']:.4f}")
    print(f"  ROC-AUC: {s3['ranking_metrics']['roc_auc']:.4f}, PR-AUC: {s3['ranking_metrics']['pr_auc']:.4f}")
    print("  Feature Group Errors:")
    for grp, err_dict in s3['feature_group_errors'].items():
        print(f"    {grp}: sum={err_dict['mse_sum']:.2f}, pct={err_dict['pct']:.2f}%")
    print("  Top 5 Error Features:")
    for item in s3['top_15_features'][:5]:
        print(f"    {item['feature']}: mse={item['mse']:.2f}, pct={item['pct']:.2f}%")

print("\n================================================================================")
print("PER-SCENARIO STAGE 3 MSE COMPARISON")
print("================================================================================")
scenarios = [
    'S0_benign_baseline',
    'S1_fdi_bss_scaling_0.5',
    'S2_fdi_substation_bias_5kw',
    'S3_industroyer_pv_trip',
    'S4_industroyer_bss_forced_discharge',
    'S5_load_b_telemetry_freeze',
    'S6_unseen_stealthy_ramp_fdi',
    'S7_unseen_coordinated_masking',
    'S8_benign_cloud_cover_shading'
]

header = f"{'Scenario':<35} | {'C0 MSE':<10} | {'C1 MSE':<10} | {'C2 MSE':<10} | {'C3 MSE':<10}"
print(header)
print("-" * len(header))

scenario_data = {}
for cid in conditions:
    with open(f"simulation/results/milestone4_5/candidates/{cid}/metrics_all_stages.json") as f:
        d = json.load(f)
    scenario_data[cid] = d['stage_3_unseen_autoencoder']['per_scenario']

for sc in scenarios:
    c0_val = scenario_data['C0_baseline'][sc]['mse_mean']
    c1_val = scenario_data['C1_voltage_alignment'][sc]['mse_mean']
    c2_val = scenario_data['C2_reactive_alignment'][sc]['mse_mean']
    c3_val = scenario_data['C3_combined_alignment'][sc]['mse_mean']
    print(f"{sc:<35} | {c0_val:<10.4f} | {c1_val:<10.4f} | {c2_val:<10.4f} | {c3_val:<10.4f}")
