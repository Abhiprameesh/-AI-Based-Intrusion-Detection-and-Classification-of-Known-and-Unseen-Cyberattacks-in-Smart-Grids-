"""
Milestone 4.5 Pre-Execution Integrity Snapshot Script
Calculates and records SHA-256 hashes for:
1. All eight protected Phase 1 artifacts
2. Canonical Milestone 4 output files
3. Recovered Autoencoder checkpoint and threshold metadata
4. Git working tree status
"""

import os
import hashlib
import json
import subprocess

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("MILESTONE 4.5: PRE-EXECUTION INTEGRITY SNAPSHOT")
    print("=" * 80)

    # 1. Protected Phase 1 Artifacts
    phase1_expected = {
        'binary_xgb': ('phase_1/results/data4cyber_dataset/binary/models/XGBoost.joblib', 'f14e1c87d3640bcc8bec44d59925c2fd91b7c630a4cfe5e8b1a8d912e8a67e02'),
        'binary_rf': ('phase_1/results/data4cyber_dataset/binary/models/Random_Forest.joblib', '6a66af5e955bc0a51413b96da549f78bbeefbadffcb19476da90da9263b172a4'),
        'binary_prep_unscaled': ('phase_1/results/data4cyber_dataset/binary/models/preprocessor_unscaled.joblib', '2c5ef4a896e75d237f4c59ca35cd3c3d6afd4b2abe23edf16bbc4e5f5d694b12'),
        'binary_prep_scaled': ('phase_1/results/data4cyber_dataset/binary/models/preprocessor_scaled.joblib', 'd80f1484077092c1ba64c9ceec793041e8b9cf037b4f9cd101b6d2b236608ac5'),
        'multiclass_xgb': ('phase_1/results/data4cyber_dataset/multiclass/models/XGBoost_multiclass.joblib', '2ec535dac31d9eafc31252fee3cc0335d8fd33bab47db527efa4e85a392459b9'),
        'multiclass_rf': ('phase_1/results/data4cyber_dataset/multiclass/models/Random_Forest_multiclass.joblib', '8987a92549af94330e33266c2eada17ee7ece6a037c43ad3e8925475d8f47501'),
        'multiclass_prep_unscaled': ('phase_1/results/data4cyber_dataset/multiclass/models/preprocessor_unscaled.joblib', '2c5ef4a896e75d237f4c59ca35cd3c3d6afd4b2abe23edf16bbc4e5f5d694b12'),
        'unseen_metrics_json': ('phase_1/results/data4cyber_dataset/unseen_attack/unseen_attack_MitM_FDI.json', '72a343107bca8dc3b916447832a7dd24398ae862cea0572012cb71c57bff30fa'),
    }

    print("\n--- 1. Protected Phase 1 Artifact Hashes ---")
    p1_ok = True
    p1_results = {}
    for name, (path, exp) in phase1_expected.items():
        assert os.path.exists(path), f"Missing Phase 1 artifact: {path}"
        actual = sha256_file(path)
        match = (actual == exp)
        if not match:
            p1_ok = False
        p1_results[name] = {'path': path, 'hash': actual, 'match': match}
        print(f"  {name:25s}: {actual[:16]}... [Match={match}]")
    print(f"All 8 Phase 1 Hashes Match: {p1_ok}")

    # 2. Canonical Milestone 4 Outputs (from milestone4_1_fixes_manifest.json & unseen_autoencoder_metrics.json)
    m4_expected = {
        'm4_telemetry_features': ('simulation/results/milestone4/telemetry_features.csv', '2b887cdb9f8ade4542d98f05441ffe02ff564f3b4f870d71ddd825342c2e2c17'),
        'm4_telemetry_metadata': ('simulation/results/milestone4/telemetry_metadata.csv', 'bf04427cd3d0705cc5274627b2f5eaa11561e33eaaca981e5d6c955f5e800cb9'),
        'm4_predictions': ('simulation/results/milestone4/predictions_binary_and_multiclass.csv', 'eec818a78a68aa59b6f2b77a96021f710050ad52552688984c8c3dfcbc42692b'),
        'm4_metrics_exp1': ('simulation/results/milestone4/metrics_experiment1_frozen_models.json', '66f7d620571190f672b9a74f011495f86e7e7bb2ed0333e53a324adaf0d27ea7'),
    }

    print("\n--- 2. Canonical Milestone 4 Baseline Output Hashes ---")
    m4_ok = True
    m4_results = {}
    for name, (path, exp) in m4_expected.items():
        assert os.path.exists(path), f"Missing Milestone 4 output: {path}"
        actual = sha256_file(path)
        match = (actual.strip().lower() == exp.strip().lower())
        if not match:
            m4_ok = False
            print(f"  MISMATCH on {name}:")
            print(f"    actual:   {actual}")
            print(f"    expected: {exp}")
        m4_results[name] = {'path': path, 'hash': actual, 'match': match}
        print(f"  {name:25s}: {actual[:16]}... [Match={match}]")
    print(f"All Milestone 4 Baseline Hashes Match: {m4_ok}")

    # 3. Recovered Autoencoder Checkpoint and Metadata (from recovery_manifest.json)
    ae_expected = {
        'recovered_autoencoder_pt': ('phase_1/results/data4cyber_dataset/anomaly/recovery_v1/Autoencoder.pt', '8df8b7333d93ff58d8d4dba44dc60cd09e0fc2558e7191a837942799fba4936c'),
        'threshold_metadata_json': ('phase_1/results/data4cyber_dataset/anomaly/recovery_v1/threshold_metadata.json', 'fcce7d97de5b5c753b4e1da72cffaa928cbb16e9840df99410163252e1aa1c90'),
    }

    print("\n--- 3. Recovered Autoencoder Artifact Hashes ---")
    ae_ok = True
    ae_results = {}
    for name, (path, exp) in ae_expected.items():
        assert os.path.exists(path), f"Missing Autoencoder artifact: {path}"
        actual = sha256_file(path)
        match = (actual.strip().lower() == exp.strip().lower())
        if not match:
            ae_ok = False
            print(f"  MISMATCH on {name}:")
            print(f"    actual:   {actual}")
            print(f"    expected: {exp}")
        ae_results[name] = {'path': path, 'hash': actual, 'match': match}
        print(f"  {name:25s}: {actual[:16]}... [Match={match}]")
    print(f"All Autoencoder Artifact Hashes Match: {ae_ok}")

    # 4. Git status
    print("\n--- 4. Git Working Tree Status ---")
    status_out = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    print("Git porcelain status output:")
    print(status_out if status_out else "  [Working tree clean, no untracked or modified files in index]")

    snapshot = {
        'phase1_artifacts': p1_results,
        'milestone4_baseline_outputs': m4_results,
        'autoencoder_artifacts': ae_results,
        'all_verified': bool(p1_ok and m4_ok and ae_ok)
    }

    os.makedirs("simulation/results/milestone4_5", exist_ok=True)
    with open("simulation/results/milestone4_5/pre_execution_snapshot.json", "w") as fp:
        json.dump(snapshot, fp, indent=2)
    print("\nSnapshot saved to: simulation/results/milestone4_5/pre_execution_snapshot.json")
    print(f"OVERALL INTEGRITY VERIFIED: {snapshot['all_verified']}")

if __name__ == '__main__':
    main()
