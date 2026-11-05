import json

with open('reports/data4cyber_audit_raw.json') as f:
    data = json.load(f)

for sc, info in data['audit_by_scenario'].items():
    fam = info['scenario_definition'].get('attack_family', 'none')
    print(f"=== {sc} ({fam}) ===")
    print(f"Rows: {info['rows']}, Nulls: {info['total_nulls']}, Dupes: {info['duplicates']}")
    print("Attack Active:", info['attack_active_counts'])
    print("Attack Phases:", info['attack_phase_counts'])
    phases = [(p.get('name'), p.get('type'), p.get('target'), p.get('phase')) for p in info.get('phases', [])]
    print("Phases definition:", phases)
    print()
