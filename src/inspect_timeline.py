import json

with open('reports/data4cyber_audit_raw.json') as f:
    data = json.load(f)

print(f"{'ID':4} | {'Scenario Name':36} | {'Start Window (Local)':25} | {'End Window (Local)':25} | {'Rows':5}")
print("-" * 105)
for sc, info in data['audit_by_scenario'].items():
    sc_def = info['scenario_definition']
    sid = sc_def.get('scenario_id', '?')
    local_w = sc_def.get('window_local', {})
    start = local_w.get('start', info.get('timestamp_min', '?'))
    end = local_w.get('end', info.get('timestamp_max', '?'))
    rows = info['rows']
    print(f"{sid:4} | {sc:36} | {start:25} | {end:25} | {rows:5}")
