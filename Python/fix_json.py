import json

path = r'c:\Ocean Library\Matlab code practice\Experiments\Scripts\Simulation\test_simulator_20260402.json'
out_path = r'c:\Ocean Library\Matlab code practice\Experiments\Scripts\Simulation\test_simulator_20260402_edited.json'

with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

ng_x = next(s['position'][0] for s in data['stations'] if s['name'] == 'Nanyang Gateway')
target_x = ng_x + 50
print(f'Target X = {target_x}')

targets = ['Tuas Link', 'Tuas West Road', 'Tuas Crescent', 'Gui Circle', 'Joo Koon', 'Pioneer']
station_map = {s['id']: s for s in data['stations']}
target_ids = [s['id'] for s in data['stations'] if s['name'] in targets]

dx_map = {}
for sid in target_ids:
    s = station_map[sid]
    old_x = s['position'][0]
    dx = target_x - old_x
    dx_map[sid] = dx
    s['position'][0] = target_x
    print(f"Shifted {s['name']} by {dx:.2f} units")

for e in data['edges']:
    src = e['source']
    tgt = e['target']
    cps = e.get('controlPoints', [])
    if not cps: continue
    
    if src in dx_map and tgt in dx_map:
        # Both endpoints shifted. We can simply map the first point to src dx, last point to tgt dx, and interpolate linearly.
        for i, cp in enumerate(cps):
            weight = i / max(1, len(cps) - 1)
            cp_dx = dx_map[src] * (1.0 - weight) + dx_map[tgt] * weight
            cp[0] += cp_dx
            
    elif src in dx_map:
        for i, cp in enumerate(cps):
            weight = 1.0 - (i / max(1, len(cps) - 1))
            cp[0] += dx_map[src] * weight
            
    elif tgt in dx_map:
        for i, cp in enumerate(cps):
            weight = (i / max(1, len(cps) - 1))
            cp[0] += dx_map[tgt] * weight

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print("Saved to test_simulator_20260402_edited.json")
