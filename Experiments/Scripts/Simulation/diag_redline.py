"""Diagnostic: find Red Line (NSL) route, then check which edges cover it."""
import json

data = json.load(open('Experiments/Scripts/Simulation/full2.json', encoding='utf-8'))

# Find NSL line
nsl = None
for ln in data.get('lines', []):
    if ln['code'] == 'NSL':
        nsl = ln
        break

if not nsl:
    print("NSL line not found!")
    exit()

print(f"NSL route: {nsl['route']}")
print(f"NSL color: {nsl['color']}")
print(f"NSL route length: {len(nsl['route'])} stations")
print()

# Build station name lookup
stn_map = {s['id']: s['name'] for s in data['stations']}

# For each consecutive pair in the NSL route, find an edge
route = nsl['route']
edge_lookup = {}
for e in data.get('edges', []):
    key = (e['source'], e['target'])
    if key not in edge_lookup:
        edge_lookup[key] = []
    edge_lookup[key].append(e)
    # Also reverse for undirected
    rkey = (e['target'], e['source'])
    if rkey not in edge_lookup:
        edge_lookup[rkey] = []
    edge_lookup[rkey].append(e)

print("=== NSL consecutive station pairs ===")
missing = []
for i in range(len(route) - 1):
    s, t = route[i], route[i+1]
    sname = stn_map.get(s, '?')
    tname = stn_map.get(t, '?')
    edges = edge_lookup.get((s, t), [])
    if edges:
        colors = [e.get('color', 'NO COLOR') for e in edges]
        has_cp = [bool(e.get('controlPoints')) for e in edges]
        print(f"  {sname}({s}) -> {tname}({t}): {len(edges)} edge(s), colors={colors}, hasCP={has_cp}")
    else:
        missing.append((s, t, sname, tname))
        print(f"  {sname}({s}) -> {tname}({t}): *** MISSING EDGE ***")

print(f"\nTotal missing edges in NSL route: {len(missing)}")
for s, t, sn, tn in missing:
    print(f"  {sn}({s}) -> {tn}({t})")

# Check for duplicate (s,t) pairs across ALL edges
print("\n=== Duplicate (source, target) edges ===")
pair_count = {}
for e in data.get('edges', []):
    key = (e['source'], e['target'])
    pair_count[key] = pair_count.get(key, 0) + 1
for key, cnt in pair_count.items():
    if cnt > 1:
        sname = stn_map.get(key[0], '?')
        tname = stn_map.get(key[1], '?')
        print(f"  {sname}({key[0]}) -> {tname}({key[1]}): {cnt} duplicate edges")
