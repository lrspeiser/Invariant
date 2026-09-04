"""Run AQ -- old (p=3/2) vs new (p=1/2) tournament, candidate by candidate."""
import json

import numpy as np

OLD = json.load(open("tournament_prefix_k15.json"))
NEW = json.load(open("tournament.json"))
o = {r["name"]: r for r in OLD["records"]}
n = {r["name"]: r for r in NEW["records"]}
assert set(o) == set(n), f"name sets differ: {len(set(o) ^ set(n))} symmetric diff"
print(f"{len(o)} candidates, identical name sets\n")


def fl(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


exposed = [k for k in o if o[k]["base"] == "rar" and o[k]["struct"] != "scalar_a0"]
inert = [k for k in o if k not in set(exposed)]
print(f"exposed (rar, k != 1): {len(exposed)}     inert (aqual/newton/scalar_a0): {len(inert)}")

# --- the inert set MUST be bit-identical; that is the control ---
moved = []
for k in inert:
    a, b = fl(o[k].get("J")), fl(n[k].get("J"))
    if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-12):
        moved.append((k, a, b))
print(f"\nCONTROL -- inert candidates whose J moved: {len(moved)}  (must be 0)")
for k, a, b in moved[:10]:
    print(f"   {k[:52]:52s} {a} -> {b}")

# --- how far did the exposed set move? ---
d = [(k, fl(o[k].get("J")), fl(n[k].get("J"))) for k in exposed]
dd = [(k, a, b) for k, a, b in d if a is not None and b is not None]
delta = np.array([b - a for _, a, b in dd])
print(f"\nEXPOSED -- J change over {len(dd)} scorable candidates")
print(f"   median {np.median(delta):+.4f}   mean {delta.mean():+.4f}   "
      f"sd {delta.std():.4f}   |max| {np.abs(delta).max():.4f}")
print(f"   improved (J down): {int((delta < 0).sum())}   worsened: {int((delta > 0).sum())}"
      f"   unchanged: {int((delta == 0).sum())}")

# --- verdict changes ---
so = {k for k in o if o[k].get("survives")}
sn = {k for k in n if n[k].get("survives")}
print(f"\nSURVIVORS   old {len(so)}   new {len(sn)}")
print(f"   lost  ({len(so - sn)}): " + ("; ".join(sorted(x[:46] for x in so - sn)) or "none"))
print(f"   gained({len(sn - so)}): " + ("; ".join(sorted(x[:46] for x in sn - so)) or "none"))
print(f"   kept  ({len(so & sn)})")

print("\nNEW survivor list, best J first")
print(f"   {'name':<46s} {'base':<6s} {'struct':<10s} {'J':>8s} {'J_old':>8s}")
for k in sorted(sn, key=lambda k: fl(n[k].get("J")) or 9e9):
    jo = fl(o[k].get("J"))
    print(f"   {k[:46]:<46s} {n[k]['base']:<6s} {n[k]['struct']:<10s} "
          f"{fl(n[k].get('J')) or float('nan'):8.4f} "
          f"{jo if jo is not None else float('nan'):8.4f}")

# --- does the funnel change shape? ---
print("\nFUNNEL")
fo, fn = OLD["funnel"], NEW["funnel"]
for key in fo:
    a, b = fo[key], fn.get(key)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        flag = "   <== CHANGED" if a != b else ""
        print(f"   {key:<20s} {a:>6} -> {b:>6}{flag}")
