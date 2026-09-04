"""Re-merge the Monte-Carlo partials into envvars_results.json.

The null and the injection realisations are independent draws produced by
`fixedeffects.py worker`, so more slices can be added at any time and merged
here without re-running the deterministic sections.
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

import fixedeffects as F

HERE = os.path.dirname(os.path.abspath(__file__))
RP = os.path.join(HERE, "envvars_results.json")
R = json.load(open(RP, encoding="utf-8"))

parts = sorted(glob.glob(os.path.join(HERE, "null_part_*.json")))
if parts:
    R["null"] = F.merge_partials(parts)
    n = R["null"]["x1_raw"]["within_object"]["n"]
    print(f"null: merged {len(parts)} slices, {n} realisations")

resp = {}
for key in ("x1", "x2", "x3", "x4a", "x4b"):
    rows = {"within_object": [], "within_class": []}
    ok = True
    for binj in (0.0, 0.3):
        pp = sorted(glob.glob(os.path.join(
            HERE, f"inj_{key}_{int(round(binj*100))}_*.json")))
        if not pp:
            ok = False
            break
        m = F.merge_partials(pp)
        for nm in rows:
            rows[nm].append((binj, m[f"{key}_raw"][nm]["mean"],
                             m[f"{key}_raw"][nm]["sem"],
                             m[f"{key}_raw"][nm]["n"],
                             m[f"{key}_raw"][nm]["frac_at_grid_edge"]))
    if not ok:
        continue
    resp[key] = {}
    for nm in rows:
        (b0, m0, e0, n0, f0), (b1, m1, e1, n1, f1) = rows[nm]
        resp[key][nm] = dict(
            slope=float((m1 - m0) / (b1 - b0)),
            slope_err=float(math.hypot(e0, e1) / (b1 - b0)),
            at_0=float(m0), at_injected=float(m1), beta_injected=float(b1),
            n_realisations=int(min(n0, n1)),
            frac_at_grid_edge=float(max(f0, f1)))
        print(f"{key:4s} {nm:14s} slope {resp[key][nm]['slope']:+.4f} +- "
              f"{resp[key][nm]['slope_err']:.4f}  "
              f"({m0:+.4f} -> {m1:+.4f}, n = {min(n0, n1)}, "
              f"edge {max(f0, f1):.2f})")
if resp:
    R["responsiveness"] = resp

with open(RP, "w", encoding="utf-8") as f:
    json.dump(R, f, indent=1)
print(f"\nwrote {RP}")
