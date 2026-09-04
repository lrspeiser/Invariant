"""run_finescan.py -- extend each amplitude scan downwards until it is unsaturated.

The first-pass grids were chosen around the a-priori fiducial knobs.  Several
of them turned out to be entirely above this corpus's sensitivity: the scan
saturates at z = 8.5 from its smallest non-zero amplitude, so the threshold
would have to be read off an interpolation across a saturated point.

This pass adds lower amplitudes -- dividing the smallest non-zero point by 4
repeatedly -- until the scan drops below the family-wise critical value, then
rewrites E3_amplitude_scans.json with the merged grid and recomputed
thresholds.  It changes no analysis, only the sampling of the knob.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from universes import generate as gn          # noqa: E402
from universes import physics as ph           # noqa: E402
from universes import provenance as pv        # noqa: E402
from universes import run_stage5 as R5        # noqa: E402
from universes import stats as st             # noqa: E402

RES = os.path.join(HERE, "results")
N = int(os.environ.get("N_AMP", 480))
MAXSTEPS = int(os.environ.get("FINE_STEPS", 4))
SEED = 5200000


def main():
    gn.get_lib(); gn.get_pool()
    pv.start_ledger(os.path.dirname(HERE))
    with open(os.path.join(RES, "E3_amplitude_scans.json")) as f:
        E3 = json.load(f)
    with open(os.path.join(RES, "channel_map.json")) as f:
        CM = json.load(f)
    R5.KEYS = CM["feature_order"]
    R5.CHAN_IDX = {c: [R5.KEYS.index(k) for k in v] for c, v in CM["channels"].items()}
    zc = E3[list(E3)[0]]["z_crit_family"]

    base = R5.pool("U03_mond_scalar", ("U03_mond_scalar", None, 1.0, 1.0), 480,
                   R5.SEED0 + 10000 * 2)
    Bc3, Ba3 = R5.quarters(base)

    for k, (uid, v) in enumerate(E3.items()):
        rows = sorted(v["rows"], key=lambda r: r["amp"])
        nz = [r for r in rows if r["amp"] > 0]
        step = 0
        while step < MAXSTEPS and nz[0]["z"] >= zc:
            amp = nz[0]["amp"] / 4.0
            recs = R5.pool(f"fine_{uid}_{amp:g}", (uid, amp, 1.0, 1.0), N,
                           SEED + 30000 * k + 1000 * step)
            Ac, Aa = R5.quarters(recs)
            n, na = min(len(Ac), len(Bc3)), min(len(Aa), len(Ba3))
            r = R5.sepmax(Ac[:n], Bc3[:n], Aa[:na], Ba3[:na], seed=900 + 10 * k + step)
            row = {"amp": amp, "z": r["z_max"], "best_test": r["best_test"],
                   "z_full": r["z_full"], "auc_full": r["auc_full"], "n": len(recs),
                   "detectors": {d: float(np.mean([x["detectors"][d] for x in recs]))
                                 for d in recs[0]["detectors"]},
                   "axis_R": float(np.mean([x["axis_R"] for x in recs])),
                   "axis_err": float(np.mean([x["axis_err"] for x in recs])),
                   "axis_proj": float(np.mean([x["axis_proj"] for x in recs])),
                   "axis_proj45": float(np.mean([x["axis_proj45"] for x in recs]))}
            rows.append(row)
            nz = sorted([x for x in rows if x["amp"] > 0], key=lambda x: x["amp"])
            print(f"  {uid} amp={amp:g} -> z={r['z_max']:.2f} ({r['best_test']})",
                  flush=True)
            step += 1
        rows = sorted(rows, key=lambda r: r["amp"])
        aa = [r["amp"] for r in rows]
        zz = [r["z"] for r in rows]
        v["rows"] = rows
        v["z5"] = st.threshold_amplitude(aa, zz, 5.0)
        v["z3"] = st.threshold_amplitude(aa, zz, 3.0)
        v["z_crit"] = st.threshold_amplitude(aa, zz, zc)
        v["responsiveness_z"] = st.responsiveness(aa, zz)
        v["fine_scan_steps_added"] = step
        print(f"  {uid}: threshold {v['z_crit'].get('amp')}", flush=True)

    with open(os.path.join(RES, "E3_amplitude_scans.json"), "w") as f:
        json.dump(E3, f, indent=1, default=float)
    gn.close_pool()
    pv.stop_ledger()
    print("rewrote E3_amplitude_scans.json")


if __name__ == "__main__":
    main()
