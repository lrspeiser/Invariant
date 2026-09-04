"""run_fillband.py -- sample the band where each scan is actually informative.

A scan z of 8.5 is the permutation-null cap, not a measurement, and a scan of
[2.2, 6.4, 8.5, 8.5, 8.5] carries only two usable points.  Reporting
"responsiveness not established" from a two-point fit would be a statement
about the sampling, not about the physics.

This pass adds three geometrically spaced amplitudes between the largest
sub-threshold point and the smallest saturated one, so every scan has at least
four unsaturated points and d z / d log10(amplitude) is measured rather than
guessed.  It changes no analysis, only the sampling of the knob.
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
from universes import provenance as pv        # noqa: E402
from universes import run_stage5 as R5        # noqa: E402
from universes import stats as st             # noqa: E402

RES = os.path.join(HERE, "results")
N = int(os.environ.get("N_AMP", 480))
ZSAT = 8.4
SEED = 6300000


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
        lo = max([r["amp"] for r in nz if r["z"] < zc], default=nz[0]["amp"])
        hi = min([r["amp"] for r in nz if r["z"] >= ZSAT], default=nz[-1]["amp"])
        if hi <= lo:
            print(f"  {uid}: band already sampled"); continue
        new = np.geomspace(lo, hi, 5)[1:-1]
        for m, amp in enumerate(new):
            amp = float(amp)
            recs = R5.pool(f"band_{uid}_{amp:.6g}", (uid, amp, 1.0, 1.0), N,
                           SEED + 40000 * k + 1500 * m)
            Ac, Aa = R5.quarters(recs)
            n, na = min(len(Ac), len(Bc3)), min(len(Aa), len(Ba3))
            r = R5.sepmax(Ac[:n], Bc3[:n], Aa[:na], Ba3[:na], seed=1700 + 10 * k + m)
            rows.append({"amp": amp, "z": r["z_max"], "best_test": r["best_test"],
                         "z_full": r["z_full"], "auc_full": r["auc_full"],
                         "n": len(recs),
                         "detectors": {d: float(np.mean([x["detectors"][d] for x in recs]))
                                       for d in recs[0]["detectors"]},
                         "axis_R": float(np.mean([x["axis_R"] for x in recs])),
                         "axis_err": float(np.mean([x["axis_err"] for x in recs])),
                         "axis_proj": float(np.mean([x["axis_proj"] for x in recs])),
                         "axis_proj45": float(np.mean([x["axis_proj45"] for x in recs]))})
            print(f"  {uid} amp={amp:.5g} -> z={r['z_max']:.2f} ({r['best_test']})",
                  flush=True)
        rows = sorted(rows, key=lambda r: r["amp"])
        aa = [r["amp"] for r in rows]
        zz = [r["z"] for r in rows]
        v["rows"] = rows
        v["z5"] = st.threshold_amplitude(aa, zz, 5.0)
        v["z3"] = st.threshold_amplitude(aa, zz, 3.0)
        v["z_crit"] = st.threshold_amplitude(aa, zz, zc)
        v["responsiveness_z"] = st.responsiveness(aa, zz)

    with open(os.path.join(RES, "E3_amplitude_scans.json"), "w") as f:
        json.dump(E3, f, indent=1, default=float)
    gn.close_pool(); pv.stop_ledger()
    print("rewrote E3_amplitude_scans.json")


if __name__ == "__main__":
    main()
