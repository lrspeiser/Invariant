"""run_ngal.py -- does the galaxy statistic really scale as sqrt(N_gal)?

The galaxy m=3 channel is the sensitive one, and the sample size it would need
at Run BF's detectable tensor amplitude is an EXTRAPOLATION well beyond the 30
galaxies a corpus contains.  An extrapolation is only worth quoting if the law
it uses has been measured, so this scans the number of galaxies over the range
the shared scene library allows (10 to 45) and fits G_ext = k sqrt(N_gal).
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import guard                                # noqa: E402
import worker                               # noqa: E402
import worker as W                          # noqa: E402

RES = os.path.join(HERE, "results")
N = int(os.environ.get("N_NG", 300))


def main():
    guard.start()
    out = {"n": N, "arms": {}}
    for arm, knob in (("U05_fid", 0.5), ("U05_A0.1", 0.1), ("U03_mond", None),
                      ("U02_cdm", None)):
        uid = "U05_tensor_axis" if arm.startswith("U05") else (
            "U03_mond_scalar" if arm == "U03_mond" else "U02_cdm")
        rows = {}
        for i, ng in enumerate((10, 20, 30, 45)):
            worker.N_GAL = ng                       # parent-side default
            jobs = [((uid, knob, 1.0, 1.0), 16_700_000 + 9001 * i + j, 12)
                    for j in range(N)]
            # run SERIALLY so the parent-side N_GAL is the one used
            pool = [W.one(j) for j in jobs]
            pool = [p for p in pool if p is not None]
            v = np.array([p["G_ext"] for p in pool])
            rows[str(ng)] = dict(mean=float(v.mean()), sd=float(v.std()),
                                 n=len(v))
            print(f"  {arm:<10} n_gal={ng:<3d} G_ext = {v.mean():8.3f} "
                  f"+- {v.std():.3f}", flush=True)
        ns = np.array([10, 20, 30, 45], float)
        ms = np.array([rows[str(int(n))]["mean"] for n in ns])
        k = float(np.sum(np.sqrt(ns) * ms) / np.sum(ns))
        pred = k * np.sqrt(ns)
        out["arms"][arm] = dict(rows=rows, k_sqrtN=k,
                                max_frac_dev=float(np.max(np.abs(pred - ms)
                                                          / np.maximum(np.abs(ms), 1e-9))),
                                sd_mean=float(np.mean([rows[str(int(n))]["sd"]
                                                       for n in ns])))
    worker.N_GAL = 30
    out["provenance"] = guard.stop()
    p = os.path.join(RES, "N_ngal_scaling.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
