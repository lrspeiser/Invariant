"""run_g45.py -- the galaxy misspecified-axis control, scanned in amplitude.

A misspecified axis is a NULL detector: Run BF measured 0.005 +- 0.003 for a
45-degree error against 1.00 for the aligned case.  ``run_power.py`` carries
the cluster version of that control (S_45); this adds the galaxy version
(G_45) on the same tensor amplitude scan and appends it to P_power.json, so
the Stage 4 certificate for the galaxy statistic has a measured control lever
rather than an assertion.
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

import guard                                               # noqa: E402
import worker as W                                         # noqa: E402
from universes.stats import responsiveness                 # noqa: E402

RES = os.path.join(HERE, "results")
N = int(os.environ.get("N_G45", 300))


def main():
    guard.start()
    amps = [0.0, 0.0125, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    scan = {}
    for i, a in enumerate(amps):
        pool = W.run_batch([(("U05_tensor_axis", a, 1.0, 1.0),
                             13_400_000 + 4001 * i + j, 12) for j in range(N)])
        scan[str(a)] = {k: dict(mean=float(np.mean([r[k] for r in pool])),
                                sd=float(np.std([r[k] for r in pool])))
                        for k in ("G_ext", "G_45")}
        print(f"  A={a:<7g} G_ext {scan[str(a)]['G_ext']['mean']:8.2f} "
              f"G_45 {scan[str(a)]['G_45']['mean']:8.2f}", flush=True)
    W.close_pool()
    p = os.path.join(RES, "P_power.json")
    doc = json.load(io.open(p, encoding="utf-8"))
    doc["P6_G45_scan"] = scan
    doc["P6_G45_responsiveness"] = {
        k: responsiveness(amps, [scan[str(a)][k]["mean"] for a in amps])
        for k in ("G_ext", "G_45")}
    doc["P6_G45_provenance"] = guard.stop()
    with open(p, "w") as f:
        json.dump(doc, f, indent=1, default=float)
    print(f"appended P6_G45_* to {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
