"""run_library_axis.py -- an accidental alignment in Run BF's shared scene library.

Run BF draws every corpus from ONE library of 18 clusters, deliberately, so
that a separation cannot come from the scene prior.  The consequence for a
DIRECTIONAL statistic was not checked: the 18 (baryon axis, external axis)
pairs are fixed, so whatever correlation those 18 draws happen to have is
present in every corpus and does not average out.

This measures it.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import guard                                   # noqa: E402


def main():
    guard.start()
    from universes import generate as gn
    lib = gn.get_lib()
    pb = np.array([g.clu.pa_bar_deg for g in lib.geoms])
    ax = np.array([g.clu.axis_ext_deg for g in lib.geoms])
    d = np.deg2rad(2 * (pb - ax))
    n = len(pb)
    c = float(np.mean(np.cos(d)))
    sd = 1.0 / np.sqrt(2 * n)
    out = dict(n_clusters=n, mean_cos2_dphi=c, mean_sin2_dphi=float(np.mean(np.sin(d))),
               expected_sd_of_mean_if_independent=sd, z=c / sd,
               per_cluster_cos2=np.round(np.cos(d), 4).tolist(),
               note=("a baryon-aligned quadrupole of studentised size S "
                     "projects onto the external axis with mean S * "
                     f"{c:.3f} in EVERY corpus drawn from this library"))
    out["provenance"] = guard.stop()
    p = os.path.join(HERE, "results", "L_library_axis.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"mean cos2(pa_bar - axis_ext) = {c:+.4f}  ({c / sd:+.2f} sigma "
          f"from zero for n = {n})")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
