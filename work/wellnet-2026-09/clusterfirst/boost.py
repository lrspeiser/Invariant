"""boost.py -- the boost the lensing DEMANDS, against the boost the RAR predicts.

The residual (observed / RAR-predicted) is a ratio of two things that both vary,
which hides what is happening. Split it.

    required boost   = Delta_Sigma_observed / Delta_Sigma_baryonic
    RAR boost        = 1 / (1 - exp(-sqrt(g_bar/a0)))

The first is what the measurement says the gravity is, per unit baryonic mass.
The second is what the radial acceleration relation says it should be, with no
free parameter. Their ratio is the residual, but separately they say which of the
two is doing the moving.

Three outcomes, each meaning something different:

    required ~= RAR          the RAR works on clusters, nothing here
    required ~= 1            gravity is NEWTONIAN on the gas alone -- no boost,
                             no dark matter, at radii where the RAR demands 5-10x
    required >  RAR          more gravity than the RAR gives (the classic cluster
                             missing-mass problem)

WHY THIS IS THE RIGHT CUT. The gas-shape explanation for the outer decline is
already refuted: fitting Vikhlinin's outer break to the same Chandra surface
brightness (`gas_truncated.py`) changes the radial slope from -2.38 to -2.34 per
dex, and moves it the WRONG WAY. Truncating the gas lowers Sigma(R) more than it
lowers the mean Sigma inside R, so Delta_Sigma_bar RISES and the predicted signal
gets larger, not smaller. Whatever produces the decline is not the gas model.

The remaining candidate is the boost itself, and this file measures it.

    python boost.py
"""
from __future__ import annotations

import io
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WELLNET = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WELLNET, "clustershear"))
sys.path.insert(0, HERE)
sys.path.insert(0, WELLNET)

import cosmo as C                                               # noqa: E402
import modifications as MOD                                     # noqa: E402
import firstprinciples as FP                                    # noqa: E402

OUT = os.path.join(HERE, "boost.json")
EDGES = [0.3, 0.6, 1.0, 1.6, 2.5, 4.6]                          # Mpc


def wmean(v, e):
    w = 1.0 / e ** 2
    return float(np.sum(w * v) / np.sum(w)), float(math.sqrt(1.0 / np.sum(w)))


def main():
    from holdout import loader                                  # noqa: E402
    loader.verify()

    os.environ.pop("R_WINDOW_MPC", None)
    rows = MOD.build_rows()
    loader.assert_not_sealed(sorted({r["cluster"] for r in rows}), "boost test")

    R = np.array([r["R"] for r in rows]) / C.MPC
    gb = np.array([r["g_bar"] for r in rows])
    dsb = np.array([r["ds_bar"] for r in rows])
    dso = np.array([r["ds_obs"] for r in rows])
    dse = np.array([r["ds_err"] for r in rows])
    cl = np.array([r["cluster"] for r in rows])

    keep = np.isfinite(dsb) & (dsb > 0) & np.isfinite(dso) & (dse > 0)
    R, gb, dsb, dso, dse, cl = (R[keep], gb[keep], dsb[keep],
                                dso[keep], dse[keep], cl[keep])

    req = dso / dsb                                             # boost demanded
    req_e = dse / dsb
    rar = MOD.boost_rar(gb)                                     # boost predicted

    print("the boost the lensing demands, against the boost the RAR predicts")
    print("")
    print("  %-13s %-5s %-11s %-16s %-9s %-9s %s"
          % ("shell Mpc", "n", "g_bar/a0", "required boost", "RAR pred",
             "Newtonian", "verdict"))

    shells = []
    for i in range(len(EDGES) - 1):
        m = (R >= EDGES[i]) & (R < EDGES[i + 1])
        if m.sum() < 3:
            continue
        v, e = wmean(req[m], req_e[m])
        w = 1.0 / req_e[m] ** 2
        rbar = float(np.sum(w * rar[m]) / np.sum(w))
        gbar_a0 = float(np.sum(w * gb[m]) / np.sum(w)) / FP.A0
        s_rar = (v - rbar) / e                                  # sigma from RAR
        s_new = (v - 1.0) / e                                   # sigma from Newton
        if abs(s_rar) < 2 and abs(s_new) >= 2:
            verdict = "RAR"
        elif abs(s_new) < 2 and abs(s_rar) >= 2:
            verdict = "NEWTONIAN"
        elif abs(s_rar) < 2 and abs(s_new) < 2:
            verdict = "cannot tell"
        else:
            verdict = "NEITHER"
        shells.append(dict(lo=EDGES[i], hi=EDGES[i + 1], n_points=int(m.sum()),
                           n_clusters=len(set(cl[m])), gbar_over_a0=gbar_a0,
                           required=v, required_err=e, rar_predicted=rbar,
                           sigma_from_rar=float(s_rar),
                           sigma_from_newton=float(s_new), verdict=verdict))
        print("  %-13s %-5d %-11.3f %5.2f +- %-8.2f %-9.2f %-9.1f %s  (%+.1f/%+.1f)"
              % ("%.1f - %.1f" % (EDGES[i], EDGES[i + 1]), m.sum(), gbar_a0,
                 v, e, rbar, 1.0, verdict, s_rar, s_new))

    print("")
    print("  columns: sigma from RAR / sigma from Newtonian")
    print("")
    n_new = sum(1 for s in shells if s["verdict"] == "NEWTONIAN")
    n_rar = sum(1 for s in shells if s["verdict"] == "RAR")
    if n_new and n_rar:
        print("  -> THE ANSWER CHANGES WITH RADIUS. %d shell(s) sit on the RAR and"
              % n_rar)
        print("     %d sit on the NEWTONIAN line, in the same clusters." % n_new)
    elif n_new:
        print("  -> Newtonian on the gas alone, where the RAR demands a large boost.")
    elif n_rar:
        print("  -> consistent with the RAR throughout.")

    out = dict(lane="clusterfirst", stage="boost-vs-radius",
               gas_file=os.environ.get("GAS_FILE", "gas_extended.json"),
               a0=FP.A0, edges_mpc=EDGES, shells=shells,
               note=("required = DS_obs/DS_bar is what the lensing says the "
                     "gravity is per unit baryon; RAR is what the radial "
                     "acceleration relation predicts with no free parameter."),
               caveat="g_bar is gas only; no stellar term",
               sealed_untouched=len(loader.sealed_names()))
    io.open(OUT, "w", newline="\n", encoding="utf-8").write(
        json.dumps(out, indent=1) + "\n")
    print("")
    print("wrote boost.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
