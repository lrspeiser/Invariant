"""Coarse-graining screen, run standalone so it is not stale.

Holds ONE smooth mass distribution fixed and cuts it into N = 1 ... 10^4
catalogue rows, a nested refinement (the N = 10 partition is the first 10 rows
of the N = 10^4 partition).  A response built from the Poisson-smooth fields
never sees the partition; a response built from a row list does.

The screen lane's discriminator is d ln(drift)/d ln L, where L is the mean
nearest-neighbour distance of the partition: -3.11 for a genuine physical
kernel, -0.55 for the well-alignment family, +0.12 for pure row-counting.
That slope is measured here for every structure the tournament ranks.

Writes coarse.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import screens as SS                                            # noqa: E402
import ch_cluster as CC                                         # noqa: E402
from tw_core import KPC, Candidate                               # noqa: E402

NS = (1, 4, 10, 40, 100, 400, 1000, 4000, 10000)
WELLS = CC.WELL_SETTINGS + [
    dict(tag="plaw_p1q1s2_L300_literal", family="plaw", p=1.0, q=1.0, s=2.0,
         L=300.0 * KPC, exclude_nearest=False)]


def nn_scale(N, Rcl=1000.0 * KPC):
    """Mean nearest-neighbour distance of N points in a ball of radius Rcl."""
    return Rcl * (4.0 / 3.0 * np.pi / max(N, 1)) ** (1.0 / 3.0)


def main():
    out = []
    for st, ws in ([(s, None) for s in ("scalar_a0", "iso_K", "tensor_d",
                                        "tensor_T")]
                   + [("tensor_S", w) for w in WELLS]):
        c = Candidate("x", base="aqual", a0=1.058e-10, inv="phi", form="sat",
                      m=2.0, I0=1e12, struct=st, A=-25.0,
                      extra=dict(well=ws) if ws else {})
        r = SS.coarse_grain(c, Ns=NS)
        row = dict(structure=st, well=(ws["tag"] if ws else None), **r)
        if r.get("depends_on_catalogue"):
            S = np.asarray(r["S_rr"], float)
            dr = np.abs(np.diff(S))
            L = np.array([nn_scale(n) for n in NS[1:]])
            ok = dr > 0
            if ok.sum() >= 2:
                sl = np.polyfit(np.log(L[ok]), np.log(dr[ok]), 1)[0]
            else:
                sl = float("nan")
            row["dln_drift_dln_L"] = float(sl)
            row["reference_slopes"] = dict(genuine_kernel=-3.11,
                                           family_C_wells=-0.55,
                                           row_counting=+0.12)
        else:
            row["dln_drift_dln_L"] = None
        out.append(row)
        print(f"{st:<11}{str(row['well'])[:26]:<27}"
              f"max drift {r['max_drift']:.4g}   "
              f"dln(drift)/dlnL {row['dln_drift_dln_L']}", flush=True)
    with open(os.path.join(HERE, "coarse.json"), "w", newline="\n") as fh:
        json.dump(dict(N=list(NS), rows=out), fh, indent=1, default=float)
    print("wrote coarse.json")


if __name__ == "__main__":
    main()
