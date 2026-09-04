"""Which survey is actually setting beta?  Decompose the likelihood.

The joint fit with every strong-lens image system entered separately returned
beta = -0.10, while eFEDS alone returned -0.40.  Something with 49 points was
overruling something with 3365.  This splits -2 ln L by survey at the two
values so the answer is a number rather than a suspicion.

Writes decompose_results.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common as K                                              # noqa: E402
import fitlib as F                                              # noqa: E402

BETAS = (-0.40, -0.35, -0.10)


def best_at(J, beta, n_amp):
    sh = np.array([beta])
    fn = lambda p: J.m2lnL("H_R", sh, p)                        # noqa: E731
    best = None
    for s0 in (np.zeros(n_amp + 3), np.array([0.35, -0.1, 0.15, 0.2]),
               np.array([0.26, 0.0, 0.25, 0.9]),
               np.array([0.65, -0.2, 0.1, 0.85])):
        if len(s0) != n_amp + 3:
            continue
        r = minimize(fn, s0, method="Nelder-Mead",
                     options=dict(maxiter=8000, xatol=1e-7, fatol=1e-7))
        r = minimize(fn, r.x, method="Nelder-Mead",
                     options=dict(maxiter=8000, xatol=1e-8, fatol=1e-8))
        if best is None or r.fun < best[0]:
            best = (float(r.fun), r.x)
    tot, parts = J.m2lnL("H_R", sh, best[1], want_parts=True)
    return float(tot), {k: float(v) for k, v in parts.items()}


def main():
    out = {}
    for tag, agg in (("uncollapsed", False), ("aggregated", True)):
        bd = K.Bundle(verbose=False, r500_mode="cat", sl_agg=agg)
        J0 = F.Joint(bd)
        sc = F.unpack("H_P", J0.fit("H_P"))
        fs = (sc["sigma_int_locuss"], sc["sigma_int_sl_within"],
              sc["sigma_int_sl_cluster"])
        J = F.Joint(bd, fixed_scatter=fs, cache=J0._cache)
        rows = {}
        print(f"\n   {tag}  (SL rows = {len(bd.sl_rows)})")
        print(f"      {'beta':>6s} {'total':>10s} {'efeds':>10s} "
              f"{'locuss':>9s} {'sl':>10s} {'prior':>8s}")
        for b in BETAS:
            tot, parts = best_at(J, b, 1)
            rows[str(b)] = dict(total=tot, **parts)
            print(f"      {b:+6.2f} {tot:10.2f} {parts['efeds']:10.2f} "
                  f"{parts['locuss']:9.2f} {parts['sl']:10.2f} "
                  f"{parts['prior']:8.2f}")
        a, c = rows["-0.4"], rows["-0.1"]
        out[tag] = dict(rows=rows,
                        efeds_prefers_steep_by=c["efeds"] - a["efeds"],
                        sl_prefers_shallow_by=a["sl"] - c["sl"],
                        locuss_prefers_shallow_by=a["locuss"] - c["locuss"],
                        prior_prefers_steep_by=c["prior"] - a["prior"],
                        n_sl_rows=len(bd.sl_rows))
        print(f"      -> eFEDS prefers beta = -0.40 by "
              f"{out[tag]['efeds_prefers_steep_by']:.2f}; the strong-lens term"
              f" prefers -0.10 by {out[tag]['sl_prefers_shallow_by']:.2f}")
    with open(os.path.join(HERE, "decompose_results.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\n   wrote decompose_results.json")


if __name__ == "__main__":
    main()
