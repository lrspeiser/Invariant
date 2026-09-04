"""Item 3, addendum.

The +1.0000 correlation between the (g_bar, r)-controlled residual in
log|DeltaPhi_b| and the residual in log S is an ALGEBRAIC IDENTITY, not a
finding: S is defined as |DeltaPhi_b|/(g_bar r), so log|DeltaPhi_b| - log g_bar
- log r IS log S, for every boundary rule, always.  What the boundary rule
actually changes is the numerical content of S -- its spread, and therefore how
much independent information the variable carries.  Measure that.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import common as C
from boundary import RULES, build_profiles, compute


def within_bin_spread(lg, lp, nbin=8, minn=20):
    edges = np.percentile(lg, np.linspace(0, 100, nbin + 1))
    sds = []
    for i in range(nbin):
        m = (lg >= edges[i]) & (lg < edges[i + 1] if i < nbin - 1
                                else lg <= edges[i + 1])
        if m.sum() >= minn:
            sds.append(np.std(lp[m]))
    return float(np.median(sds))


def main():
    d = C.load_ladder()
    profs, _ = build_profiles(d)
    p = os.path.join(C.LANE, "boundary_sensitivity.json")
    res = json.load(open(p))
    add = {}
    print("=" * 78)
    print("ITEM 3 addendum -- what the boundary rule actually changes")
    print("=" * 78)
    print(f"\n{'rule':<6} {'sd log S':>9} {'log S min':>10} {'log S max':>10} "
          f"{'leverage':>9} {'R2 quad':>8} {'resid sd':>9} "
          f"{'err reduction vs RAR':>21}")
    for rule in RULES:
        lp, rref, bk = compute(d, profs, rule)
        ok = np.isfinite(lp)
        lS = lp[ok] - d["lg"][ok] - np.log10(d["r_kpc"][ok] * C.KPC)
        lev = within_bin_spread(d["lg"][ok], lp[ok])
        # collinearity of the row-level variable on (log g_bar, log r)
        x, y = d["lg"][ok], d["lr"][ok]
        A = np.column_stack([np.ones(ok.sum()), x, y, x ** 2, y ** 2, x * y])
        c, *_ = np.linalg.lstsq(A, lp[ok], rcond=None)
        r_ = lp[ok] - A @ c
        r2 = 1 - np.var(r_) / np.var(lp[ok])
        r0 = res["rules"][rule]["transfer_rms_M0"]
        r1 = res["rules"][rule]["transfer_rms_M1"]
        r3 = res["rules"][rule]["transfer_rms_M3"]
        rec = dict(sd_log_S=float(lS.std()), log_S_min=float(lS.min()),
                   log_S_max=float(lS.max()),
                   median_within_gbar_bin_sd=lev,
                   r2_quadratic_on_lg_lr=float(r2),
                   residual_sd_dex=float(r_.std()),
                   error_reduction_vs_RAR_M1=float(1 - r1 / r0),
                   error_reduction_vs_RAR_M3=float(1 - r3 / r0),
                   q_implied=res["rules"][rule]["q_implied"],
                   q_required_for_cluster_excess=0.3706163299042279,
                   q_implied_over_required=float(
                       res["rules"][rule]["q_implied"] / 0.3706163299042279))
        add[rule] = rec
        print(f"{rule:<6} {lS.std():9.4f} {lS.min():10.4f} {lS.max():10.4f} "
              f"{lev:9.4f} {r2:8.4f} {r_.std():9.4f} "
              f"{1 - r1 / r0:20.1%}")
    res["rule_information_content"] = add

    print(f"\n{'rule':<6} {'q implied':>10} {'q/q_required':>13} "
          f"{'M1 reduction':>13} {'M3 reduction':>13}")
    for rule in RULES:
        a = add[rule]
        print(f"{rule:<6} {a['q_implied']:+10.4f} "
              f"{a['q_implied_over_required']:13.2f} "
              f"{a['error_reduction_vs_RAR_M1']:12.1%} "
              f"{a['error_reduction_vs_RAR_M3']:12.1%}")

    print("\nThe identity is definitional.  log|DeltaPhi_b| = log g_bar + log r")
    print("+ log S holds for EVERY rule because S is defined as that ratio, so")
    print("corr = +1.0000 under all four is a tautology and not evidence about")
    print("any of them.  The rule-dependent quantity is sd(log S), which is the")
    print("entire information content of the variable beyond (g_bar, r):")
    sds = {r: add[r]["sd_log_S"] for r in RULES}
    print("   " + ", ".join(f"{k} {v:.4f}" for k, v in sds.items()))
    res["sd_log_S_by_rule"] = sds
    res["sd_log_S_range"] = float(max(sds.values()) - min(sds.values()))

    json.dump(res, open(p, "w"), indent=2)
    print(f"\nmerged into {p}")


if __name__ == "__main__":
    main()
