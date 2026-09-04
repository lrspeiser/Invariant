"""Items 1+2, fifth pass: the within-class gradient.

The transfer test asks whether a frozen model lands the held-out clusters at the
right LEVEL.  It does not ask whether the model tracks the right GRADIENT.  A
model can pass the first and fail the second, and if it does then it is doing
the job of a constant, not the job of a physical variable.

Measure beta inside each population separately -- one class, one instrument, one
pipeline, no label to lean on -- and inside the held-out set itself.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import common as C
from boundary import build_profiles, compute

RNG = np.random.default_rng(20260911)


def beta_within(t, mask, nb=8000):
    i = np.where(mask)[0]
    lg, lp, dev = t["lg"][i], t["lp"][i], t["dev"][i]
    A = np.column_stack([np.ones(len(i)), lg, lg ** 2, lp])
    c, *_ = np.linalg.lstsq(A, dev, rcond=None)
    b = []
    for _ in range(nb):
        p = RNG.integers(0, len(i), len(i))
        try:
            cc, *_ = np.linalg.lstsq(A[p], dev[p], rcond=None)
            b.append(cc[3])
        except np.linalg.LinAlgError:
            pass
    b = np.array(b)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return dict(n=len(i), beta=float(c[3]), q=float(2 * c[3]),
                ci95=[float(lo), float(hi)],
                p_positive=float((b > 0).mean()),
                lp_span=float(lp.max() - lp.min()))


def main():
    d = C.load_ladder()
    profs, _ = build_profiles(d)
    out = {}
    print("=" * 78)
    print("WITHIN-CLASS beta -- one class, one instrument, one pipeline, no label")
    print("=" * 78)
    for rule in ("TAIL", "BARY"):
        lp, _, _ = compute(d, profs, rule)
        t = C.system_table(d, lp_override=lp)
        print(f"\n   rule {rule}")
        print(f"   {'population':<24} {'n':>4} {'lp span':>8} {'beta':>9} "
              f"{'95%':>22} {'q':>8} {'P(b>0)':>7}")
        pops = (("galaxies (rung 1)", t["rank"] == 1),
                ("groups (rungs 2-4)", (t["rank"] >= 2) & (t["rank"] <= 4)),
                ("clusters (rungs 5-6)", t["rank"] >= 5),
                ("groups+clusters (2-6)", t["rank"] >= 2),
                ("all 252 pooled", np.ones(len(t["lg"]), bool)))
        out[rule] = {}
        for nm, m in pops:
            r = beta_within(t, m)
            out[rule][nm] = r
            print(f"   {nm:<24} {r['n']:4d} {r['lp_span']:8.2f} "
                  f"{r['beta']:+9.4f}   [{r['ci95'][0]:+.4f}, "
                  f"{r['ci95'][1]:+.4f}] {r['q']:+8.3f} {r['p_positive']:7.3f}")

    # the raw, uncontrolled split inside the held-out clusters
    lp, _, _ = compute(d, profs, "TAIL")
    t = C.system_table(d, lp_override=lp)
    i = np.where(t["rank"] >= 5)[0]
    lpv, dev, lgv = t["lp"][i], t["dev"][i], t["lg"][i]
    med = float(np.median(lpv))
    lo_, hi_ = lpv < med, lpv >= med
    dlp = float(np.median(lpv[hi_]) - np.median(lpv[lo_]))
    ddev = float(dev[hi_].mean() - dev[lo_].mean())
    print(f"\n   RAW split of the held-out clusters at the median "
          f"log|Phi_b| = {med:.3f}, no control at all:")
    print(f"      low  half n={int(lo_.sum())}  median log|Phi_b| "
          f"{np.median(lpv[lo_]):.3f}  mean deviation {dev[lo_].mean():+.4f} "
          f"+- {dev[lo_].std():.4f}")
    print(f"      high half n={int(hi_.sum())}  median log|Phi_b| "
          f"{np.median(lpv[hi_]):.3f}  mean deviation {dev[hi_].mean():+.4f} "
          f"+- {dev[hi_].std():.4f}")
    print(f"      observed slope {ddev:+.4f} / {dlp:+.4f} = {ddev / dlp:+.4f}"
          f"  ->  q = {2 * ddev / dlp:+.4f}")
    print(f"      the FROZEN beta = +0.1719 predicts a step of "
          f"{0.17188370232387992 * dlp:+.4f} dex across that span; observed "
          f"{ddev:+.4f} dex")
    out["raw_split_of_holdout"] = dict(
        median_lp=med, n_low=int(lo_.sum()), n_high=int(hi_.sum()),
        mean_dev_low=float(dev[lo_].mean()), mean_dev_high=float(dev[hi_].mean()),
        d_lp=dlp, d_dev=ddev, slope=float(ddev / dlp), q=float(2 * ddev / dlp),
        predicted_step_from_frozen_beta=float(0.17188370232387992 * dlp))
    out["caveat"] = (
        "The negative within-group and within-cluster beta must NOT be read as "
        "a measurement of gravity.  Run Z showed that for hydrostatic systems "
        "the shape factor S and the observable share the density log-slope, "
        "which drove beta to -0.400 on eFEDS and flipped it to +0.463 when the "
        "log-slope was controlled.  The safe reading is only that NO "
        "within-class measurement supports a positive beta of the published "
        "size, and that the galaxy value, which is not hydrostatic, is "
        "consistent with zero.")

    p = os.path.join(C.LANE, "ablation.json")
    js = json.load(open(p))
    js["within_class_beta"] = out
    json.dump(js, open(p, "w"), indent=2)
    print(f"\nmerged into {p}")


if __name__ == "__main__":
    main()
