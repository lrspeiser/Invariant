"""Item 1, fourth pass: how stable is the arm-B result?

The first pass found that a RAR-only model fitted on the 36 groups predicts the
52 held-out clusters at 0.0871 dex -- better than the published potential-depth
model.  Before that becomes a headline it has to survive resampling of its own
36-system training set, because the groups span only 0.47 dex in log g_bar while
the clusters reach 0.8 dex beyond them, so a free quadratic is extrapolating.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import common as C

RNG = np.random.default_rng(20260910)


def main():
    d = C.load_ladder()
    t = C.system_table(d)
    n = len(t["lg"])
    gal = t["rank"] == 1
    grp = (t["rank"] >= 2) & (t["rank"] <= 4)
    clu = t["rank"] >= 5
    y = t["dev"]
    itr, ite = np.where(grp)[0], np.where(clu)[0]
    jtr = np.where(gal | grp)[0]

    print("=" * 78)
    print("ARM B STABILITY -- can 36 groups really beat the published models?")
    print("=" * 78)
    print(f"   training groups span log g_bar "
          f"[{t['lg'][grp].min():+.3f}, {t['lg'][grp].max():+.3f}] "
          f"= {t['lg'][grp].max() - t['lg'][grp].min():.3f} dex")
    print(f"   held-out clusters span        "
          f"[{t['lg'][clu].min():+.3f}, {t['lg'][clu].max():+.3f}] "
          f"and reach {t['lg'][clu].max() - t['lg'][grp].max():+.3f} dex beyond "
          f"the training range")

    cands = {
        "groups / M0 quadratic": (itr, np.column_stack(
            [np.ones(n), t["lg"], t["lg"] ** 2])),
        "groups / M0 linear": (itr, np.column_stack([np.ones(n), t["lg"]])),
        "groups / constant offset": (itr, np.ones((n, 1))),
        "galaxies+groups / M1": (jtr, C.design(t, "M1")),
        "galaxies+groups / M3": (jtr, C.design(t, "M3")),
        "galaxies+groups / M0": (jtr, C.design(t, "M0")),
    }
    out = {}
    print(f"\n   {'model':<26} {'point':>7} {'train-resampled 95%':>26} "
          f"{'median':>8}")
    for nm, (tr, A) in cands.items():
        c, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None)
        pt = math.sqrt(float(np.mean((y[ite] - A[ite] @ c) ** 2)))
        r = []
        for _ in range(8000):
            p = tr[RNG.integers(0, len(tr), len(tr))]
            try:
                cc, *_ = np.linalg.lstsq(A[p], y[p], rcond=None)
            except np.linalg.LinAlgError:
                continue
            r.append(math.sqrt(float(np.mean((y[ite] - A[ite] @ cc) ** 2))))
        r = np.array(r)
        lo, hi = np.percentile(r, [2.5, 97.5])
        out[nm] = dict(point_rms=pt, resampled_mean=float(r.mean()),
                       resampled_median=float(np.median(r)),
                       ci95=[float(lo), float(hi)],
                       n_train=int(len(tr)),
                       coef=[float(x) for x in c])
        print(f"   {nm:<26} {pt:7.4f}   [{lo:.4f}, {hi:.4f}] {np.median(r):8.4f}")

    # paired, object by object, frozen: the constant group offset vs M1 and M3
    print("\n   paired object bootstrap, frozen coefficients, same 52 clusters:")
    cc = float(y[grp].mean())
    ec = y[ite] - cc
    pair = {}
    for m in ("M1", "M3", "M0"):
        A = C.design(t, m)
        c, *_ = np.linalg.lstsq(A[jtr], y[jtr], rcond=None)
        e = y[ite] - A[ite] @ c
        obs = math.sqrt(float(np.mean(ec ** 2))) - math.sqrt(float(np.mean(e ** 2)))
        dd = np.empty(20000)
        for k in range(20000):
            p = RNG.integers(0, len(ite), len(ite))
            dd[k] = math.sqrt(float(np.mean(ec[p] ** 2))) \
                - math.sqrt(float(np.mean(e[p] ** 2)))
        lo, hi = np.percentile(dd, [2.5, 97.5])
        pair[m] = dict(observed_delta_rms=float(obs),
                       ci95=[float(lo), float(hi)],
                       p_constant_better=float((dd < 0).mean()),
                       n_objects_constant_better=int(
                           (np.abs(ec) < np.abs(e)).sum()))
        print(f"      constant group offset ({cc:+.4f} dex) vs published {m}: "
              f"dRMS {obs:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"P(constant better) = {pair[m]['p_constant_better']:.3f}  "
              f"closer on {pair[m]['n_objects_constant_better']}/52")
    out["paired_constant_vs_published"] = pair
    out["constant_group_offset_dex"] = cc
    out["training_lg_range_groups"] = [float(t["lg"][grp].min()),
                                       float(t["lg"][grp].max())]
    out["test_lg_range_clusters"] = [float(t["lg"][clu].min()),
                                     float(t["lg"][clu].max())]

    p = os.path.join(C.LANE, "ablation.json")
    js = json.load(open(p))
    js["arm_b_stability"] = out
    json.dump(js, open(p, "w"), indent=2)
    print(f"\nmerged into {p}")


if __name__ == "__main__":
    main()
