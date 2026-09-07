"""stacked.py -- redo the modification search with enough signal to answer it.

WHY RUN BU HAD TO BE REDONE. Its conclusion -- "no family reduces the scatter,
so the residual is an amplitude not a dependence" -- was VACUOUS. Decomposing
the 0.523 dex scatter afterwards:

    observed scatter              0.523 dex
    median per-point measurement  0.365 dex
    points with >50% error        45 of 55
    points with <20% error        0
    intrinsic scatter             0.000 dex

The spread is entirely measurement error. No modification could have reduced it,
whatever the physics, so the test could not have returned any other answer. It
belongs with Run AY's "baryon-only control was VACUOUS, not passed", not with a
result.

WHAT FIXES IT is what built the galaxy RAR in the first place: stop asking each
point to carry the answer. The RAR is a STACKED relation -- 153 galaxies, 2693
points, binned -- and its 0.13 dex is the scatter of the RELATION, not of a
single measurement. Averaging n points beats the error down by sqrt(n), and with
55 points in 3-4 bins the per-bin error falls from 0.37 dex to roughly 0.1.

So the question changes shape. Not "what exponent fits each point" but:

    Binned by a candidate variable, does the residual MOVE across the bins by
    more than the binned errors allow?

That is a test with power, and it is the same test Milgrom's case rested on: not
that dark halos fit badly, but that the residual showed a regularity the halo
picture permits without requiring, and with a scatter too small for per-object
freedom to explain.

    python stacked.py
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

import cosmo as C                                              # noqa: E402
import modifications as MOD                                    # noqa: E402

OUT = os.path.join(HERE, "stacked.json")
NBIN = 4
N_PERM = 2000
SEED = 20260907

VARIABLES = [
    ("gas density", "rho", "log"),
    ("temperature kT", "kt", "lin"),
    ("radius", "R", "log"),
    ("redshift", "z", "lin"),
    ("enclosed gas mass", "m_gas", "log"),
    ("baryonic acceleration g_bar", "g_bar", "log"),
]


def weighted(vals, errs):
    """Inverse-variance mean of a ratio and its error, in linear space."""
    w = 1.0 / np.asarray(errs) ** 2
    m = float(np.sum(w * np.asarray(vals)) / np.sum(w))
    e = float(math.sqrt(1.0 / np.sum(w)))
    return m, e


def main():
    from holdout import loader                                 # noqa: E402
    loader.verify()

    rows = MOD.build_rows()
    clusters = sorted({r["cluster"] for r in rows})
    loader.assert_not_sealed(clusters, "stacked search")

    gb = np.array([r["g_bar"] for r in rows])
    dsb = np.array([r["ds_bar"] for r in rows])
    dso = np.array([r["ds_obs"] for r in rows])
    dse = np.array([r["ds_err"] for r in rows])
    pred = dsb * MOD.boost_rar(gb)
    keep = np.isfinite(pred) & (pred > 0) & np.isfinite(dso) & (dse > 0)

    # the residual: how much MORE lensing there is than the RAR predicts.
    # Kept in LINEAR space with its own error, because the errors are large
    # enough that log-space would bias the mean -- 45 of 55 points have a
    # fractional error above 50%, and log of a noisy positive quantity is not
    # unbiased there.
    ratio = (dso / pred)[keep]
    rerr = (dse / pred)[keep]
    rows = [r for r, k in zip(rows, keep) if k]

    all_m, all_e = weighted(ratio, rerr)
    print("residual (observed / RAR-predicted), all %d points stacked:" % len(ratio))
    print("   %.2f +- %.2f      -> the RAR under-predicts by this factor\n"
          % (all_m, all_e), flush=True)

    rng = np.random.default_rng(SEED)
    out = []
    print("  %-28s %-34s %s" % ("binned by", "residual per bin", "verdict"))
    for label, key, scale in VARIABLES:
        x = np.array([r[key] if r[key] is not None else np.nan for r in rows],
                     dtype=float)
        good = np.isfinite(x) & (x > 0 if scale == "log" else np.isfinite(x))
        if good.sum() < 16:
            print("  %-28s too few points (%d)" % (label, good.sum()))
            continue
        xs, rs, es = x[good], ratio[good], rerr[good]
        order = np.argsort(xs)
        chunks = np.array_split(order, NBIN)
        cells = []
        for c in chunks:
            if len(c) < 3:
                continue
            m, e = weighted(rs[c], es[c])
            cells.append(dict(n=int(len(c)), x_mid=float(np.median(xs[c])),
                              residual=m, err=e))
        if len(cells) < 3:
            continue

        # is the residual FLAT across the bins?
        v = np.array([c["residual"] for c in cells])
        ev = np.array([c["err"] for c in cells])
        wmean = float(np.sum(v / ev ** 2) / np.sum(1 / ev ** 2))
        chi2 = float(np.sum(((v - wmean) / ev) ** 2))
        dof = len(cells) - 1

        # permutation null: shuffle the binning variable, rebin, recompute chi2
        nulls = []
        for _ in range(N_PERM):
            p = rng.permutation(len(xs))
            o2 = np.argsort(xs[p])
            cc = []
            for c in np.array_split(o2, NBIN):
                if len(c) < 3:
                    continue
                cc.append(weighted(rs[p][c], es[p][c]))
            if len(cc) < 3:
                continue
            vv = np.array([a for a, _ in cc])
            ee = np.array([b for _, b in cc])
            wm = np.sum(vv / ee ** 2) / np.sum(1 / ee ** 2)
            nulls.append(float(np.sum(((vv - wm) / ee) ** 2)))
        nulls = np.array(nulls)
        p_val = float((nulls >= chi2).mean()) if len(nulls) else float("nan")

        verdict = ("VARIES with %s (p=%.3f)" % (label, p_val) if p_val < 0.05
                   else "flat, p=%.2f" % p_val)
        out.append(dict(variable=label, key=key, bins=cells, chi2=chi2, dof=dof,
                        p_permutation=p_val, flat_value=wmean, verdict=verdict))
        show = "  ".join("%.1f+-%.1f" % (c["residual"], c["err"]) for c in cells)
        print("  %-28s %-34s %s" % (label, show, verdict), flush=True)

    res = dict(lane="clusterfirst", stage="stacked",
               n_points=int(len(ratio)), n_clusters=len(clusters),
               residual_all=dict(value=all_m, err=all_e),
               scatter_note=("Run BU's per-point search was vacuous: intrinsic "
                             "scatter 0.000 dex, all 0.523 was measurement error, "
                             "45 of 55 points above 50% error. This stacks instead."),
               nbin=NBIN, n_perm=N_PERM, variables=out,
               sealed_untouched=len(loader.sealed_names()))
    io.open(OUT, "w", newline="\n", encoding="utf-8").write(
        json.dumps(res, indent=1) + "\n")
    print("\nwrote stacked.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
