"""universality.py -- is the cluster residual one number, and if not, what is it a function of?

THE QUESTION, posed the way Milgrom's was. He did not fit a better curve. He
asked what single modification produces flat rotation curves AND Tully-Fisher at
once, and noticed that dark halos permit both while predicting neither, needing
three free numbers per galaxy where the data needed none. The decisive property
was UNIVERSALITY: one constant, no per-object freedom, and residual scatter
consistent with measurement error.

So for clusters the question is not "what modification closes the gap". It is:

    Is the residual ONE NUMBER across clusters?

If yes, it is an amplitude -- dark matter with a near-constant baryon fraction,
or a shifted acceleration scale -- and there is nothing further to find in it.
If no, it is a function of something, and the something must be a property
clusters have and galaxy rotation curves do not.

WHAT WENT WRONG BEFORE. `stacked.py` reported every variable "flat", three of
them at p = 1.00 -- the real binning more consistent than all 2000 permutations,
which is not a result, it is a broken null. The null permuted POINTS across
clusters. Real bins hold several correlated points from one cluster; permuted
bins mix clusters and scatter more, inflating the null chi-square and making real
data look falsely flat. Permuting CLUSTER LABELS, keeping each cluster's points
together, fixes it.

With that fixed the ten per-cluster residuals give chi-square 22.0 on 9 degrees
of freedom, p = 0.009. NOT one number.

THE SHARED-QUANTITY CHECK, run before any correlation is quoted, because two of
this programme's ten artefacts were exactly this. The residual is obs/pred and
pred depends on g_bar, which is built from the gas density. So any candidate
variable also built from that density shares a term with the residual:

    kT              INDEPENDENT   eRASS1 MBProj2D, unrelated to ACCEPT's n_e
    redshift        INDEPENDENT
    beta            SHARED        it is the gas model's own outer slope
    gas mass        SHARED        it IS the numerator of g_bar
    entropy         SHARED        K = kT / n_e^(2/3)
    cooling time    SHARED        depends on n_e

Only the independent ones can carry a correlation that means anything. The
shared ones are reported alongside, labelled, so the difference is visible.

    python universality.py
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

import modifications as MOD                                    # noqa: E402

OUT = os.path.join(HERE, "universality.json")
N_PERM = 20000
SEED = 20260907

# variable -> (label, shares a construction term with the residual?)
VARS = [
    ("kt", "X-ray temperature kT", False),
    ("z", "redshift", False),
    ("beta", "gas outer slope beta", True),
    ("m_gas_1mpc", "gas mass inside 1 Mpc", True),
    ("n_shear", "background sources (a noise proxy, not physics)", False),
]


def per_cluster(rows):
    """Inverse-variance residual per cluster, with its error."""
    gb = np.array([r["g_bar"] for r in rows])
    dsb = np.array([r["ds_bar"] for r in rows])
    dso = np.array([r["ds_obs"] for r in rows])
    dse = np.array([r["ds_err"] for r in rows])
    pred = dsb * MOD.boost_rar(gb)
    keep = np.isfinite(pred) & (pred > 0) & np.isfinite(dso) & (dse > 0)
    out = {}
    for r, p, o, e, k in zip(rows, pred, dso, dse, keep):
        if not k:
            continue
        out.setdefault(r["cluster"], []).append((o / p, e / p))
    res = {}
    for c, vs in out.items():
        v = np.array([a for a, _ in vs])
        er = np.array([b for _, b in vs])
        w = 1.0 / er ** 2
        res[c] = (float(np.sum(w * v) / np.sum(w)),
                  float(math.sqrt(1.0 / np.sum(w))), len(v))
    return res


def weighted_pearson(x, y, w):
    mx = np.sum(w * x) / np.sum(w)
    my = np.sum(w * y) / np.sum(w)
    cx, cy = x - mx, y - my
    den = math.sqrt(np.sum(w * cx ** 2) * np.sum(w * cy ** 2))
    return float(np.sum(w * cx * cy) / den) if den > 0 else float("nan")


def main():
    from holdout import loader                                 # noqa: E402
    loader.verify()

    rows = MOD.build_rows()
    res = per_cluster(rows)
    loader.assert_not_sealed(list(res), "universality test")

    ext = json.load(io.open(os.path.join(HERE, "gas_extended.json"), encoding="utf-8"))
    match = {m["erass"]: m for m in json.load(
        io.open(os.path.join(HERE, "accept_overlap.json"), encoding="utf-8"))}

    names = sorted(res, key=lambda c: -res[c][0])
    val = np.array([res[c][0] for c in names])
    err = np.array([res[c][1] for c in names])
    w = 1.0 / err ** 2
    grand = float(np.sum(w * val) / np.sum(w))
    chi2 = float(np.sum(((val - grand) / err) ** 2))
    dof = len(names) - 1

    # permutation p for "one universal value": scatter the residuals at their
    # own errors about the grand mean and see how often chi2 is exceeded
    rng = np.random.default_rng(SEED)
    sim = ((rng.normal(grand, err[None, :], size=(N_PERM, len(err))) - grand)
           / err[None, :]) ** 2
    p_uni = float((sim.sum(axis=1) >= chi2).mean())

    print("residual per cluster (observed / RAR-predicted)\n")
    print("  %-24s %-14s %-6s %-7s %s" % ("cluster", "residual", "kT", "z", "bins"))
    table = []
    for c in names:
        m = match[c]
        g = ext[c]
        try:
            kt = float(m.get("kt"))
        except (TypeError, ValueError):
            kt = float("nan")
        rowd = dict(cluster=c, residual=res[c][0], err=res[c][1], n_bins=res[c][2],
                    kt=kt, z=g["z"], beta=g["beta"], n_shear=m["shear"],
                    m_gas_1mpc=float(g["n0"] * g["rc_mpc"] ** 3))
        table.append(rowd)
        print("  %-24s %5.2f +- %-5.2f %-6s %-7.3f %d"
              % (c, res[c][0], res[c][1],
                 ("%.2f" % kt) if np.isfinite(kt) else "-", g["z"], res[c][2]))

    print("\n  grand mean %.2f   chi2 %.1f / %d dof   p(one universal value) = %.4f"
          % (grand, chi2, dof, p_uni))
    print("  -> %s\n" % ("NOT one number -- the residual varies between clusters"
                         if p_uni < 0.05 else
                         "consistent with a single universal amplitude"))

    print("  what does it track?  (shared = shares a construction term with g_bar)")
    print("  %-46s %-7s %-8s %s" % ("variable", "r_w", "p", "status"))
    corr = []
    for key, label, shared in VARS:
        x = np.array([t[key] for t in table], dtype=float)
        good = np.isfinite(x) & np.isfinite(val)
        if good.sum() < 6:
            continue
        xs = np.log10(np.abs(x[good])) if key in ("m_gas_1mpc", "n_shear") else x[good]
        r = weighted_pearson(xs, val[good], w[good])
        nul = []
        for _ in range(N_PERM // 10):
            r2 = weighted_pearson(xs, rng.permutation(val[good]), w[good])
            if np.isfinite(r2):
                nul.append(abs(r2))
        pv = float((np.array(nul) >= abs(r)).mean()) if nul else float("nan")
        corr.append(dict(variable=label, key=key, shares_term=shared,
                         r_weighted=r, p_permutation=pv, n=int(good.sum())))
        print("  %-46s %+6.2f  %-8.3f %s"
              % (label, r, pv,
                 "SHARED - not admissible" if shared else
                 ("independent" if pv >= 0.05 else "independent, p<0.05")))

    out = dict(lane="clusterfirst", stage="universality",
               n_clusters=len(names), grand_mean=grand, chi2=chi2, dof=dof,
               p_one_universal_value=p_uni, clusters=table, correlations=corr,
               null_note=("stacked.py's flatness null permuted POINTS across "
                          "clusters, breaking within-cluster correlation and "
                          "inflating the null chi2 -- three variables came back "
                          "p=1.00. This permutes cluster labels instead."),
               sealed_untouched=len(loader.sealed_names()))
    io.open(OUT, "w", newline="\n", encoding="utf-8").write(
        json.dumps(out, indent=1) + "\n")
    print("\nwrote universality.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
