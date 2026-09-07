"""learn.py -- does the lensing signal know anything about the baryon ASYMMETRY?

THE QUESTION, posed so it can come back no. Every previous cluster input to this
programme was azimuthally averaged, so a law that depends on direction was
indistinguishable from one that does not. This table keeps the angle. So:

    Does knowing how much brighter or hotter one DIRECTION is than the average
    at the same radius improve the prediction of the lensing signal there,
    beyond what radius alone already tells you?

THE DESIGN. Two nested models, both gradient-boosted, both cross-validated by
CLUSTER (never by row -- cells from one cluster are not independent, and a
row-wise split would let the model memorise a cluster in training and be graded
on it in test):

    RADIAL   R, z, kT              -- everything a profile could have told you
    FULL     + sb_asym, hardness_asym, n_phot   -- plus the asymmetry

The number that matters is the FULL model's improvement over RADIAL.

THE NULL THAT MATTERS. Permuting cluster labels is too easy to beat. The right
null permutes the AZIMUTHAL INDEX WITHIN each cluster and radial bin: every
radial quantity is untouched, every marginal distribution is identical, and only
the association between a direction's baryons and that direction's gravity is
destroyed. A purely radial model scores exactly the same under it. Only a
genuinely directional signal beats it.

If the improvement does not clear that null, the honest answer is that at this
depth the lensing cells cannot see the baryon asymmetry -- which is a real
result about what the data can carry, not a failure of the model.

    python learn.py
"""
from __future__ import annotations

import io
import json
import math
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "learn.json")

RADIAL = ["R_mpc", "z", "kT"]
#: n_phot was tried and carries NOTHING (gain -0.0002, z=+0.63, p=0.23);
#: it is a brightness/exposure proxy, not an asymmetry, and is excluded so
#: the measured gain cannot be attributed to it.
ASYM = ["sb_asym", "hardness_asym"]
N_PERM = 200
SEED = 20260907


def load():
    import csv
    rows = []
    with io.open(os.path.join(HERE, "table.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            def f(k):
                v = r.get(k, "")
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return np.nan
            if not r.get("g_t") or not r.get("g_err") or r.get("xray_covered") != "1":
                continue
            if not r.get("sb_asym"):
                continue
            rows.append(dict(cluster=r["cluster"], i_rad=int(r["i_rad"]),
                             j_az=int(r["j_az"]),
                             y=f("g_t"), err=f("g_err"),
                             R_mpc=f("R_mpc"), z=f("z"), kT=f("kT"),
                             sb_asym=f("sb_asym"),
                             hardness_asym=f("hardness_asym"),
                             n_phot=f("n_phot")))
    return [r for r in rows if np.isfinite(r["y"]) and r["err"] > 0]


def design(rows, cols):
    return np.column_stack([[r[c] for r in rows] for c in cols])


def wmse_cv(rows, cols, seed=SEED):
    """Group-wise CV. Returns weighted MSE against the weighted-mean baseline."""
    X = design(rows, cols)
    y = np.array([r["y"] for r in rows])
    w = np.array([1.0 / r["err"] ** 2 for r in rows])
    g = np.array([r["cluster"] for r in rows])
    pred = np.zeros_like(y)
    gkf = GroupKFold(n_splits=6)
    for tr, te in gkf.split(X, y, groups=g):
        m = HistGradientBoostingRegressor(
            max_depth=3, max_iter=200, learning_rate=0.05,
            min_samples_leaf=25, l2_regularization=1.0, random_state=seed)
        m.fit(X[tr], y[tr], sample_weight=w[tr])
        pred[te] = m.predict(X[te])
    base = np.sum(w * y) / np.sum(w)
    sse = np.sum(w * (y - pred) ** 2)
    sst = np.sum(w * (y - base) ** 2)
    return 1.0 - sse / sst, pred


def permute_azimuth(rows, rng):
    """Shuffle the asymmetry features between sectors, within cluster and radius."""
    out = [dict(r) for r in rows]
    idx = {}
    for i, r in enumerate(out):
        idx.setdefault((r["cluster"], r["i_rad"]), []).append(i)
    for _, group in idx.items():
        if len(group) < 2:
            continue
        perm = rng.permutation(len(group))
        vals = [(out[j]["sb_asym"], out[j]["hardness_asym"], out[j]["n_phot"])
                for j in group]
        for a, j in enumerate(group):
            out[j]["sb_asym"], out[j]["hardness_asym"], out[j]["n_phot"] = vals[perm[a]]
    return out


def permute_cluster(rows, rng):
    out = [dict(r) for r in rows]
    vals = [(r["sb_asym"], r["hardness_asym"], r["n_phot"]) for r in out]
    perm = rng.permutation(len(out))
    for a, r in enumerate(out):
        r["sb_asym"], r["hardness_asym"], r["n_phot"] = vals[perm[a]]
    return out


def main():
    sys.path.insert(0, os.path.dirname(HERE))
    from holdout import loader                                # noqa: E402
    loader.verify()

    rows = load()
    clusters = sorted({r["cluster"] for r in rows})
    loader.assert_not_sealed(clusters, "learn input")
    print("rows %d over %d clusters (sealed %d untouched)"
          % (len(rows), len(clusters), len(loader.sealed_names())), flush=True)

    # the cross component is a null on the data itself, not on the model
    gx_rows = [r for r in rows]
    r2_rad, _ = wmse_cv(rows, RADIAL)
    r2_full, _ = wmse_cv(rows, RADIAL + ASYM)
    gain = r2_full - r2_rad
    print("\n  radial-only  R^2 = %+.4f" % r2_rad)
    print("  + asymmetry  R^2 = %+.4f" % r2_full)
    print("  gain from the asymmetry features = %+.4f" % gain, flush=True)

    rng = np.random.default_rng(SEED)
    print("\n  running %d azimuthal-permutation nulls ..." % N_PERM, flush=True)
    null_az = []
    for k in range(N_PERM):
        pr = permute_azimuth(rows, rng)
        r2p, _ = wmse_cv(pr, RADIAL + ASYM, seed=SEED + k)
        null_az.append(r2p - r2_rad)
    null_az = np.array(null_az)

    print("  running %d cluster-permutation nulls ..." % N_PERM, flush=True)
    null_cl = []
    for k in range(N_PERM):
        pr = permute_cluster(rows, rng)
        r2p, _ = wmse_cv(pr, RADIAL + ASYM, seed=SEED + k)
        null_cl.append(r2p - r2_rad)
    null_cl = np.array(null_cl)

    p_az = float((null_az >= gain).mean())
    p_cl = float((null_cl >= gain).mean())
    z_az = float((gain - null_az.mean()) / null_az.std()) if null_az.std() > 0 else float("nan")

    # Deliberately cautious language. R^2 goes from about -0.01 to about +0.01:
    # the model explains essentially none of the variance, and z ~ 2.7 is a hint,
    # not a detection, in a field where 5 sigma is the bar and in a programme that
    # has already retired eight artefacts that looked at least this good.
    if p_az >= 0.05:
        verdict = "no directional information detectable at this depth"
    elif z_az < 3.0:
        verdict = ("WEAK POSITIVE, %.1f sigma: the asymmetry features clear the "
                   "azimuthal null, but R^2 stays near zero, so this is a lead to "
                   "pursue with more data -- NOT a detection" % z_az)
    else:
        verdict = ("directional signal at %.1f sigma against the azimuthal null; "
                   "effect size still small (R^2 %.4f)" % (z_az, r2_full))
    out = dict(
        lane="clusterjoint", n_rows=len(rows), n_clusters=len(clusters),
        features_radial=RADIAL, features_asymmetry=ASYM,
        r2_radial=r2_rad, r2_full=r2_full, gain=gain,
        null_azimuthal=dict(mean=float(null_az.mean()), sd=float(null_az.std()),
                            p=p_az, z=z_az, n=N_PERM),
        null_cluster=dict(mean=float(null_cl.mean()), sd=float(null_cl.std()),
                          p=p_cl, n=N_PERM),
        verdict=verdict,
        cv="GroupKFold by cluster -- never by row",
        sealed_untouched=len(loader.sealed_names()))
    io.open(OUT, "w", newline="\n", encoding="utf-8").write(
        json.dumps(out, indent=1) + "\n")

    print("\n  azimuthal null : mean %+.4f  sd %.4f  ->  p = %.3f   z = %+.2f"
          % (null_az.mean(), null_az.std(), p_az, z_az))
    print("  cluster null   : mean %+.4f  sd %.4f  ->  p = %.3f"
          % (null_cl.mean(), null_cl.std(), p_cl))
    print("\n  VERDICT: %s" % verdict)
    del gx_rows
    return 0


if __name__ == "__main__":
    sys.exit(main())
