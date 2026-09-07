"""radial.py -- the residual as a function of physical radius, in matched shells.

WHERE THIS CAME FROM. The per-cluster universality test said the cluster residual
is not one number (p = 0.018). It is not: that verdict was an artefact of the
aperture. Both the shear cone and the Chandra field are ANGULAR, so the physical
radius each cluster is measured at is set by its distance --

    weighted-mean radius   0.62 Mpc (z = 0.069)  to  3.42 Mpc (z = 0.540)
    corr(redshift, weighted-mean radius) = +0.98

At that collinearity the ten residuals are not ten measurements of one quantity,
they are ten measurements at ten different radii. Comparing them to each other
tests radius, not clusters. Restricting every cluster to a matched physical
window returns p = 0.11 (0.8-1.8 Mpc) and p = 0.28 (0.8-2.4 Mpc): consistent with
one number, as soon as the number means the same thing everywhere.

WHAT THAT LEAVES. The matched windows do not agree on the VALUE:

    0.8 - 1.8 Mpc    2.01
    0.8 - 2.4 Mpc    1.45
    all radii        1.20

The excess over the RAR prediction is largest close in and falls outward. Those
windows overlap, so the trend has to be measured in shells that do not.

THE TEST. Every (cluster, bin) point is placed in a non-overlapping physical
shell, and the inverse-variance-weighted residual is taken per shell. The null
is a radial trend of zero, and it is evaluated by permuting CLUSTER LABELS, not
points -- points from one cluster are correlated, and permuting them across
clusters inflates the null and manufactures flatness. That error produced three
p = 1.00 results in `stacked.py` and is the reason this file permutes labels.

WHAT WOULD MAKE THIS UNINTERESTING, stated before looking:

  - g_bar here counts GAS ONLY. Stars are not in it. Adding them raises g_bar
    and lowers the residual, most at small radius where the BCG lives, which is
    the same direction as the trend being measured. The shells inside ~0.5 Mpc
    cannot be read without a stellar term.
  - a residual near 1 everywhere would mean the RAR already works and there is
    nothing here.
  - a residual that is a constant above 1 is an amplitude -- a baryon fraction
    or a shifted a0 -- and not a new dependence.

Only a trend that survives the shared-quantity check and the label permutation
is evidence of a radial dependence.

    python radial.py
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

OUT = os.path.join(HERE, "radial.json")
N_PERM = 20000
SEED = 20260907
EDGES = [0.3, 0.6, 1.0, 1.6, 2.5, 4.6]                          # Mpc


def wmean(v, e):
    w = 1.0 / e ** 2
    return float(np.sum(w * v) / np.sum(w)), float(math.sqrt(1.0 / np.sum(w)))


def slope(r, v, e):
    """Weighted least-squares slope of residual against log10 radius."""
    x = np.log10(r)
    w = 1.0 / e ** 2
    sw = np.sum(w)
    mx = np.sum(w * x) / sw
    my = np.sum(w * v) / sw
    den = np.sum(w * (x - mx) ** 2)
    return float(np.sum(w * (x - mx) * (v - my)) / den) if den > 0 else float("nan")


def main():
    from holdout import loader                                  # noqa: E402
    loader.verify()

    os.environ.pop("R_WINDOW_MPC", None)                        # shells, not a window
    rows = MOD.build_rows()
    loader.assert_not_sealed(sorted({r["cluster"] for r in rows}), "radial shell test")

    R = np.array([r["R"] for r in rows]) / C.MPC
    gb = np.array([r["g_bar"] for r in rows])
    dsb = np.array([r["ds_bar"] for r in rows])
    dso = np.array([r["ds_obs"] for r in rows])
    dse = np.array([r["ds_err"] for r in rows])
    cl = np.array([r["cluster"] for r in rows])

    pred = dsb * MOD.boost_rar(gb)
    keep = np.isfinite(pred) & (pred > 0) & np.isfinite(dso) & (dse > 0)
    R, dso, dse, pred, cl = R[keep], dso[keep], dse[keep], pred[keep], cl[keep]
    val, err = dso / pred, dse / pred

    print("residual (observed / RAR-predicted) in NON-OVERLAPPING physical shells")
    print("")
    print("  %-16s %-6s %-6s %-16s %s"
          % ("shell Mpc", "n_pts", "n_cl", "residual", "sigma from 1"))
    shells = []
    for i in range(len(EDGES) - 1):
        m = (R >= EDGES[i]) & (R < EDGES[i + 1])
        if m.sum() < 3:
            continue
        v, e = wmean(val[m], err[m])
        ncl = len(set(cl[m]))
        shells.append(dict(lo=EDGES[i], hi=EDGES[i + 1], n_points=int(m.sum()),
                           n_clusters=ncl, residual=v, err=e,
                           r_eff=float(np.average(R[m], weights=1 / err[m] ** 2))))
        print("  %-16s %-6d %-6d %5.2f +- %-8.2f %+.1f"
              % ("%.1f - %.1f" % (EDGES[i], EDGES[i + 1]), m.sum(), ncl, v, e,
                 (v - 1.0) / e))

    # trend, with the null that permutes CLUSTER LABELS
    r_eff = np.array([s["r_eff"] for s in shells])
    v_s = np.array([s["residual"] for s in shells])
    e_s = np.array([s["err"] for s in shells])
    obs_slope = slope(r_eff, v_s, e_s)

    rng = np.random.default_rng(SEED)
    names = sorted(set(cl))
    idx = {c: np.where(cl == c)[0] for c in names}
    null = []
    for _ in range(N_PERM // 10):
        perm = rng.permutation(names)
        Rp = R.copy()
        for a, b in zip(names, perm):                           # move each cluster's
            Rp[idx[a]] = R[idx[b]][:len(idx[a])] if len(idx[b]) >= len(idx[a]) \
                else np.resize(R[idx[b]], len(idx[a]))          # points to another's radii
        sv, se, sr = [], [], []
        for i in range(len(EDGES) - 1):
            m = (Rp >= EDGES[i]) & (Rp < EDGES[i + 1])
            if m.sum() < 3:
                continue
            a, b = wmean(val[m], err[m])
            sv.append(a); se.append(b)
            sr.append(float(np.average(Rp[m], weights=1 / err[m] ** 2)))
        if len(sv) >= 3:
            s2 = slope(np.array(sr), np.array(sv), np.array(se))
            if np.isfinite(s2):
                null.append(abs(s2))
    p_slope = float((np.array(null) >= abs(obs_slope)).mean()) if null else float("nan")

    grand, gerr = wmean(val, err)
    print("")
    print("  all radii together        %5.2f +- %.2f   (%+.1f sigma from 1)"
          % (grand, gerr, (grand - 1.0) / gerr))
    print("  slope d(residual)/d(log10 r) = %+.2f per dex   p = %.4f"
          % (obs_slope, p_slope))
    print("")
    if np.isfinite(p_slope) and p_slope < 0.05:
        print("  -> the residual DEPENDS ON RADIUS. It is not one amplitude.")
    else:
        print("  -> no radial trend at this power; consistent with one amplitude.")
    print("")
    print("  CAVEAT that binds the inner shells: g_bar counts gas only. A stellar")
    print("  term raises g_bar most where the BCG is, lowering the inner residual,")
    print("  which is the same direction as any trend seen here.")

    out = dict(lane="clusterfirst", stage="radial-shells",
               gas_file=os.environ.get("GAS_FILE", "gas_extended.json"),
               edges_mpc=EDGES, shells=shells, grand=grand, grand_err=gerr,
               slope_per_dex=obs_slope, p_slope=p_slope, n_perm=len(null),
               null_note=("cluster LABELS permuted, not points -- points within a "
                          "cluster are correlated and permuting them across "
                          "clusters inflates the null"),
               caveat="g_bar is gas only; no stellar term",
               sealed_untouched=len(loader.sealed_names()))
    io.open(OUT, "w", newline="\n", encoding="utf-8").write(
        json.dumps(out, indent=1) + "\n")
    print("")
    print("wrote radial.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
