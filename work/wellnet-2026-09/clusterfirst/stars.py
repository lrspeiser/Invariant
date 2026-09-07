"""stars.py -- can a stellar component account for the radius dependence?

THE ONE SYSTEMATIC LEFT. The boost the lensing demands departs from the RAR with
a radius coefficient C = -11.8 +- 3.0 at fixed g_bar (-3.9 sigma, permutation
p = 0.011). Three checks have already failed to remove it:

  * injection -- RAR-generated data on the same radii, gas models and error bars
    returns C = -0.19 +- 2.88, and injected r^-0.5 / r^-1.0 laws come back at
    -4.27 / -8.71 against -4.20 / -8.53 expected. The estimator is unbiased in
    both directions.
  * projection -- C moves by 0.2 across line-of-sight limits from 8 to 200 Mpc
    and inner grids from 0.05 to 0.001 R_min.
  * gas shape -- Vikhlinin's outer break fitted to the same Chandra surface
    brightness moves the radial slope from -2.38 to -2.34 per dex, and in the
    WRONG direction.

But g_bar counts GAS ONLY. Cluster stars are centrally concentrated, so adding
them raises g_bar most at small radius, lowers the boost the data demands there,
and flattens exactly this trend. That is the remaining way for the result to be
mundane, and it has to be bounded rather than waved at.

HOW IT IS BOUNDED. No new data is used and no lensing-calibrated quantity is
touched (eRASS1's M500 comes from eROCOP, which is weak-lensing calibrated, so
using it here would be circular). Instead a stellar component is ADDED at a
range of masses and concentrations that brackets and then exceeds anything
observed, and the coefficient is watched:

    M_star = f x M_gas(<reach),  f = 0.05 to 0.40
    Hernquist profile, scale a = 0.05 Mpc (BCG-like) to 0.5 Mpc (satellites)

Observed clusters sit near f = 0.1-0.2. If f = 0.4 with a 50 kpc scale -- more
stellar mass than any cluster has, packed tighter than any cluster packs it --
still leaves the coefficient, no stellar population can be the explanation.

    python stars.py
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
import firstprinciples as FP                                    # noqa: E402
import modifications as MOD                                     # noqa: E402

OUT = os.path.join(HERE, "stars.json")
F_STAR = [0.0, 0.05, 0.10, 0.20, 0.40]
A_STAR = [0.05, 0.15, 0.50]                                     # Mpc


def rows_with_stars(f_star, a_star_mpc):
    """Rebuild g_bar and Delta_Sigma_bar with a Hernquist stellar component."""
    ext = json.load(io.open(os.path.join(
        HERE, os.environ.get("GAS_FILE", "gas_extended_expcorr.json")),
        encoding="utf-8"))
    prof = {}
    for line in io.open(os.path.join(WELLNET, "clustershear", "profiles.jsonl"),
                        encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("profile"):
                prof[r["name"]] = r

    a = a_star_mpc * C.MPC
    out = []
    for name, g in ext.items():
        rec = prof.get(name)
        if rec is None:
            continue
        beta, rc, n0 = g["beta"], g["rc_mpc"] * C.MPC, g["n0"]
        reach = g["reach_mpc"] * C.MPC
        rs = g.get("rs_mpc")
        eps = g.get("eps", 0.0)
        rs = rs * C.MPC if (rs and eps > 0) else None

        def rho_gas(x, _rs=rs, _eps=eps):
            x = np.atleast_1d(np.asarray(x, dtype=float))
            ne = n0 * (1.0 + (x / rc) ** 2) ** (-1.5 * beta)
            if _rs:
                ne = ne * (1.0 + (x / _rs) ** 3) ** (-_eps / 6.0)
            return FP.MU_E * FP.M_P * ne * 1e6

        rr = np.geomspace(1e-3 * C.MPC, reach * 1.02, 700)
        dg = rho_gas(rr)
        mgas = np.concatenate(([0.0], np.cumsum(
            0.5 * (4 * math.pi * rr[:-1] ** 2 * dg[:-1]
                   + 4 * math.pi * rr[1:] ** 2 * dg[1:]) * np.diff(rr))))
        m_star = f_star * mgas[-1]

        def rho_of(x, _ms=m_star, _a=a):
            x = np.atleast_1d(np.asarray(x, dtype=float))
            d = rho_gas(x)
            if _ms > 0:
                d = d + _ms * _a / (2 * math.pi * np.maximum(x, 1e-6 * C.MPC)
                                    * (x + _a) ** 3)
            return d

        # Hernquist enclosed mass is analytic: M r^2/(r+a)^2
        obs = FP.observed(rec)
        R = np.array([o[0] for o in obs])
        inside = R <= reach
        if inside.sum() < 2:
            continue
        ds_bar = FP.delta_sigma_bar(rho_of, R, reach)
        mtot = np.interp(R, rr, mgas) + m_star * R ** 2 / (R + a) ** 2
        gbar = FP.G * mtot / R ** 2
        for i, o in enumerate(obs):
            if not inside[i] or not (ds_bar[i] > 0) or not (o[2] > 0):
                continue
            out.append((float(R[i]) / C.MPC, float(gbar[i]), float(ds_bar[i]),
                        float(o[1]), float(o[2])))
    return np.array(out)


def coef(arr):
    R, gb, dsb, dso, dse = arr.T
    rar = MOD.boost_rar(gb)
    lg, lr = np.log10(gb / FP.A0), np.log10(R)
    y = dso / dsb - rar
    w = (dsb / dse) ** 2
    X = np.column_stack([np.ones_like(lg), lg, lr])
    XtW = X.T * w
    cov = np.linalg.inv(XtW @ X)
    c = cov @ (XtW @ y)
    return float(c[2]), float(np.sqrt(np.diag(cov))[2]), len(R)


def main():
    from holdout import loader                                  # noqa: E402
    loader.verify()
    os.environ.pop("R_WINDOW_MPC", None)

    print("radius coefficient C, with a stellar component added")
    print("")
    print("  f_star = M_star / M_gas(<reach);  a = Hernquist scale")
    print("")
    hdr = "  %-10s" % "f_star"
    for a in A_STAR:
        hdr += " %-18s" % ("a = %.2f Mpc" % a)
    print(hdr)

    grid = []
    for f in F_STAR:
        line = "  %-10.2f" % f
        for a in A_STAR:
            arr = rows_with_stars(f, a)
            c, s, n = coef(arr)
            grid.append(dict(f_star=f, a_mpc=a, C=c, C_err=s, n=n,
                             sigma=c / s))
            line += " %+7.2f (%+.1f)  " % (c, c / s)
        print(line)

    worst = max(grid, key=lambda g: g["C"])                     # closest to zero
    print("")
    print("  weakest case in the whole grid: f_star = %.2f, a = %.2f Mpc"
          % (worst["f_star"], worst["a_mpc"]))
    print("     C = %+.2f +- %.2f  (%+.1f sigma)"
          % (worst["C"], worst["C_err"], worst["sigma"]))
    print("")
    obs_like = [g for g in grid if 0.05 <= g["f_star"] <= 0.20]
    print("  over the observed stellar range (f = 0.05-0.20): C from %+.2f to %+.2f"
          % (min(g["C"] for g in obs_like), max(g["C"] for g in obs_like)))
    if worst["sigma"] <= -3.0:
        print("")
        print("  -> no stellar population removes it. Even f_star = 0.40 packed at a")
        print("     50 kpc scale -- more stars than any cluster has, more centrally")
        print("     concentrated than any cluster puts them -- leaves %+.1f sigma."
              % worst["sigma"])

    out = dict(lane="clusterfirst", stage="stellar-bound",
               gas_file=os.environ.get("GAS_FILE", "gas_extended_expcorr.json"),
               f_star_grid=F_STAR, a_star_mpc_grid=A_STAR, grid=grid,
               weakest=worst,
               note=("stellar mass added as a Hernquist profile scaled to the "
                     "gas mass; no lensing-calibrated quantity used, since "
                     "eRASS1 M500 comes from eROCOP and would be circular"),
               sealed_untouched=len(loader.sealed_names()))
    io.open(OUT, "w", newline="\n", encoding="utf-8").write(
        json.dumps(out, indent=1) + "\n")
    print("")
    print("wrote stars.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
