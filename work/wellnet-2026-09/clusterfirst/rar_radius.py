"""rar_radius.py -- at fixed g_bar, does the cluster boost depend on radius?

THE CLAIM THE RAR MAKES. The radial acceleration relation says the observed
acceleration is a function of the baryonic one ALONE:

    g_obs = F(g_bar),   F(x) = x / (1 - exp(-sqrt(x/a0)))

No radius, no mass, no environment. That is the whole content of it, and it is
why Milgrom's law needs zero free parameters per galaxy. In galaxies the claim is
nearly untestable in this direction, because a rotation curve traces one object
and radius and g_bar move together. Clusters are different: the sample spans
enough mass that at a given radius g_bar varies by an order of magnitude, and

    corr(log g_bar, log r) = -0.55

so 84% of the radial range survives conditioning on g_bar. Radius can be tested
as an independent variable, which in galaxy data it cannot.

THE TEST.

    DS_obs / DS_bar  -  F(g_bar)/g_bar  =  A + B log10(g_bar/a0) + C log10(r)

C must be zero if the RAR holds. The null is a permutation of CLUSTER LABELS,
which keeps each cluster's points together -- permuting points across clusters
breaks the within-cluster correlation, inflates the null and manufactures
flatness, which is the error that produced three p = 1.00 results in stacked.py.

WHAT HAS ALREADY FAILED TO REMOVE IT.

  injection      RAR-generated data on the same radii, gas models and error bars
                 returns C = -0.19 +- 2.88; injected r^-0.5 and r^-1.0 laws come
                 back at -4.27 and -8.71 against -4.20 and -8.53 expected, so the
                 estimator is unbiased in both directions.   [injection.py]
  projection     C moves by 0.2 across line-of-sight limits of 8 to 200 Mpc and
                 inner grids of 0.05 to 0.001 R_min.
  gas outer shape  Vikhlinin's break fitted to the same Chandra surface
                 brightness moves the radial slope by 0.04, the WRONG way:
                 truncating the gas lowers Sigma(R) more than the mean Sigma
                 inside R, so DS_bar rises.               [gas_truncated.py]
  stellar mass   nulling C needs ~1e14 Msun of stars inside 50 kpc, a hundred
                 times any observed BCG.                        [stars.py]
  vignetting     measured with CIAO exposure maps, not assumed. [regas.py]

This file runs the test itself with the stellar component included, and puts it
through leave-one-cluster-out and two alternative functional forms.

    python rar_radius.py
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

OUT = os.path.join(HERE, "rar_radius.json")
N_PERM = 20000
SEED = 20260907

# Realistic cluster stellar content, from the observed split rather than a guess:
# the BCG is of order 1.5e12 Msun against a median gas mass of 1.7e14, and the
# satellites carry the rest and trace the total mass, not the core.
F_BCG, A_BCG = 0.015, 0.03                                      # Mpc
F_SAT, A_SAT = 0.085, 0.50                                      # Mpc


def build(f_bcg=F_BCG, a_bcg=A_BCG, f_sat=F_SAT, a_sat=A_SAT):
    """Rows with gas + a two-component stellar term. Returns arrays and labels."""
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

    R, GB, DSB, DSO, DSE, CL = [], [], [], [], [], []
    for name, g in ext.items():
        rec = prof.get(name)
        if rec is None:
            continue
        beta, rc, n0 = g["beta"], g["rc_mpc"] * C.MPC, g["n0"]
        reach = g["reach_mpc"] * C.MPC

        def rho_gas(x):
            x = np.atleast_1d(np.asarray(x, dtype=float))
            return FP.MU_E * FP.M_P * n0 * (1.0 + (x / rc) ** 2) ** (-1.5 * beta) * 1e6

        rr = np.geomspace(1e-3 * C.MPC, reach * 1.02, 700)
        dg = rho_gas(rr)
        mgas = np.concatenate(([0.0], np.cumsum(
            0.5 * (4 * math.pi * rr[:-1] ** 2 * dg[:-1]
                   + 4 * math.pi * rr[1:] ** 2 * dg[1:]) * np.diff(rr))))
        comps = [(f_bcg * mgas[-1], a_bcg * C.MPC),
                 (f_sat * mgas[-1], a_sat * C.MPC)]

        def rho_of(x, _c=comps):
            x = np.atleast_1d(np.asarray(x, dtype=float))
            d = rho_gas(x)
            for M, a in _c:
                if M > 0:
                    d = d + M * a / (2 * math.pi
                                     * np.maximum(x, 1e-6 * C.MPC) * (x + a) ** 3)
            return d

        obs = FP.observed(rec)
        Rv = np.array([o[0] for o in obs])
        ins = Rv <= reach
        if ins.sum() < 2:
            continue
        dsb = FP.delta_sigma_bar(rho_of, Rv, reach)
        mt = np.interp(Rv, rr, mgas) + sum(M * Rv ** 2 / (Rv + a) ** 2
                                           for M, a in comps)
        gb = FP.G * mt / Rv ** 2
        for i, o in enumerate(obs):
            if not ins[i] or not (dsb[i] > 0) or not (o[2] > 0):
                continue
            R.append(Rv[i] / C.MPC); GB.append(gb[i]); DSB.append(dsb[i])
            DSO.append(o[1]); DSE.append(o[2]); CL.append(name)
    return (np.array(R), np.array(GB), np.array(DSB), np.array(DSO),
            np.array(DSE), np.array(CL))


def fit(R, gb, dsb, dso, dse, form="linear"):
    """Returns (C, sigma_C). `form` selects the parameterisation of g_bar."""
    rar = MOD.boost_rar(gb)
    lg, lr = np.log10(gb / FP.A0), np.log10(R)
    y = dso / dsb - rar
    w = (dsb / dse) ** 2
    if form == "linear":
        cols = [np.ones_like(lg), lg, lr]
    elif form == "quadratic":                                   # g_bar more freely
        cols = [np.ones_like(lg), lg, lg ** 2, lr]
    elif form == "ratio":                                       # y as a ratio, not a difference
        y = (dso / dsb) / rar
        w = (dsb * rar / dse) ** 2
        cols = [np.ones_like(lg), lg, lr]
    else:
        raise ValueError(form)
    X = np.column_stack(cols)
    XtW = X.T * w
    cov = np.linalg.inv(XtW @ X)
    c = cov @ (XtW @ y)
    se = np.sqrt(np.diag(cov))
    return float(c[-1]), float(se[-1])


def main():
    from holdout import loader                                  # noqa: E402
    loader.verify()
    os.environ.pop("R_WINDOW_MPC", None)

    R, gb, dsb, dso, dse, cl = build()
    loader.assert_not_sealed(sorted(set(cl)), "RAR radius-dependence test")
    names = sorted(set(cl))

    c0, s0 = fit(R, gb, dsb, dso, dse)
    print("at fixed g_bar, does the boost depend on radius?")
    print("")
    print("  gas + realistic stars (BCG %.3f at %.2f Mpc, satellites %.3f at %.2f Mpc)"
          % (F_BCG, A_BCG, F_SAT, A_SAT))
    print("  n = %d points, %d clusters" % (len(R), len(names)))
    print("")
    print("  C (log r coefficient) = %+.2f +- %.2f   (%+.1f sigma)"
          % (c0, s0, c0 / s0))
    print("  the RAR requires C = 0")
    print("")

    # permutation null: cluster LABELS, keeping each cluster's points together
    rng = np.random.default_rng(SEED)
    idx = {n: np.where(cl == n)[0] for n in names}
    lr = np.log10(R)
    null = []
    for _ in range(N_PERM // 5):
        perm = rng.permutation(names)
        lr2 = lr.copy()
        for a, b in zip(names, perm):
            src = lr[idx[b]]
            lr2[idx[a]] = (src[:len(idx[a])] if len(src) >= len(idx[a])
                           else np.resize(src, len(idx[a])))
        rar = MOD.boost_rar(gb)
        y = dso / dsb - rar
        w = (dsb / dse) ** 2
        X = np.column_stack([np.ones_like(lr2), np.log10(gb / FP.A0), lr2])
        XtW = X.T * w
        try:
            c2 = np.linalg.inv(XtW @ X) @ (XtW @ y)
        except np.linalg.LinAlgError:
            continue
        null.append(abs(c2[2]))
    null = np.array(null)
    p_perm = float((null >= abs(c0)).mean())
    print("  permutation p (cluster labels shuffled) = %.4f  [%d draws]"
          % (p_perm, len(null)))
    print("")

    # leave one cluster out
    print("  leave-one-cluster-out")
    jack = []
    for n in names:
        k = cl != n
        c, s = fit(R[k], gb[k], dsb[k], dso[k], dse[k])
        jack.append(dict(dropped=n, C=c, C_err=s, sigma=c / s))
        print("     drop %-24s C = %+6.2f +- %-5.2f (%+.1f sigma)%s"
              % (n, c, s, c / s, "   <- KILLS IT" if abs(c / s) < 2 else ""))
    worst = max(jack, key=lambda j: j["sigma"])
    print("")
    print("  worst case: dropping %s leaves %+.1f sigma"
          % (worst["dropped"], worst["sigma"]))
    print("")

    # alternative functional forms
    print("  functional form")
    forms = {}
    for f in ("linear", "quadratic", "ratio"):
        c, s = fit(R, gb, dsb, dso, dse, form=f)
        forms[f] = dict(C=c, C_err=s, sigma=c / s)
        print("     %-12s C = %+6.2f +- %-5.2f (%+.1f sigma)" % (f, c, s, c / s))

    ok = (abs(c0 / s0) >= 3 and p_perm < 0.05
          and abs(worst["sigma"]) >= 2
          and all(abs(v["sigma"]) >= 2 for v in forms.values()))
    print("")
    if ok:
        print("  -> the RAR's defining assumption FAILS on clusters. g_obs is not a")
        print("     function of g_bar alone: at fixed g_bar the required boost falls")
        print("     with radius, and it survives every control run against it.")
    else:
        print("  -> does not survive all controls; not established.")

    out = dict(lane="clusterfirst", stage="rar-radius-dependence",
               gas_file=os.environ.get("GAS_FILE", "gas_extended_expcorr.json"),
               stellar=dict(f_bcg=F_BCG, a_bcg_mpc=A_BCG,
                            f_sat=F_SAT, a_sat_mpc=A_SAT),
               n_points=int(len(R)), n_clusters=len(names),
               C=c0, C_err=s0, sigma=c0 / s0, p_permutation=p_perm,
               leave_one_out=jack, worst_case_sigma=worst["sigma"],
               worst_case_dropped=worst["dropped"], forms=forms,
               survives_all_controls=bool(ok),
               sealed_untouched=len(loader.sealed_names()))
    io.open(OUT, "w", newline="\n", encoding="utf-8").write(
        json.dumps(out, indent=1) + "\n")
    print("")
    print("wrote rar_radius.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
