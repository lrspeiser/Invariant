"""modifications.py -- start from MOND and ask what closes the cluster factor of two.

Run BT measured the residual: with g_bar computed from the observed gas, the RAR
under-predicts cluster lensing by about 2.2x, on fifteen points. `extend_gas.py`
turned that into a profile reaching the lensing radii, so the residual can now be
asked a harder question than "how big" -- namely "what is it a function of".

THE FAMILIES, one free parameter each, each multiplying the MOND/RAR prediction:

    slip          M = s                       a constant: lensing sees s times
                                              the dynamical potential
    a0 shift      a0 -> f a0                  a different acceleration scale in
                                              clusters
    density       M = (rho_gas/rho_ref)^p     local baryon density
    temperature   M = (kT/5 keV)^p            the heat
    radius        M = (r/1 Mpc)^p             position in the cluster
    redshift      M = (1+z)^p                 time
    gas fraction  M = (M_gas/M_gas,ref)^p     how much baryon there is at all

THE DISCRIMINATOR, and it is the whole point. A modification that merely
rescales -- slip, a0 shift -- can always drive the median ratio to 1, because it
has an amplitude and the residual has an amplitude. That is not evidence of
anything. What separates a real dependence from a rescaling is whether it also
reduces the SCATTER. A variable the residual genuinely depends on will pull the
eleven clusters onto a tighter locus; one it does not will move them all
together and leave the spread where it was.

So every family is reported as a pair: median ratio after fitting, and scatter
after fitting, against the unmodified RAR's 0.65 dex.

NULLS. Each family is refitted with its variable permuted across clusters. A
one-parameter family fitted to ten clusters can absorb a surprising amount, and
the permuted version measures exactly how much.

    python modifications.py
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
sys.path.insert(0, WELLNET)

import cosmo as C                                              # noqa: E402
import firstprinciples as FP                                   # noqa: E402

OUT = os.path.join(HERE, "modifications.json")
N_PERM = 300
SEED = 20260907

RHO_REF = 1e-25          # kg/m^3, a mid-cluster gas density
KT_REF = 5.0             # keV
R_REF = 1.0 * C.MPC
MG_REF = 1e13 * 1.989e30 # kg


def build_rows():
    """One row per (cluster, lensing bin) with g_bar from the EXTENDED gas."""
    # GAS_FILE selects the gas model: the default is the analytic-vignetting
    # version, gas_extended_expcorr.json is the one built on real CIAO
    # exposure maps. Both are kept so the difference is attributable.
    ext = json.load(io.open(os.path.join(HERE, os.environ.get(
        "GAS_FILE", "gas_extended.json")), encoding="utf-8"))
    match = {m["erass"]: m for m in json.load(
        io.open(os.path.join(HERE, "accept_overlap.json"), encoding="utf-8"))}
    prof = {}
    for line in io.open(os.path.join(WELLNET, "clustershear", "profiles.jsonl"),
                        encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("profile"):
                prof[r["name"]] = r

    rows = []
    for name, g in ext.items():
        rec = prof.get(name)
        if rec is None:
            continue
        z, beta, rc, n0 = g["z"], g["beta"], g["rc_mpc"] * C.MPC, g["n0"]
        reach = g["reach_mpc"] * C.MPC
        # Optional outer break (gas_truncated.json). A beta-model has no outer
        # truncation -- rho ~ r^-1.8 gives M(r) ~ r^1.2, which diverges -- while
        # real cluster gas steepens beyond R500. gas_truncated.py fits
        #     n_e = n0 (1+(r/rc)^2)^(-3 beta/2) (1+(r/rs)^3)^(-eps/6)
        # to the same Chandra surface brightness. The X-ray data does not
        # REQUIRE the break (rescaled dchi2 0.0-2.4 for two parameters, 0 of 10
        # clusters at p<0.05) but it PERMITS one that removes 43% of the gas
        # mass inside 4.6 Mpc. Running with and without it brackets how much of
        # any radial trend the gas model alone can account for.
        rs = g.get("rs_mpc")
        eps = g.get("eps", 0.0)
        rs = rs * C.MPC if (rs and eps > 0) else None
        kt = match[name].get("kt")
        try:
            kt = float(kt)
        except (TypeError, ValueError):
            kt = None

        def rho_of(x, _rs=rs, _eps=eps):
            x = np.atleast_1d(np.asarray(x, dtype=float))
            ne = n0 * (1.0 + (x / rc) ** 2) ** (-1.5 * beta)    # cm^-3
            if _rs:
                ne = ne * (1.0 + (x / _rs) ** 3) ** (-_eps / 6.0)
            return FP.MU_E * FP.M_P * ne * 1e6                  # kg/m^3

        rr = np.geomspace(1e-3 * C.MPC, reach * 1.02, 700)
        dens = rho_of(rr)
        mass = np.concatenate(([0.0], np.cumsum(
            0.5 * (4 * math.pi * rr[:-1] ** 2 * dens[:-1]
                   + 4 * math.pi * rr[1:] ** 2 * dens[1:]) * np.diff(rr))))

        obs = FP.observed(rec)
        R = np.array([o[0] for o in obs])
        inside = R <= reach

        # MATCHED PHYSICAL WINDOW. The shear aperture and the Chandra field are
        # both ANGULAR, so the physical radius they correspond to is set by the
        # cluster's distance. Measured on the rows this function returns, the
        # inverse-variance-weighted radius runs from 0.62 Mpc for the z = 0.069
        # cluster to 3.42 Mpc for the z = 0.540 one -- a factor of 5.5 -- and
        #
        #     corr(redshift, weighted-mean radius) = +0.98
        #
        # At that collinearity redshift and radius are not two variables, they
        # are one. Any "the residual tracks redshift" is equally "the residual
        # tracks radius", and the ten per-cluster residuals are not comparable
        # to each other because each is measured somewhere else.
        #
        # R_WINDOW_MPC="lo,hi" restricts every cluster to the same physical
        # annulus, which costs bins and signal-to-noise and buys a test whose
        # ten numbers mean the same thing. Unset, the behaviour is as before.
        win = os.environ.get("R_WINDOW_MPC", "").strip()
        if win:
            lo, hi = (float(x) for x in win.split(","))
            inside = inside & (R >= lo * C.MPC) & (R <= hi * C.MPC)
        if inside.sum() < 2:
            continue
        ds_bar = FP.delta_sigma_bar(rho_of, R, reach)
        gbar = FP.G * np.interp(R, rr, mass) / R ** 2
        for i, o in enumerate(obs):
            if not inside[i] or not (ds_bar[i] > 0) or not (o[2] > 0):
                continue
            rows.append(dict(cluster=name, z=z, kt=kt, R=float(R[i]),
                             ds_obs=float(o[1]), ds_err=float(o[2]),
                             ds_bar=float(ds_bar[i]), g_bar=float(gbar[i]),
                             rho=float(rho_of(R[i])[0]),
                             m_gas=float(np.interp(R[i], rr, mass))))
    return rows


# --------------------------------------------------------------- the families
def boost_rar(gb, a0=FP.A0):
    y = np.maximum(gb / a0, 1e-12)
    return 1.0 / (1.0 - np.exp(-np.sqrt(y)))


FAMILIES = {
    "none (plain RAR)":  lambda r, p: np.ones_like(r["g_bar"]),
    "slip (constant)":   lambda r, p: np.full_like(r["g_bar"], p),
    "a0 shift":          None,          # handled specially: changes the boost
    "density^p":         lambda r, p: (r["rho"] / RHO_REF) ** p,
    "temperature^p":     lambda r, p: (r["kt"] / KT_REF) ** p,
    "radius^p":          lambda r, p: (r["R"] / R_REF) ** p,
    "redshift (1+z)^p":  lambda r, p: (1.0 + r["z"]) ** p,
    "gas mass^p":        lambda r, p: (r["m_gas"] / MG_REF) ** p,
}


def score(arr, p, name):
    gb, dsb, dso = arr["g_bar"], arr["ds_bar"], arr["ds_obs"]
    if name == "a0 shift":
        pred = dsb * boost_rar(gb, a0=FP.A0 * max(p, 1e-3))
    else:
        pred = dsb * boost_rar(gb) * FAMILIES[name](arr, p)
    rat = pred / dso
    ok = np.isfinite(rat) & (rat > 0)
    if ok.sum() < 5:
        return None
    lr = np.log10(rat[ok])
    return float(10 ** np.median(lr)), float(np.std(lr)), int(ok.sum())


def fit(arr, name):
    """One-parameter grid search minimising |log median| + scatter."""
    grid = (np.linspace(0.2, 12.0, 120) if name in ("slip (constant)", "a0 shift")
            else np.linspace(-3.0, 3.0, 121))
    best = None
    for p in grid:
        s = score(arr, p, name)
        if s is None:
            continue
        med, sc, n = s
        cost = abs(math.log10(med)) + sc
        if best is None or cost < best[0]:
            best = (cost, float(p), med, sc, n)
    return best


def main():
    from holdout import loader                                 # noqa: E402
    loader.verify()
    rows = build_rows()
    clusters = sorted({r["cluster"] for r in rows})
    loader.assert_not_sealed(clusters, "modification search")
    print("rows %d over %d clusters (sealed %d untouched)\n"
          % (len(rows), len(clusters), len(loader.sealed_names())), flush=True)

    def pack(rs):
        return {k: np.array([r[k] if r[k] is not None else np.nan for r in rs])
                for k in ("g_bar", "ds_bar", "ds_obs", "ds_err", "rho", "R",
                          "z", "kt", "m_gas")}

    base = pack(rows)
    b0 = score(base, 0.0, "none (plain RAR)")
    print("  plain RAR (no modification):  median %.3f   scatter %.2f dex   n=%d\n"
          % (b0[0], b0[1], b0[2]))

    rng = np.random.default_rng(SEED)
    out = []
    print("  %-20s %-7s %-9s %-9s %-9s %s"
          % ("family", "p", "median", "scatter", "null sd", "verdict"))
    for name in FAMILIES:
        if name == "none (plain RAR)":
            continue
        arr = pack(rows)
        if name == "temperature^p" and not np.isfinite(arr["kt"]).any():
            continue
        r = fit(arr, name)
        if r is None:
            continue
        _, p, med, sc, n = r

        # null: permute the family's variable ACROSS CLUSTERS
        key = {"density^p": "rho", "temperature^p": "kt", "radius^p": "R",
               "redshift (1+z)^p": "z", "gas mass^p": "m_gas"}.get(name)
        nulls = []
        if key:
            bycl = {}
            for i, rr_ in enumerate(rows):
                bycl.setdefault(rr_["cluster"], []).append(i)
            for _ in range(N_PERM):
                a2 = {k: v.copy() for k, v in arr.items()}
                cl = list(bycl)
                perm = rng.permutation(len(cl))
                vals = [arr[key][bycl[c]].copy() for c in cl]
                for a, c in enumerate(cl):
                    src = vals[perm[a]]
                    tgt = bycl[c]
                    a2[key][tgt] = src[:len(tgt)] if len(src) >= len(tgt) else \
                        np.resize(src, len(tgt))
                rn = fit(a2, name)
                if rn:
                    nulls.append(rn[3])       # the scatter it achieves
        nsd = float(np.std(nulls)) if nulls else float("nan")
        nmean = float(np.mean(nulls)) if nulls else float("nan")
        improved = b0[1] - sc
        beats = (bool(nulls) and sc < nmean - 2 * nsd)
        verdict = ("reduces scatter beyond its null" if beats else
                   "rescales only" if abs(math.log10(med)) < 0.05 and improved < 0.03
                   else "no gain beyond null")
        out.append(dict(family=name, p=p, median=med, scatter=sc, n=n,
                        null_mean=nmean, null_sd=nsd,
                        scatter_gain_vs_plain_rar=improved, beats_null=beats,
                        verdict=verdict))
        print("  %-20s %-7.2f %-9.3f %-9.2f %-9s %s"
              % (name, p, med, sc,
                 ("%.2f" % nsd) if nulls else "-", verdict), flush=True)

    res = dict(lane="clusterfirst", stage="modifications",
               n_rows=len(rows), n_clusters=len(clusters),
               plain_rar=dict(median=b0[0], scatter=b0[1], n=b0[2]),
               families=out, n_perm=N_PERM,
               sealed_untouched=len(loader.sealed_names()))
    io.open(OUT, "w", newline="\n", encoding="utf-8").write(
        json.dumps(res, indent=1) + "\n")
    print("\nwrote modifications.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
