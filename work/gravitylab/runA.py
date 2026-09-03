"""Run A of the anisotropic-void test program: one-dimensional galaxy screening.

Steps 4-7. Global gravity parameters are fitted on the TRAIN split only and
then evaluated, unchanged, on galaxies that were sealed before any fitting.

Nuisances, drawn per galaxy per the program's step 4:

    D    ~ N(D0, e_D)          catalogue distance and its error
    i    ~ N(i0, e_i)          catalogue inclination and its error
    Ups_d ~ lognormal(0.5, 0.10 dex)    stellar-population prior
    Ups_b ~ lognormal(0.7, 0.10 dex)

and the transformations the program specifies:

    R      = R0 (D/D0)
    V_obs  = V_obs,0 sin(i0)/sin(i)
    V_b^2  = (D/D0)[ V_gas|V_gas| + Ups_d V_disk^2 + Ups_b V_bulge^2 ]

The likelihood is marginalised over the draws rather than evaluated at their
mean, so a model cannot win by driving a nuisance to the edge of its prior.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

import data as D
import models as M
import qfield as Q

G = 6.674e-11
KPC = 3.0856775814913673e19
KMS = 1e3
MSUN = 1.98892e30
NDRAW = 16
SEED = 20260903
BAR = "=" * 78


def head(t):
    print("\n" + BAR + "\n" + t + "\n" + BAR)


# ------------------------------------------------------------ nuisance draws
def build_draws(gals):
    """Frozen nuisance realisations, one bundle per galaxy."""
    rng = np.random.default_rng(SEED)
    for g in gals:
        nd, npt = NDRAW, len(g)
        Dd = rng.normal(g.D0, max(g.eD, 1e-3), nd)
        Dd = np.clip(Dd, 0.2 * g.D0, 3.0 * g.D0)
        ii = rng.normal(g.i0, max(g.ei, 0.5), nd)
        ii = np.clip(ii, 15.0, 90.0)
        ud = 0.5 * 10 ** rng.normal(0.0, 0.10, nd)
        ub = 0.7 * 10 ** rng.normal(0.0, 0.10, nd)
        f = (Dd / g.D0)[:, None]
        R = g.R0[None, :] * f                                   # kpc
        Vobs = g.Vobs0[None, :] * (math.sin(math.radians(g.i0))
                                   / np.sin(np.radians(ii)))[:, None]
        Vb2 = f * (g.Vgas[None, :] * np.abs(g.Vgas)[None, :]
                   + ud[:, None] * g.Vdisk[None, :] ** 2
                   + ub[:, None] * g.Vbul[None, :] ** 2)
        ok = np.all(Vb2 > 0, axis=0)
        if ok.sum() < D.CUTS["min_points"]:
            g.draws = None
            continue
        R, Vobs, Vb2 = R[:, ok], Vobs[:, ok], Vb2[:, ok]
        eV = g.eV[ok]
        Rm = R * KPC
        g.draws = dict(
            R=R, Vobs=Vobs, eV=eV,
            gbar=(Vb2 * KMS ** 2) / Rm,
            gobs=(Vobs * KMS) ** 2 / Rm,
            Mb=(Vb2[:, -1] * KMS ** 2) * Rm[:, -1] / G / MSUN,
            Rdisk=g.Rdisk if g.Rdisk > 0 else max(g.R0[-1] / 4.0, 0.1),
        )
    return [g for g in gals if getattr(g, "draws", None) is not None]


# ------------------------------------------------------------- model wrapper
def predict(spec, p, d):
    """Predicted acceleration, shape (ndraw, npts)."""
    gb, R, Mb = d["gbar"], d["R"], d["Mb"]
    kind = spec["kind"]
    if kind == "newton":
        return gb
    if kind == "aqual":
        return M.D2_aqual(gb, a0=p["a0"], form=spec["form"])
    if kind == "piecewise":
        return M.D3_piecewise(gb, a0=p["a0"])
    if kind == "flatlog":
        out = np.empty_like(gb)
        for k in range(gb.shape[0]):
            out[k] = M.D4_flatlog(gb[k], R_kpc=R[k], Mb_msun=Mb[k],
                                  a0=p["a0"], c_rc=p["c_rc"],
                                  Rdisk_kpc=d["Rdisk"])
        return out
    if kind == "void":
        out = np.empty_like(gb)
        for k in range(gb.shape[0]):
            if spec["q"] == "Q1_rho":
                q = Q.q_rho(gb[k], R[k], rho_c=p["rho_c"], m=p["m"])
            elif spec["q"] == "Q2_g":
                q = Q.q_g(gb[k], a0=p["a0"], n=p["n"])
            elif spec["q"] == "Q3_rho_g":
                q = Q.q_rho_g(gb[k], R[k], rho_c=p["rho_c"], m=p["m"],
                              a0=p["a0"], n=p["n"])
            else:
                q = Q.q_nonlocal(gb[k], R[k], L_q_kpc=p["L_q"],
                                 rho_c=p["rho_c"], m=p["m"],
                                 a0=p["a0"], n=p["n"])
            out[k] = M.scalar_void(gb[k], q, alpha=p["alpha"])
        return out
    raise ValueError(kind)


def galaxy_loglike(spec, p, d):
    """Marginal log-likelihood for one galaxy, averaged over nuisance draws."""
    gpred = np.maximum(predict(spec, p, d), 1e-30)
    Vm = np.sqrt(gpred * d["R"] * KPC) / KMS
    chi2 = np.sum(((d["Vobs"] - Vm) / d["eV"][None, :]) ** 2, axis=1)
    return float(logsumexp(-0.5 * chi2) - math.log(len(chi2)))


def total_nll(spec, vec, gals):
    p = unpack(spec, vec)
    if p is None:
        return 1e12
    s = 0.0
    for g in gals:
        s -= galaxy_loglike(spec, p, g.draws)
    return s if np.isfinite(s) else 1e12


# ------------------------------------------------------------- param packing
PRIORS = dict(a0=(-13.0, -8.0), rho_c=(-28.0, -20.0), m=(0.25, 8.0),
              n=(0.25, 8.0), alpha=(-5.0, 5.0), c_rc=(-2.0, 1.0),
              L_q=(-1.0, 3.0))
LOGPAR = {"a0", "rho_c", "c_rc", "L_q"}


def unpack(spec, vec):
    p = {}
    for name, v in zip(spec["free"], vec):
        lo, hi = PRIORS[name]
        if not (lo <= v <= hi):
            return None
        p[name] = 10 ** v if name in LOGPAR else v
    p.setdefault("a0", 1.2e-10)
    return p


def x0_of(spec):
    start = dict(a0=math.log10(1.2e-10), rho_c=-24.0, m=1.0, n=1.0,
                 alpha=1.0, c_rc=math.log10(0.5), L_q=1.0)
    return [start[k] for k in spec["free"]]


# -------------------------------------------------------------------- report
def metrics(spec, p, gals):
    lo, n_pt, res, gobs_all, gpred_all = 0.0, 0, [], [], []
    chi2 = 0.0
    for g in gals:
        d = g.draws
        lo += galaxy_loglike(spec, p, d)
        gp = np.maximum(predict(spec, p, d), 1e-30)
        Vm = np.sqrt(gp * d["R"] * KPC) / KMS
        chi2 += float(np.mean(np.sum(((d["Vobs"] - Vm) / d["eV"]) ** 2, axis=1)))
        n_pt += d["gobs"].shape[1]
        gobs_all.append(np.mean(d["gobs"], axis=0))
        gpred_all.append(np.mean(gp, axis=0))
    go = np.concatenate(gobs_all)
    gp = np.concatenate(gpred_all)
    rms = float(np.sqrt(np.mean((np.log10(go) - np.log10(gp)) ** 2)))
    return dict(loglike=lo, chi2=chi2, chi2_red=chi2 / max(n_pt, 1),
                rms_dex=rms, n_points=n_pt, n_gal=len(gals))


SPECS = [
    dict(name="D1 Newton", kind="newton", free=[]),
    dict(name="D2 AQUAL simple", kind="aqual", form="simple", free=["a0"]),
    dict(name="D2 AQUAL standard", kind="aqual", form="standard", free=["a0"]),
    dict(name="D3 piecewise", kind="piecewise", free=["a0"]),
    dict(name="D4 flattened log", kind="flatlog", free=["a0", "c_rc"]),
    dict(name="K1xQ1 rho", kind="void", q="Q1_rho",
         free=["rho_c", "m", "alpha"]),
    dict(name="K1xQ2 g", kind="void", q="Q2_g", free=["a0", "n", "alpha"]),
    dict(name="K1xQ3 rho+g", kind="void", q="Q3_rho_g",
         free=["a0", "n", "rho_c", "m", "alpha"]),
    dict(name="K1xQ4 nonlocal", kind="void", q="Q4_nonlocal",
         free=["a0", "n", "rho_c", "m", "alpha", "L_q"]),
]


def main():
    head("Run A, steps 4-7: nuisance sampling, fit on train, evaluate blind")
    gals = D.ingest(verbose=True)
    D.stratified_split(gals, verbose=True)
    gals = build_draws(gals)
    tr = [g for g in gals if g.split == "train"]
    va = [g for g in gals if g.split == "validation"]
    bl = [g for g in gals if g.split == "blind"]
    print(f"\n   nuisance draws per galaxy  : {NDRAW}")
    print(f"   train / validation / blind : {len(tr)} / {len(va)} / {len(bl)}")

    head("Fitting on TRAIN only")
    print(f"   {'model':<22}{'free':>5}{'chi2/pt tr':>12}{'RMS tr':>9}"
          f"{'chi2/pt BL':>12}{'RMS BL':>9}{'lnL BL':>11}")
    print("   " + "-" * 80)
    rows = []
    for spec in SPECS:
        if spec["free"]:
            r = minimize(lambda v: total_nll(spec, v, tr), x0_of(spec),
                         method="Nelder-Mead",
                         options=dict(maxiter=600, xatol=1e-3, fatol=1e-2))
            p = unpack(spec, r.x) or unpack(spec, x0_of(spec))
            best = {k: (10 ** v if k in LOGPAR else v)
                    for k, v in zip(spec["free"], r.x)}
        else:
            p, best = unpack(spec, []), {}
        mt = metrics(spec, p, tr)
        mb = metrics(spec, p, bl)
        rows.append(dict(name=spec["name"], nfree=len(spec["free"]),
                         params=best, train=mt, blind=mb))
        print(f"   {spec['name']:<22}{len(spec['free']):>5}"
              f"{mt['chi2_red']:>12.2f}{mt['rms_dex']:>9.4f}"
              f"{mb['chi2_red']:>12.2f}{mb['rms_dex']:>9.4f}"
              f"{mb['loglike']:>11.1f}")
    print("   " + "-" * 80)

    head("Fitted global parameters")
    for r in rows:
        if not r["params"]:
            continue
        s = "  ".join(f"{k}={v:.4g}" for k, v in r["params"].items())
        print(f"   {r['name']:<22} {s}")

    out = ROOTOUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "outputs")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "runA_results.json"), "w",
              encoding="utf-8", newline="\n") as fh:
        json.dump(rows, fh, indent=1, default=float)
    print(f"\n   wrote outputs/runA_results.json")
    return rows


if __name__ == "__main__":
    main()
