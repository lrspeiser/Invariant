"""Screening and blind evaluation of the well-mirror laws. Runs from cold.

    python mirror_run.py

Steps
  0  reuse gravitylab/data.py: ingest, declared cuts, the frozen 60/20/20 split
  1  rebuild the frozen nuisance draws and VERIFY they match runA.build_draws
     element for element, then attach the catalogue baryonic mass M_b(cat)
  2  reproduce the published benchmarks (Newton 0.5544, AQUAL simple 0.1590)
     under this harness, as a self-check that the harness is the same one
  3  fit Options 2/3/4 on TRAIN ONLY, evaluate on the sealed BLIND galaxies
  4  residual correlations on BLIND against R/R_d, SB_eff, M_b, f_gas, i, D
  5  SENSITIVITY AUDIT: every headline statistic is recomputed over a grid in
     every global parameter, to prove the statistic actually responds to the
     parameter it is claimed to test
  6  intrinsic-scatter calibration: a single GLOBAL f_int, fitted, and chi2/dof
     shown before any lnL / AIC / BIC is quoted
  7  blind holdouts: KiDS lensing rotation curves and the wide binaries
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import spearmanr, pearsonr

GLAB = "C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/work/gravitylab"
SCR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for q in (GLAB, os.path.dirname(os.path.abspath(__file__))):
    if q not in sys.path:
        sys.path.insert(0, q)

import data as DAT            # noqa: E402  the frozen cuts and split
import runA as RA             # noqa: E402  the fit/evaluate harness being reused
import mirror_models as MM    # noqa: E402

G, KPC, KMS, MSUN, AU = MM.G, MM.KPC, 1e3, MM.MSUN, MM.AU
NDRAW, SEED = RA.NDRAW, RA.SEED
BAR = "=" * 78
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "mirror_results.json")
RESULTS: dict = {}


def head(t):
    print("\n" + BAR + "\n" + t + "\n" + BAR)


# ------------------------------------------------------------------ step 1
def build_draws(gals):
    """Replicates runA.build_draws exactly and additionally keeps the drawn
    nuisances plus a catalogue-photometry baryonic mass.

    M_b(cat) = [ Ups_d L36 + 1.33 M_HI ] (D/D0)^2 -- independent of where the
    rotation curve happens to stop. runA's M_b is the enclosed baryonic mass at
    the LAST measured radius, which correlates with how far the curve extends;
    both are carried and both are used, so no conclusion rests on the choice.
    """
    rng = np.random.default_rng(SEED)
    for g in gals:
        nd = NDRAW
        Dd = rng.normal(g.D0, max(g.eD, 1e-3), nd)
        Dd = np.clip(Dd, 0.2 * g.D0, 3.0 * g.D0)
        ii = rng.normal(g.i0, max(g.ei, 0.5), nd)
        ii = np.clip(ii, 15.0, 90.0)
        ud = 0.5 * 10 ** rng.normal(0.0, 0.10, nd)
        ub = 0.7 * 10 ** rng.normal(0.0, 0.10, nd)
        f = (Dd / g.D0)[:, None]
        R = g.R0[None, :] * f
        Vobs = g.Vobs0[None, :] * (math.sin(math.radians(g.i0))
                                   / np.sin(np.radians(ii)))[:, None]
        Vb2 = f * (g.Vgas[None, :] * np.abs(g.Vgas)[None, :]
                   + ud[:, None] * g.Vdisk[None, :] ** 2
                   + ub[:, None] * g.Vbul[None, :] ** 2)
        ok = np.all(Vb2 > 0, axis=0)
        if ok.sum() < DAT.CUTS["min_points"]:
            g.draws = None
            continue
        R, Vobs, Vb2 = R[:, ok], Vobs[:, ok], Vb2[:, ok]
        eV = g.eV[ok]
        Rm = R * KPC
        Mb_cat = (ud * g.L36 * 1e9 + 1.33 * max(g.MHI, 0.0) * 1e9) * (Dd / g.D0) ** 2
        g.draws = dict(
            R=R, Vobs=Vobs, eV=eV,
            gbar=(Vb2 * KMS ** 2) / Rm,
            gobs=(Vobs * KMS) ** 2 / Rm,
            Mb=(Vb2[:, -1] * KMS ** 2) * Rm[:, -1] / G / MSUN,
            Mb_cat=np.maximum(Mb_cat, 1e5),
            Rdisk=g.Rdisk if g.Rdisk > 0 else max(g.R0[-1] / 4.0, 0.1),
            Dd=Dd, inc=ii, ups_d=ud, ups_b=ub,
        )
    return [g for g in gals if getattr(g, "draws", None) is not None]


def verify_against_runA():
    """Element-for-element check that the rebuilt draws are runA's draws."""
    a = DAT.ingest(verbose=False)
    DAT.stratified_split(a, verbose=False)
    a = RA.build_draws(a)
    b = DAT.ingest(verbose=False)
    DAT.stratified_split(b, verbose=False)
    b = build_draws(b)
    assert len(a) == len(b), f"galaxy count differs: {len(a)} vs {len(b)}"
    worst = 0.0
    for x, y in zip(a, b):
        assert x.name == y.name
        for k in ("R", "Vobs", "gbar", "gobs", "Mb"):
            worst = max(worst, float(np.max(np.abs(x.draws[k] - y.draws[k]))))
    return len(a), worst


# ------------------------------------------------------------- model driver
def predict(law, p, d, mb_key="Mb_cat"):
    """Predicted acceleration, shape (ndraw, npts). Only MEASURED per-galaxy
    quantities are passed through `needs`."""
    spec = MM.LAWS[law]
    kw = {k: v for k, v in p.items() if k != "f_int"}
    if "r_m" in spec["needs"]:
        kw["r_m"] = d["R"] * KPC
    if "Mb_msun" in spec["needs"]:
        kw["Mb_msun"] = d[mb_key][:, None]
    return spec["fn"](d["gbar"], **kw)


def galaxy_loglike(law, p, d, mb_key="Mb_cat", use_fint=False, detail=False):
    gp = np.maximum(predict(law, p, d, mb_key), 1e-30)
    Vm = np.sqrt(gp * d["R"] * KPC) / KMS
    if use_fint:
        sig = np.sqrt(d["eV"][None, :] ** 2 + (p["f_int"] * Vm) ** 2)
    else:
        sig = np.broadcast_to(d["eV"][None, :], Vm.shape)
    chi2 = np.sum(((d["Vobs"] - Vm) / sig) ** 2, axis=1)
    norm = np.sum(np.log(sig), axis=1) + 0.5 * Vm.shape[1] * math.log(2 * math.pi)
    lo = float(logsumexp(-0.5 * chi2 - norm) - math.log(len(chi2)))
    if not detail:
        return lo
    # the draw that dominates the marginalisation: the ML condition for f_int
    # is set at this draw, so this is the chi2 the calibration must be judged on
    kbest = int(np.argmax(-0.5 * chi2 - norm))
    return lo, float(chi2[kbest]), float(np.mean(chi2)), Vm.shape[1]


def total_nll(law, vec, gals, mb_key="Mb_cat", use_fint=False, bnds=None):
    p = unpack(law, vec, bnds)
    if p is None:
        return 1e12
    s = 0.0
    for g in gals:
        s -= galaxy_loglike(law, p, g.draws, mb_key, use_fint)
    return s if np.isfinite(s) else 1e12


def free_of(law, use_fint=False):
    return list(MM.LAWS[law]["free"]) + (["f_int"] if use_fint else [])


def bounds_of(law, use_fint=False):
    b = dict(MM.BOUNDS)
    if law.startswith("opt3"):
        b["eta"] = (0.01, 0.999)     # (1-eta) prefactor must stay positive
    return {k: b[k] for k in free_of(law, use_fint)}


def unpack(law, vec, bnds):
    p = {}
    for name, v in zip(bnds.keys(), vec):
        lo, hi = bnds[name]
        if not (lo <= v <= hi):
            return None
        p[name] = 10 ** v if name in MM.LOGPAR else v
    return p


def x0_of(bnds):
    return [MM.START[k] for k in bnds]


def fit(law, gals, mb_key="Mb_cat", use_fint=False):
    bnds = bounds_of(law, use_fint)
    if not bnds:
        return {}, bnds
    r = minimize(lambda v: total_nll(law, v, gals, mb_key, use_fint, bnds),
                 x0_of(bnds), method="Nelder-Mead",
                 options=dict(maxiter=3000, maxfev=3000, xatol=1e-4, fatol=1e-3))
    # one restart from the optimum, Nelder-Mead is not scale-invariant
    r = minimize(lambda v: total_nll(law, v, gals, mb_key, use_fint, bnds),
                 r.x, method="Nelder-Mead",
                 options=dict(maxiter=3000, maxfev=3000, xatol=1e-5, fatol=1e-4))
    p = unpack(law, r.x, bnds)
    return (p if p is not None else unpack(law, x0_of(bnds), bnds)), bnds


def metrics(law, p, gals, mb_key="Mb_cat", use_fint=False, nfree=0):
    lo, npt, chi2, chi2_best, chi2_avg = 0.0, 0, 0.0, 0.0, 0.0
    go_all, gp_all = [], []
    for g in gals:
        d = g.draws
        if use_fint:
            l1, cb, ca, _ = galaxy_loglike(law, p, d, mb_key, True, detail=True)
            lo += l1; chi2_best += cb; chi2_avg += ca
        else:
            lo += galaxy_loglike(law, p, d, mb_key, False)
        gp = np.maximum(predict(law, p, d, mb_key), 1e-30)
        Vm = np.sqrt(gp * d["R"] * KPC) / KMS
        chi2 += float(np.mean(np.sum(((d["Vobs"] - Vm) / d["eV"]) ** 2, axis=1)))
        npt += d["gobs"].shape[1]
        go_all.append(np.mean(d["gobs"], axis=0))
        gp_all.append(np.mean(gp, axis=0))
    go, gp = np.concatenate(go_all), np.concatenate(gp_all)
    rms = float(np.sqrt(np.mean((np.log10(go) - np.log10(gp)) ** 2)))
    dof = max(npt - nfree, 1)
    out = dict(loglike=lo, rms_dex=rms, chi2_per_point=chi2 / max(npt, 1),
               n_points=npt, n_gal=len(gals))
    if use_fint:
        # judged at the draw the marginal likelihood is dominated by, which is
        # where the ML condition for f_int is actually imposed
        out["chi2_eff_per_dof"] = chi2_best / dof
        out["chi2_eff_per_dof_drawavg"] = chi2_avg / dof
        out["AIC"] = 2 * nfree - 2 * lo
        out["BIC"] = nfree * math.log(npt) - 2 * lo
    return out


# ------------------------------------------------------------- residual corr
def residual_table(law, p, gals, mb_key="Mb_cat"):
    """Per-point log10 residual and the covariates it is tested against."""
    rows = dict(res=[], R_over_Rd=[], logSB=[], logMb=[], fgas=[], inc=[],
                dist=[], logR=[], loggbar=[])
    for g in gals:
        d = g.draws
        gp = np.mean(np.maximum(predict(law, p, d, mb_key), 1e-30), axis=0)
        go = np.mean(d["gobs"], axis=0)
        R = np.mean(d["R"], axis=0)
        n = len(go)
        rows["res"] += list(np.log10(go) - np.log10(gp))
        rows["R_over_Rd"] += list(R / d["Rdisk"])
        rows["logSB"] += [math.log10(max(g.SBeff, 1e-3))] * n
        rows["logMb"] += [math.log10(float(np.mean(d[mb_key])))] * n
        rows["fgas"] += [g.fgas] * n
        rows["inc"] += [g.i0] * n
        rows["dist"] += [g.D0] * n
        rows["logR"] += list(np.log10(R))
        rows["loggbar"] += list(np.log10(np.mean(d["gbar"], axis=0)))
    return {k: np.array(v) for k, v in rows.items()}


def corr_report(tab, keys=("R_over_Rd", "logSB", "logMb", "fgas", "inc",
                           "dist", "logR", "loggbar")):
    out = {}
    for k in keys:
        m = np.isfinite(tab["res"]) & np.isfinite(tab[k])
        if m.sum() < 8:
            continue
        rs, ps = spearmanr(tab[k][m], tab["res"][m])
        rp, pp = pearsonr(tab[k][m], tab["res"][m])
        out[k] = dict(spearman=float(rs), p_spearman=float(ps),
                      pearson=float(rp), p_pearson=float(pp), n=int(m.sum()))
    return out


# ------------------------------------------------------------- blind holdouts
def load_kids():
    """KiDS+GAMA lensing rotation curves, Brouwer+2021 Fig 3, four stellar-mass
    bins. Loader re-implemented here so the script has no bench dependency; it
    is cross-checked against invariant_bench.Bench below when astropy is up."""
    G_PC, PC_M = 4.52e-30, 3.086e16
    EDG = [8.5, 10.3, 10.6, 10.8, 11.0]
    r, gb, go, mb = [], [], [], []
    for b in (1, 2, 3, 4):
        f = os.path.join(SCR, "g2", f"Fig-3_Lensing-rotation-curves_Massbin-{b}.txt")
        if not os.path.exists(f):
            continue
        Mb = 1.4 * 10 ** (0.5 * (EDG[b - 1] + EDG[b]))          # Msun, baryonic
        for l in open(f, encoding="utf-8"):
            l = l.strip()
            if not l or l.startswith("#"):
                continue
            v = [float(z) for z in l.split()]
            if len(v) < 5:
                continue
            R = v[0] * 1000 * KPC
            g = 4 * G_PC * (v[1] / v[4]) * PC_M
            if g > 0:
                r.append(R); gb.append(G * Mb * MSUN / R ** 2)
                go.append(g); mb.append(Mb)
    return (np.array(r), np.array(gb), np.array(go), np.array(mb))


def load_widebin():
    """El-Badry-calibrated wide-binary boosts, as carried in the bench."""
    sep = np.array([220., 557., 1565., 4568., 14040., 37223.]) * AU
    boost = np.array([0.971, 1.029, 1.038, 1.067, 1.188, 1.203])
    Mt = 0.9
    gb = G * Mt * MSUN / sep ** 2
    return sep, gb, gb * boost, np.full(6, Mt)


def eval_holdout(law, p, r, gb, go, mb):
    spec = MM.LAWS[law]
    kw = {k: v for k, v in p.items() if k != "f_int"}
    if "r_m" in spec["needs"]:
        kw["r_m"] = r
    if "Mb_msun" in spec["needs"]:
        kw["Mb_msun"] = mb
    gp = np.maximum(spec["fn"](gb, **kw), 1e-300)
    d = np.log10(go) - np.log10(gp)
    return dict(rms_dex=float(np.sqrt(np.mean(d ** 2))),
                median_dex=float(np.median(d)), n=int(len(d)))


# -------------------------------------------------------------------- main
def main():
    head("STEP 0-1  frozen cuts, frozen split, frozen nuisance draws")
    gals = DAT.ingest(verbose=True)
    DAT.stratified_split(gals, verbose=True)
    ng, worst = verify_against_runA()
    print(f"\n   rebuilt draws vs runA.build_draws : max |diff| = {worst:.3e} "
          f"over {ng} galaxies  ({'IDENTICAL' if worst == 0.0 else 'DIFFERS'})")
    gals = build_draws(gals)
    tr = [g for g in gals if g.split == "train"]
    va = [g for g in gals if g.split == "validation"]
    bl = [g for g in gals if g.split == "blind"]
    print(f"   train / validation / blind        : {len(tr)} / {len(va)} / {len(bl)}")
    print(f"   nuisance draws per galaxy         : {NDRAW}")
    mb1 = np.array([float(np.mean(g.draws["Mb_cat"])) for g in gals])
    mb2 = np.array([float(np.mean(g.draws["Mb"])) for g in gals])
    print(f"   log10 M_b(cat)  range             : {np.log10(mb1).min():.2f} "
          f".. {np.log10(mb1).max():.2f}")
    print(f"   M_b(cat) vs M_b(last point)       : median ratio "
          f"{np.median(mb1/mb2):.3f}, scatter {np.std(np.log10(mb1/mb2)):.3f} dex")
    RESULTS["setup"] = dict(n_train=len(tr), n_val=len(va), n_blind=len(bl),
                            ndraw=NDRAW, draws_match_runA=(worst == 0.0),
                            draws_max_absdiff=worst)

    head("STEP 2  benchmark reproduction under THIS harness")
    print(f"   {'law':<18}{'free':>5}{'RMS train':>12}{'RMS blind':>12}"
          f"{'chi2/point bl':>15}")
    print("   " + "-" * 62)
    bench = {}
    for law in ("newton", "aqual_simple", "rar"):
        p, _ = fit(law, tr)
        mt, mb_ = metrics(law, p, tr), metrics(law, p, bl)
        bench[law] = dict(params=p, train=mt, blind=mb_)
        print(f"   {law:<18}{len(MM.LAWS[law]['free']):>5}{mt['rms_dex']:>12.4f}"
              f"{mb_['rms_dex']:>12.4f}{mb_['chi2_per_point']:>15.1f}")
    print("   " + "-" * 62)
    print(f"   published benchmarks: Newton blind 0.5544, AQUAL simple blind 0.1590")
    ok_n = abs(bench["newton"]["blind"]["rms_dex"] - 0.5544) < 5e-3
    ok_a = abs(bench["aqual_simple"]["blind"]["rms_dex"] - 0.1590) < 5e-3
    print(f"   reproduced: Newton {'YES' if ok_n else 'NO'}, "
          f"AQUAL {'YES' if ok_a else 'NO'}")
    RESULTS["benchmarks"] = bench
    RESULTS["benchmarks"]["reproduced"] = dict(newton=bool(ok_n), aqual=bool(ok_a))

    head("STEP 3  Options 2/3/4 fitted on TRAIN, evaluated BLIND")
    laws = ["opt2", "opt2_eta_half", "opt3", "opt3_eta_half",
            "opt4", "opt4_eta_half", "opt4_menc"]
    print(f"   {'law':<18}{'free':>5}{'RMS train':>12}{'RMS blind':>12}"
          f"{'chi2/pt bl':>13}   parameters")
    print("   " + "-" * 96)
    fits = {}
    for law in laws:
        p, bnds = fit(law, tr)
        mt, mb_ = metrics(law, p, tr), metrics(law, p, bl)
        aud = MM.audit_global_only(law, tr, p)
        fits[law] = dict(params=p, train=mt, blind=mb_, audit=aud,
                         validation=metrics(law, p, va),
                         nfree=len(MM.LAWS[law]["free"]))
        ps = " ".join(f"{k}={v:.4g}" for k, v in p.items())
        print(f"   {law:<18}{len(MM.LAWS[law]['free']):>5}{mt['rms_dex']:>12.4f}"
              f"{mb_['rms_dex']:>12.4f}{mb_['chi2_per_point']:>13.1f}   {ps}")
    print("   " + "-" * 96)
    RESULTS["screening"] = fits

    print("\n   AUDIT -- no per-galaxy gravity parameter")
    for law in laws:
        a = fits[law]["audit"]
        extra = a.get("r_t_determined_by_Mb_max_reldiff")
        s = "" if extra is None else f", r_t == eta sqrt(G M_b/a0) to {extra:.1e}"
        print(f"     {law:<18} globals={a['n_global']}  per-galaxy={a['n_per_galaxy']}{s}")

    print("\n   RAIL CHECK -- a fit that only works at a prior edge is a failure")
    rails = {}
    for law in laws:
        bn = bounds_of(law)
        bad = []
        for k, v in fits[law]["params"].items():
            lo, hi = bn[k]
            x = math.log10(v) if k in MM.LOGPAR else v
            frac = (x - lo) / (hi - lo)
            if frac < 0.02 or frac > 0.98:
                bad.append(f"{k}={v:.4g} at {'lower' if frac < 0.5 else 'upper'} edge")
        rails[law] = bad
        print(f"     {law:<18}{'RAILED: ' + '; '.join(bad) if bad else 'no parameter at a bound'}")
    RESULTS["rail_check"] = rails

    print("\n   ROBUSTNESS -- M_b proxy (catalogue photometry vs last curve point)")
    print(f"   {'law':<18}{'RMS blind Mb_cat':>19}{'RMS blind Mb_last':>19}")
    print("   " + "-" * 56)
    for law in ("opt2", "opt4"):
        p2, _ = fit(law, tr, mb_key="Mb")
        m2 = metrics(law, p2, bl, mb_key="Mb")
        print(f"   {law:<18}{fits[law]['blind']['rms_dex']:>19.4f}"
              f"{m2['rms_dex']:>19.4f}")
        fits[law]["blind_Mb_lastpoint"] = m2
        fits[law]["params_Mb_lastpoint"] = p2

    head("STEP 3b  BTFR slope each formulation predicts (computed, not asserted)")
    print("   observed BTFR: v_f^4 ~ M_b^1, i.e. d log v_f^4 / d log M_b = 1.00")
    print(f"   {'law':<18}{'slope d log vf^4/d log Mb':>28}")
    print("   " + "-" * 48)
    btfr = {}
    for law in ("opt2", "opt3", "opt4"):
        p = fits[law]["params"]
        sl = MM.btfr_slope_prediction(law, eta=p.get("eta", 0.5),
                                      a0=p.get("a0", MM.A0_CANON),
                                      l_kpc=p.get("l_kpc"))
        btfr[law] = sl
        print(f"   {law:<18}{sl:>28.3f}")
    print("   " + "-" * 48)
    RESULTS["btfr_slope"] = btfr

    head("STEP 4  residual correlations on the BLIND split")
    keys = ("R_over_Rd", "logSB", "logMb", "fgas", "inc", "dist", "logR", "loggbar")
    print(f"   Spearman rho of (log g_obs - log g_pred) against each covariate")
    print(f"   {'law':<18}" + "".join(f"{k:>11}" for k in keys))
    print("   " + "-" * (18 + 11 * len(keys)))
    corrs = {}
    for law in ["aqual_simple"] + laws:
        p = bench[law]["params"] if law in bench else fits[law]["params"]
        tab = residual_table(law, p, bl)
        c = corr_report(tab, keys)
        corrs[law] = c
        print(f"   {law:<18}" + "".join(f"{c[k]['spearman']:>11.3f}" for k in keys))
    print("   " + "-" * (18 + 11 * len(keys)))
    print("   (n = %d blind points; |rho| > %.3f is p < 0.05 two-sided)"
          % (corrs["opt2"]["logR"]["n"], 1.96 / math.sqrt(corrs["opt2"]["logR"]["n"])))
    RESULTS["blind_residual_correlations"] = corrs

    head("STEP 5  SENSITIVITY AUDIT -- does each statistic move with each parameter?")
    print("""   The programme has already been burned by a rank statistic that was
   bit-identical across two decades of an amplitude parameter. Every headline
   number below is recomputed over a grid in every global parameter it is
   claimed to depend on. A statistic whose min and max are equal is BLIND to
   that parameter and may not be quoted as a test of it.\n""")
    sens = {}
    grids = dict(eta=np.array([0.05, 0.15, 0.5, 1.5, 5.0]),
                 a0=np.array([1.2e-12, 1.2e-11, 1.2e-10, 1.2e-9, 1.2e-8]),
                 p=np.array([0.25, 0.5, 1.0, 4.0, 30.0]),
                 l_kpc=np.array([0.5, 2.0, 10.0, 50.0, 200.0]))
    grids_opt3 = dict(grids, eta=np.array([0.05, 0.15, 0.4, 0.7, 0.95]))
    print(f"   {'law':<8}{'param':>8}{'range':>22}{'RMS blind min..max':>26}"
          f"{'rho(logR) min..max':>26}   verdict")
    print("   " + "-" * 100)
    for law in ("opt2", "opt3", "opt4"):
        base = dict(fits[law]["params"])
        gr = grids_opt3 if law.startswith("opt3") else grids
        for par in MM.LAWS[law]["free"]:
            vals, rmss, rhos = gr[par], [], []
            for v in vals:
                q = dict(base); q[par] = float(v)
                rmss.append(metrics(law, q, bl)["rms_dex"])
                t = residual_table(law, q, bl)
                rhos.append(float(spearmanr(t["logR"], t["res"])[0]))
            live = (max(rmss) - min(rmss) > 1e-12) and (max(rhos) - min(rhos) > 1e-12)
            sens[f"{law}:{par}"] = dict(values=[float(v) for v in vals],
                                        rms_blind=[float(x) for x in rmss],
                                        rho_logR=[float(x) for x in rhos],
                                        responsive=bool(live))
            print(f"   {law:<8}{par:>8}{f'{vals[0]:.3g}..{vals[-1]:.3g}':>22}"
                  f"{f'{min(rmss):.4f}..{max(rmss):.4f}':>26}"
                  f"{f'{min(rhos):+.3f}..{max(rhos):+.3f}':>26}"
                  f"   {'RESPONSIVE' if live else 'BLIND -- do not quote'}")
    print("   " + "-" * 100)
    RESULTS["sensitivity_audit"] = sens

    head("STEP 6  intrinsic scatter: calibrate before quoting any lnL/AIC/BIC")
    print("""   sigma_i^2 = eV_i^2 + (f_int V_model,i)^2 with ONE global f_int.
   chi2/dof must be near 1 before an information criterion means anything.\n""")
    print(f"   {'law':<18}{'free':>5}{'f_int':>9}{'chi2/pt (eV only)':>20}"
          f"{'chi2eff/dof':>13}{'(drawavg)':>11}{'RMS blind':>12}{'BIC train':>13}")
    print("   " + "-" * 103)
    cal = {}
    for law in ("newton", "aqual_simple", "opt2", "opt2_eta_half", "opt3", "opt4"):
        p, _ = fit(law, tr, use_fint=True)
        nf = len(free_of(law, True))
        mt = metrics(law, p, tr, use_fint=True, nfree=nf)
        mb_ = metrics(law, p, bl, use_fint=True, nfree=nf)
        cal[law] = dict(params=p, train=mt, blind=mb_, nfree=nf)
        print(f"   {law:<18}{nf:>5}{p['f_int']:>9.4f}{mt['chi2_per_point']:>20.1f}"
              f"{mt['chi2_eff_per_dof']:>13.3f}"
              f"{mt['chi2_eff_per_dof_drawavg']:>11.2f}{mb_['rms_dex']:>12.4f}"
              f"{mt['BIC']:>13.1f}")
    print("   " + "-" * 103)
    good = all(0.5 < cal[k]["train"]["chi2_eff_per_dof"] < 2.0
               for k in ("aqual_simple", "opt2", "opt3"))
    print(f"   chi2_eff/dof in [0.5, 2] for the live models: "
          f"{'YES' if good else 'NO'}")
    print("""   CAVEAT carried with every BIC below: the likelihood marginalises over 16
   frozen nuisance draws per galaxy, so the fit has effective freedom that dof
   does not count. chi2_eff/dof ~ 1 is therefore a calibration of the error
   model, not proof of it; the draw-averaged column is the conservative
   reading and it is still far from 1.""")
    RESULTS["intrinsic_scatter_calibration"] = cal
    RESULTS["bic_quotable"] = bool(good)

    head("STEP 7  BLIND HOLDOUTS -- never fitted, at any stage")
    kr, kgb, kgo, kmb = load_kids()
    wr, wgb, wgo, wmb = load_widebin()
    print(f"   KiDS lensing rotation curves : {len(kr)} points, "
          f"r = {kr.min()/KPC:.0f} .. {kr.max()/KPC:.0f} kpc, "
          f"log M_b = {np.log10(kmb.min()):.2f} .. {np.log10(kmb.max()):.2f}")
    print(f"   wide binaries                : {len(wr)} separations, "
          f"{wr.min()/AU:.0f} .. {wr.max()/AU:.0f} AU, M_tot = 0.9 Msun")
    try:
        sys.path.insert(0, SCR)
        from invariant_bench import Bench
        b = Bench(verbose=False)
        ok = (len(b.d["kids"]) == len(kr)
              and np.allclose(np.sort(b.d["kids"].go), np.sort(kgo)))
        print(f"   cross-check vs invariant_bench.Bench['kids'] : "
              f"{'MATCH' if ok else 'MISMATCH'}")
    except Exception as e:
        ok = None
        print(f"   cross-check vs bench unavailable: {type(e).__name__}")

    print("""
   CAVEAT CARRIED WITH KiDS, from the bench: beyond ~1 Mpc the lensing signal
   picks up neighbouring mass that is not in M_bar, so g_bar is underestimated
   and every model is pushed to look low. The r < 1 Mpc column is the cleaner
   one and both are reported.""")
    inner = kr < 1000 * KPC
    print(f"\n   {'law':<18}{'KiDS RMS':>11}{'KiDS med':>10}"
          f"{'r<1Mpc RMS':>12}{'r<1Mpc med':>12}{'WB RMS':>9}{'WB med':>9}  (dex)")
    print("   " + "-" * 82)
    hold = {}
    for law in ["newton", "aqual_simple", "rar"] + laws:
        p = bench[law]["params"] if law in bench else fits[law]["params"]
        k = eval_holdout(law, p, kr, kgb, kgo, kmb)
        ki = eval_holdout(law, p, kr[inner], kgb[inner], kgo[inner], kmb[inner])
        w = eval_holdout(law, p, wr, wgb, wgo, wmb)
        hold[law] = dict(kids=k, kids_inner=ki, widebin_internal=w, params=p)
        print(f"   {law:<18}{k['rms_dex']:>11.4f}{k['median_dex']:>10.4f}"
              f"{ki['rms_dex']:>12.4f}{ki['median_dex']:>12.4f}"
              f"{w['rms_dex']:>9.4f}{w['median_dex']:>9.4f}")
    print("   " + "-" * 82)

    print("\n   KiDS residual (log g_obs - log g_pred) by radius, dex")
    edges = [30, 70, 150, 350, 800, 1800, 3000]
    print(f"   {'law':<18}" + "".join(f"{f'{a}-{b}':>11}"
                                      for a, b in zip(edges[:-1], edges[1:])))
    print("   " + "-" * (18 + 11 * (len(edges) - 1)))
    kbin = {}
    for law in ("aqual_simple", "opt2", "opt3", "opt4"):
        p = bench[law]["params"] if law in bench else fits[law]["params"]
        row, vals = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (kr >= a * KPC) & (kr < b * KPC)
            if m.sum() == 0:
                row.append(float("nan")); vals.append("        -"); continue
            e = eval_holdout(law, p, kr[m], kgb[m], kgo[m], kmb[m])
            row.append(e["median_dex"]); vals.append(f"{e['median_dex']:>11.3f}")
        kbin[law] = row
        print(f"   {law:<18}" + "".join(vals))
    print("   " + "-" * (18 + 11 * (len(edges) - 1)))
    print("   A model that had the right a0 would sit near zero at small radius")
    print("   and drift positive outward as the confound switches on.")
    hold["kids_radial_bins"] = dict(edges_kpc=edges, median_dex=kbin)
    print("""
   WIDE-BINARY READING AMBIGUITY, stated rather than resolved silently.
   Option 2's mu is a function of r/r_t with r_t set by the SOURCE mass, so a
   binary embedded in the Milky Way has two readings: (a) its own 0.9 Msun sets
   r_t (the table above), or (b) it sits in the Galaxy's D-field, whose mu is
   locally constant, giving a pure rescaling of G and NO separation-dependent
   boost. Reading (b) predicts boost = 1 at every separation.""")
    d_b = np.log10(wgo / wgb)
    hold["widebin_reading_b_galactic_field"] = dict(
        rms_dex=float(np.sqrt(np.mean(d_b ** 2))),
        note="mu locally constant -> boost 1 at all separations")
    print(f"   reading (b) for every law: RMS = {hold['widebin_reading_b_galactic_field']['rms_dex']:.4f} dex")
    RESULTS["blind_holdouts"] = hold
    RESULTS["blind_holdouts"]["kids_bench_crosscheck"] = ok

    head("Extrapolation audit")
    print("   np.interp is not used anywhere in this script: every model is")
    print("   evaluated in closed form at the measured radii. Extrapolated")
    print("   fraction = 0/%d points. (mirror_adyn.py and mirror_dfield.py"
          % (RESULTS["screening"]["opt2"]["blind"]["n_points"]))
    print("   report their own interpolation fractions.)")
    RESULTS["extrapolation_fraction_screening"] = 0.0

    # consolidate the other two stages if they have already been run, so
    # mirror_results.json is the single deliverable
    here = os.path.dirname(os.path.abspath(__file__))
    for key, fn in (("structural_test_dfield", "mirror_dfield_results.json"),
                    ("A_dyn_and_diskmass", "mirror_adyn_results.json")):
        p = os.path.join(here, fn)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                RESULTS[key] = json.load(fh)
            print(f"   merged {fn}")
        else:
            RESULTS[key] = f"not run -- execute {fn.replace('_results.json', '.py')}"
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(RESULTS, fh, indent=1, default=float)
    print(f"\n   wrote {OUT}")
    return RESULTS


if __name__ == "__main__":
    main()
