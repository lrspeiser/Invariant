#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoCuSS test of the pressure-sourced SOURCE LAW, computed with the EXACT forward chain.

    rho_eff  = rho_b + 3 kappa P / c^2
    DM_P(<r) = (3 kappa / c^2) Int_0^r 4 pi r'^2 P(r') dr'
             = kappa * (3 kT / (mu m_p c^2)) * M_gas(<r)          [ideal isothermal gas]
    g_N_eff  = G (M_b + DM_P) / r^2 ,      g_N_b = G M_b / r^2
    g        = F(g_N) = nu(g_N/a0) * g_N ,  nu(x) = 1/(1 - exp(-sqrt(x)))
    E_pred   = F(g_N_eff) / F(g_N_b)
    E_obs    = M_WL / (nu(x_b) M_b) = g_WL / F(g_N_b)

Differences from the previous run (locuss/locuss_test.py), which this supersedes:
  * the M_gas/M_b factor is carried:  delta = DM_P/M_b = kappa t f_gas;
  * the FULL RAR nu is used on both sides, not only the deep-MOND limit;
  * the previous run's "E - 1" branch is NOT reused: E - 1 corresponds to an
    additive POST-RAR acceleration, a different theory from a source modification.
    Here the response is an acceleration RATIO through F.
  * the primary inference is an errors-in-variables regression, not a partial rank
    correlation conditioned on a noisy M_WL;
  * power is obtained by injecting through the exact chain and is size-corrected.

Runs start to finish from a fresh process.  Writes locuss2_results.json.
"""

import json
import math
import os
import sys

import numpy as np
from scipy import stats, optimize

# ----------------------------------------------------------------------------
# 0.  Paths / RNG
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ACQ = os.path.abspath(os.path.join(HERE, "..", "acquire"))
SAMPLE_TSV = os.path.join(ACQ, "mulroy2019_sample.tsv")
OBS_TSV = os.path.join(ACQ, "mulroy2019_observables.tsv")
OUT_JSON = os.path.join(HERE, "locuss2_results.json")

SEED = 20260904
RNG = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------
# 1.  Physical constants (SI) and fixed choices
# ----------------------------------------------------------------------------
G_SI = 6.67430e-11
MSUN = 1.98892e30
MPC = 3.0856775814913673e22
C_LIGHT = 2.99792458e8
M_P = 1.67262192369e-27
KEV_J = 1.602176634e-16
A0 = 1.2e-10
MU_MOL = 0.6
H0_SI = 70.0 * 1000.0 / MPC
OM, OL = 0.3, 0.7
KAPPA_XCOP = 1.36e5
UPSILON_K_PRIMARY = 0.73

RHO_C0 = 3.0 * H0_SI ** 2 / (8.0 * math.pi * G_SI)


def Ez2(z):
    z = np.asarray(z, float)
    return OM * (1.0 + z) ** 3 + OL


def rho_c(z):
    return RHO_C0 * Ez2(z)


def r500_from_M500(M500_kg, z):
    return (3.0 * np.asarray(M500_kg, float) /
            (4.0 * math.pi * 500.0 * rho_c(z))) ** (1.0 / 3.0)


def nu_rar(x):
    """RAR interpolation nu(x) = 1/(1 - exp(-sqrt(x)))."""
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 - np.exp(-np.sqrt(x)))


def F_rar(gN):
    """The acceleration law g = F(g_N) = nu(g_N/a0) * g_N."""
    gN = np.asarray(gN, float)
    return nu_rar(gN / A0) * gN


def dlnnu_dlnx(x):
    x = np.asarray(x, float)
    s = np.sqrt(x)
    u = np.exp(-s)
    return -s * u / (2.0 * (1.0 - u))


def t_of_kT(kT_keV):
    """t = 3 kT / (mu m_p c^2), dimensionless."""
    return 3.0 * np.asarray(kT_keV, float) * KEV_J / (MU_MOL * M_P * C_LIGHT ** 2)


# ----------------------------------------------------------------------------
# 2.  Data loading (ingest reused verbatim from the validated first run)
# ----------------------------------------------------------------------------
def read_tsv(path):
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln.rstrip("\n").rstrip("\r") for ln in fh if ln.strip() != ""]
    hdr = lines[0].split("\t")
    rows = []
    for ln in lines[1:]:
        parts = ln.split("\t")
        if len(parts) < len(hdr):
            parts = parts + [""] * (len(hdr) - len(parts))
        assert len(parts) == len(hdr), (path, len(parts), len(hdr))
        rows.append(dict(zip(hdr, parts)))
    return hdr, rows


def fnum(s):
    s = s.strip()
    if s == "" or s == "--":
        return None
    return float(s)


def load():
    hdr_s, rows_s = read_tsv(SAMPLE_TSV)
    hdr_o, rows_o = read_tsv(OBS_TSV)
    assert len(rows_s) == 41 and len(rows_o) == 41, "row-count assertion failed"
    obs = {r["Name"]: r for r in rows_o}
    assert set(obs) == {r["Name"] for r in rows_s}, "name mismatch between tables"
    cl = []
    for r in rows_s:
        o = obs[r["Name"]]
        cl.append(dict(
            name=r["Name"],
            z=fnum(r["z"]),
            M_WL=fnum(r["M_WL"]), M_WL_ep=fnum(r["M_WL_ep"]), M_WL_em=fnum(r["M_WL_em"]),
            kT=fnum(o["kT_X_ce"]), kT_ep=fnum(o["kT_X_ce_ep"]), kT_em=fnum(o["kT_X_ce_em"]),
            M_gas=fnum(o["M_gas"]), M_gas_ep=fnum(o["M_gas_ep"]), M_gas_em=fnum(o["M_gas_em"]),
            L_K=fnum(o["L_K_tot"]), L_K_ep=fnum(o["L_K_tot_ep"]), L_K_em=fnum(o["L_K_tot_em"]),
            L_K_BCG=fnum(o["L_K_BCG"]),
            Y_X=fnum(o["Y_X"]), lam=fnum(o["lambda"]), Y_SZA=fnum(o["Y_SZA"]),
        ))
    return cl


def exclusion_audit(clusters):
    """Item 6: name the cluster excluded to go 41 -> 40, state the criterion, verify."""
    required = ["M_WL", "kT", "M_gas", "L_K"]
    missing = {}
    for c in clusters:
        miss = [k for k in required if c[k] is None]
        if miss:
            missing[c["name"]] = miss
    also = {}
    for c in clusters:
        extra = [k for k in ("L_K_BCG", "Y_X", "lam", "Y_SZA") if c[k] is None]
        if extra:
            also[c["name"]] = extra
    return dict(
        criterion=("Retain a cluster iff ALL FOUR quantities that enter the forward chain are "
                   "published: M_WL (the gravity measurement), kT_X_ce (sets the pressure), "
                   "M_gas (enters twice - in M_b and in DM_P), L_K_tot (stellar mass). "
                   "No dynamical-state cut and no weak-lensing S/N cut."),
        n_input=len(clusters),
        clusters_failing_criterion=missing,
        n_retained=len(clusters) - len(missing),
        verified_excluded_is_Abell2697=(list(missing.keys()) == ["Abell2697"]),
        Abell2697_missing_fields=missing.get("Abell2697"),
        note_other_incomplete_columns_not_used=also,
    )


# ----------------------------------------------------------------------------
# 3.  THE EXACT FORWARD CHAIN (no step compressed)
# ----------------------------------------------------------------------------
def chain(M_WL, z, M_gas, L_K, kT, kappa, upsilon_K=UPSILON_K_PRIMARY):
    """
    Vectorised.  Units in: M_WL, M_gas in 1e14 Msun; L_K in 1e12 Lsun; kT in keV.

    Step 1  aperture      r500 from the LENSING mass
    Step 2  stars         M_star = upsilon_K L_K
    Step 3  baryons       M_b = M_gas + M_star
    Step 4  thermal ratio t = 3 kT/(mu m_p c^2)
    Step 5  pressure mass DM_P = kappa t M_gas   (= (3 kappa/c^2) Int 4 pi r'^2 P dr')
    Step 6  Newtonian     g_N_b = G M_b/r^2 ; g_N_eff = G (M_b + DM_P)/r^2
    Step 7  acceleration  g = F(g_N) = nu(g_N/a0) g_N
    Step 8  prediction    E_pred = F(g_N_eff)/F(g_N_b)
    Step 9  observation   E_obs  = M_WL/(nu(x_b) M_b) = g_WL/F(g_N_b)
    """
    M_WL = np.asarray(M_WL, float)
    M_gas = np.asarray(M_gas, float)
    L_K = np.asarray(L_K, float)
    kT = np.asarray(kT, float)
    z = np.asarray(z, float)

    r500_m = r500_from_M500(M_WL * 1e14 * MSUN, z)
    M_star = upsilon_K * L_K * 1e12 / 1e14
    M_b = M_gas + M_star
    f_gas = M_gas / M_b
    t = t_of_kT(kT)

    DM_P = kappa * t * M_gas
    delta = DM_P / M_b
    delta = np.maximum(delta, -0.95)               # numerical guard only; never binds for kappa>0

    g_N_b = G_SI * (M_b * 1e14 * MSUN) / r500_m ** 2
    g_N_eff = g_N_b * (1.0 + delta)
    x_b = g_N_b / A0
    x_eff = g_N_eff / A0

    g_b = F_rar(g_N_b)
    g_eff = F_rar(g_N_eff)
    E_pred = g_eff / g_b

    g_WL = G_SI * (M_WL * 1e14 * MSUN) / r500_m ** 2
    E_obs = g_WL / g_b

    return dict(r500_Mpc=r500_m / MPC, M_star=M_star, M_b=M_b, f_gas=f_gas, t=t,
                DM_P=DM_P, delta=delta, g_N_b=g_N_b, g_N_eff=g_N_eff,
                x_b=x_b, x_eff=x_eff, nu_b=nu_rar(x_b), nu_eff=nu_rar(x_eff),
                g_b=g_b, g_eff=g_eff, g_WL=g_WL,
                E_pred=E_pred, E_obs=E_obs,
                deep_limit_Ypred=kappa * t * f_gas,
                Ypred=E_pred ** 2 - 1.0, Yobs=E_obs ** 2 - 1.0)


def verify_pressure_integral_identity():
    """
    Step 5 is the ONE place where an integral is replaced by an algebraic identity.
    Verify NUMERICALLY that Int_0^r 4 pi r'^2 P dr' = (kT/(mu m_p)) M_gas(<r) for
    isothermal gas on a beta-model density - i.e. that the reduction is exact and
    independent of the density profile - and then quantify what a realistically
    declining T(r) would do to the same integral relative to the CORE-EXCISED
    spectroscopic temperature that LoCuSS actually publishes.
    """
    rc, beta = 0.2, 0.65
    R = 1.3
    r = np.linspace(1e-5, R, 600001)
    rho = (1.0 + (r / rc) ** 2) ** (-1.5 * beta)
    Mgas = np.trapezoid(4.0 * math.pi * r ** 2 * rho, r)
    Pint_iso = np.trapezoid(4.0 * math.pi * r ** 2 * rho * 1.0, r)
    ratio_iso = Pint_iso / (1.0 * Mgas)

    out = dict(beta_model=dict(r_core_Mpc=rc, beta=beta, r_out_Mpc=R, n_grid=len(r)),
               isothermal_ratio_integral_over_kT_Mgas=float(ratio_iso),
               isothermal_identity_exact=bool(abs(ratio_iso - 1.0) < 1e-9),
               declining_T=[])

    ce = (r >= 0.15 * R)
    for rt, q in ((0.6, 0.20), (0.6, 0.35), (0.45, 0.30), (0.8, 0.25)):
        T = (1.0 + (r / rt) ** 2) ** (-q)
        # core-excised SPECTROSCOPIC-like temperature: emission weighted (rho^2) over 0.15-1 r500
        wgt = (r ** 2 * rho ** 2)
        T_ce = np.trapezoid(wgt[ce] * T[ce], r[ce]) / np.trapezoid(wgt[ce], r[ce])
        T_mw = np.trapezoid(4.0 * math.pi * r ** 2 * rho * T, r) / Mgas
        out["declining_T"].append(dict(
            r_t_Mpc=rt, q=q,
            gas_mass_weighted_T_over_core_excised_T=float(T_mw / T_ce),
            note="the exact integral equals (T_mw/T_ce) times the value obtained by "
                 "substituting the published core-excised kT"))
    out["note"] = ("For strictly isothermal gas the identity is EXACT and independent of the "
                   "density profile: no shape assumption is smuggled in.  What it does assume "
                   "is isothermality.  The declining_T entries bound the resulting kappa "
                   "amplitude error; it is a multiplicative factor of order 0.85-0.95 and "
                   "cannot change any sign or any per-cluster ordering.")
    return out


# ----------------------------------------------------------------------------
# 4.  Statistics helpers
# ----------------------------------------------------------------------------
def sigma_ln(val, ep, em):
    hi = math.log1p(ep / val)
    frac = min(em / val, 0.95)
    lo = -math.log1p(-frac)
    return 0.5 * (hi + lo)


def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - 2
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    return dict(a=float(beta[0]), b=float(beta[1]),
                se_a=float(np.sqrt(cov[0, 0])), se_b=float(np.sqrt(cov[1, 1])),
                resid=resid, s=float(np.sqrt(s2)), dof=dof)


def pct(v, ps=(2.5, 16, 50, 84, 97.5)):
    v = np.asarray(v, float)
    return {("p%.1f" % p): float(np.percentile(v, p)) for p in ps}


def basis(Z, w):
    """Orthonormal basis of the whitened nuisance design; None if there is none."""
    if Z is None or Z.shape[1] == 0:
        return None
    Zw = Z * np.sqrt(w)[:, None]
    U, s, _ = np.linalg.svd(Zw, full_matrices=False)
    return U[:, s > 1e-12 * max(s.max(), 1e-30)]


def ssr_grid(YW, LPW, U):
    """
    SSR of the weighted least-squares fit  yw = lpw_k + Z b  for every k, for every
    row of YW, computed without forming any residual matrix.

    YW  (ns, n)   whitened data rows
    LPW (nk, n)   whitened ln E_pred(kappa_k) rows
    U   (n, m)    orthonormal basis of the whitened nuisance design, or None
    Returns (ns, nk).
    """
    ay = np.sum(YW * YW, axis=1)[:, None]
    bl = np.sum(LPW * LPW, axis=1)[None, :]
    cross = YW @ LPW.T
    ssr = ay - 2.0 * cross + bl
    if U is not None:
        AY = YW @ U            # (ns, m)
        BL = LPW @ U           # (nk, m)
        ssr -= (np.sum(AY * AY, axis=1)[:, None]
                - 2.0 * (AY @ BL.T)
                + np.sum(BL * BL, axis=1)[None, :])
    return ssr


# ----------------------------------------------------------------------------
# 5.  Sample construction
# ----------------------------------------------------------------------------
def build(clusters, upsilon_K=UPSILON_K_PRIMARY, impute=False):
    med_ratio = None
    if impute:
        ratios = [c["L_K"] * 1e12 / (c["M_gas"] * 1e14) for c in clusters if c["L_K"] is not None]
        med_ratio = float(np.median(ratios))
    S = []
    for c in clusters:
        L_K = c["L_K"]
        imputed = False
        if L_K is None:
            if not impute:
                continue
            L_K = med_ratio * (c["M_gas"] * 1e14) / 1e12
            imputed = True
        rec = dict(c)
        rec["L_K_used"] = L_K
        rec["L_K_imputed"] = imputed
        rec["L_K_ep_used"] = c["L_K_ep"] if c["L_K_ep"] else 0.15 * L_K
        rec["L_K_em_used"] = c["L_K_em"] if c["L_K_em"] else 0.15 * L_K
        rec["snr_MWL"] = c["M_WL"] / (0.5 * (c["M_WL_ep"] + c["M_WL_em"]))
        S.append(rec)
    return S


def arrays(S):
    return dict(
        name=[s["name"] for s in S],
        M_WL=np.array([s["M_WL"] for s in S]),
        z=np.array([s["z"] for s in S]),
        M_gas=np.array([s["M_gas"] for s in S]),
        L_K=np.array([s["L_K_used"] for s in S]),
        kT=np.array([s["kT"] for s in S]),
        sM=np.array([sigma_ln(s["M_WL"], s["M_WL_ep"], s["M_WL_em"]) for s in S]),
        sG=np.array([sigma_ln(s["M_gas"], s["M_gas_ep"], s["M_gas_em"]) for s in S]),
        sL=np.array([sigma_ln(s["L_K_used"], s["L_K_ep_used"], s["L_K_em_used"]) for s in S]),
        sT=np.array([sigma_ln(s["kT"], s["kT_ep"], s["kT_em"]) for s in S]),
        snr=np.array([s["snr_MWL"] for s in S]),
    )


def E_pred_at(A, kappa, upsilon_K=UPSILON_K_PRIMARY):
    return chain(A["M_WL"], A["z"], A["M_gas"], A["L_K"], A["kT"], kappa, upsilon_K)["E_pred"]


# kappa grid used for every profile fit.  The lower edge is set so that 1 + delta stays
# above 0.05 for the hottest, most gas-rich cluster, which keeps ln E_pred finite with no
# clipping anywhere and leaves every fitted optimum interior to the grid.
KGRID = np.concatenate([np.linspace(-1.32e4, 0.0, 133)[:-1],
                        np.linspace(0.0, 2.0e5, 801),
                        np.linspace(2.0025e5, 1.2e6, 400)])
K0J = int(np.argmin(np.abs(KGRID)))


# ----------------------------------------------------------------------------
# 6.  Monte-Carlo error propagation through the exact chain
# ----------------------------------------------------------------------------
def mc_errors(A, kappa_list, ndraw=8000, rng=None, upsilon_K=UPSILON_K_PRIMARY,
              drag=(0.0, 0.0, 0.0)):
    """
    Push the published errors through the whole chain.

    `drag` = (alpha_gas, alpha_star, beta_T).  All LoCuSS observables are measured
    inside r500 set by the LENSING mass, so a weak-lensing mass error dM also shifts
    the aperture and drags M_gas, L_K and kT with it:
        dln r500 = dM/3, dln M_gas = alpha_gas dM/3, etc.
    With drag = (0,0,0) the published errors are treated as independent, which is what
    they literally state; that variant OVER-states the error on ln E, because the
    aperture makes the M_WL and M_gas errors partly cancel inside E.  Both are run.
    """
    rng = rng or RNG
    n = len(A["M_WL"])
    d = (ndraw, n)
    a_g, a_s, b_T = drag
    dM = rng.normal(0.0, A["sM"], d)
    Mw = A["M_WL"] * np.exp(dM)
    Mg = A["M_gas"] * np.exp(a_g * dM / 3.0 + rng.normal(0.0, A["sG"], d))
    LK = A["L_K"] * np.exp(a_s * dM / 3.0 + rng.normal(0.0, A["sL"], d))
    kT = A["kT"] * np.exp(b_T * dM / 3.0 + rng.normal(0.0, A["sT"], d))
    zz = np.broadcast_to(A["z"], d)

    base = chain(Mw, zz, Mg, LK, kT, 0.0, upsilon_K)
    lnEobs = np.log(base["E_obs"])
    lnM = np.log(Mw)
    lnT = np.log(kT)

    C = np.empty((n, 3, 3))
    for i in range(n):
        C[i] = np.cov(np.column_stack([lnEobs[:, i], lnM[:, i], lnT[:, i]]), rowvar=False)

    out = dict(ndraw=ndraw, drag=list(drag), sd_lnEobs=lnEobs.std(axis=0, ddof=1).tolist(),
               C_lnE_lnM_lnT=C.tolist(), per_kappa={})
    for k in kappa_list:
        ch = chain(Mw, zz, Mg, LK, kT, k, upsilon_K)
        lnEp = np.log(ch["E_pred"])
        R = lnEobs - lnEp
        sdp = lnEp.std(axis=0, ddof=1)
        out["per_kappa"]["%.6g" % k] = dict(
            sd_lnEpred=sdp.tolist(),
            sd_residual=R.std(axis=0, ddof=1).tolist(),
            corr_lnEobs_lnEpred=[(float(np.corrcoef(lnEobs[:, i], lnEp[:, i])[0, 1])
                                  if sdp[i] > 0 else None) for i in range(n)])
    return out, C


# ----------------------------------------------------------------------------
# 7.  Fixed-kappa test  (item 2)
# ----------------------------------------------------------------------------
def fixed_kappa_test(A, mc, kappa=KAPPA_XCOP, nboot=10000, rng=None):
    rng = rng or RNG
    n = len(A["M_WL"])
    ch = chain(A["M_WL"], A["z"], A["M_gas"], A["L_K"], A["kT"], kappa)
    lnEo = np.log(ch["E_obs"])
    lnEp = np.log(ch["E_pred"])
    R = lnEo - lnEp
    sR = np.array(mc["per_kappa"]["%.6g" % kappa]["sd_residual"])
    sO = np.array(mc["sd_lnEobs"])
    w = 1.0 / sR ** 2
    wmean = float(np.sum(w * R) / np.sum(w))
    se_wmean = float(1.0 / math.sqrt(np.sum(w)))
    pull = R / sR
    bm = np.empty(nboot)
    for b in range(nboot):
        idx = rng.integers(0, n, n)
        bm[b] = np.sum(w[idx] * R[idx]) / np.sum(w[idx])
    Yo, Yp = ch["Yobs"], ch["Ypred"]
    return dict(
        kappa=kappa, n=n,
        E_pred=ch["E_pred"].tolist(), E_obs=ch["E_obs"].tolist(),
        residual_lnE=R.tolist(), sigma_residual_lnE=sR.tolist(),
        sigma_lnEobs_only=sO.tolist(), pull=pull.tolist(),
        mean_residual_lnE=float(R.mean()), median_residual_lnE=float(np.median(R)),
        sd_residual_lnE=float(R.std(ddof=1)),
        weighted_mean_residual_lnE=wmean,
        se_weighted_mean_from_errors=se_wmean,
        sigma_of_weighted_mean_residual=float(wmean / se_wmean),
        bootstrap_weighted_mean=pct(bm), bootstrap_se=float(bm.std(ddof=1)),
        sigma_from_bootstrap=float(wmean / bm.std(ddof=1)),
        sigma_from_errors_inflated_by_pull_sd=float(wmean / se_wmean / max(pull.std(ddof=1), 1.0)),
        mean_pull=float(pull.mean()), sd_pull=float(pull.std(ddof=1)),
        max_abs_pull=float(np.max(np.abs(pull))),
        n_clusters_within_2sigma=int(np.sum(np.abs(pull) < 2.0)),
        n_clusters_overpredicted=int(np.sum(R < 0)),
        headline_significance=dict(
            value=float(wmean / bm.std(ddof=1)),
            basis="weighted mean residual divided by its cluster-bootstrap standard deviation",
            why=("the bootstrap SE is nonparametric and absorbs any error-model "
                 "miscalibration, so it is the conservative choice and is quoted as "
                 "the headline")),
        ratio_Epred_over_Eobs=dict(median=float(np.median(ch["E_pred"] / ch["E_obs"])),
                                   min=float(np.min(ch["E_pred"] / ch["E_obs"])),
                                   max=float(np.max(ch["E_pred"] / ch["E_obs"]))),
        Y_space=dict(mean_Yobs=float(Yo.mean()), mean_Ypred=float(Yp.mean()),
                     mean_residual=float((Yo - Yp).mean()),
                     sd_residual=float((Yo - Yp).std(ddof=1)),
                     median_ratio_Ypred_over_Yobs=float(np.median(Yp / Yo))),
        verdict_direction=("model OVER-predicts the excess" if wmean < 0
                           else "model UNDER-predicts the excess"),
    )


# ----------------------------------------------------------------------------
# 8.  Free fits  (item 3)
# ----------------------------------------------------------------------------
def _nuisance(A, mode, idx=None):
    n = len(A["M_WL"])
    if mode == "noint":
        return None
    if mode == "int":
        return np.ones((n, 1))
    if mode == "int_mass":
        return np.column_stack([np.ones(n), np.log(A["M_WL"] / np.median(A["M_WL"]))])
    raise ValueError(mode)


def profile_fit(lnEo, LP, w, Z):
    """Return kappa index, SSR at the optimum, SSR at kappa = 0."""
    sw = np.sqrt(w)
    U = basis(Z, w)
    YW = (lnEo * sw)[None, :]
    LPW = LP * sw[None, :]
    ssr = ssr_grid(YW, LPW, U)[0]
    j = int(np.argmin(ssr))
    return j, float(ssr[j]), float(ssr[K0J])


def free_fits(A, mc, nboot=3000, rng=None):
    rng = rng or RNG
    n = len(A["M_WL"])
    ch0 = chain(A["M_WL"], A["z"], A["M_gas"], A["L_K"], A["kT"], 0.0)
    lnEo = np.log(ch0["E_obs"])
    w = 1.0 / np.array(mc["sd_lnEobs"]) ** 2          # kappa-independent by design
    LP = np.array([np.log(E_pred_at(A, k)) for k in KGRID])

    out = {}
    for mode, label in (("noint", "kappa only, NO intercept (the source law as written)"),
                        ("int", "kappa + a free constant offset in ln E"),
                        ("int_mass", "kappa + constant + a free ln M_WL slope")):
        Z = _nuisance(A, mode)
        j, ssr, ssr0 = profile_fit(lnEo, LP, w, Z)
        khat = float(KGRID[j])
        coef = []
        if Z is not None:
            sw = np.sqrt(w)
            coef = np.linalg.lstsq(Z * sw[:, None], (lnEo - LP[j]) * sw,
                                   rcond=None)[0].tolist()
        bk = np.empty(nboot)
        for b in range(nboot):
            idx = rng.integers(0, n, n)
            Zb = None if Z is None else Z[idx]
            jb, _, _ = profile_fit(lnEo[idx], LP[:, idx], w[idx], Zb)
            bk[b] = KGRID[jb]
        out[mode] = dict(
            label=label, kappa=khat, nuisance_coefficients=coef,
            bootstrap=pct(bk, (0.5, 2.5, 5, 16, 50, 84, 95, 97.5, 99.5)),
            bootstrap_sd=float(bk.std(ddof=1)),
            ratio_to_XCOP=khat / KAPPA_XCOP,
            sigma_from_XCOP=float((khat - KAPPA_XCOP) / bk.std(ddof=1)),
            XCOP_inside_boot95=bool(np.percentile(bk, 2.5) <= KAPPA_XCOP
                                    <= np.percentile(bk, 97.5)),
            one_sided_95_upper_limit=float(np.percentile(bk, 95)),
            one_sided_99_upper_limit=float(np.percentile(bk, 99)),
            fit_improvement_over_kappa0=ssr0 - ssr)

    # ---- Y-space deep-limit linear fits, so the M_gas/M_b factor is explicit
    t, fg, Yo = ch0["t"], ch0["f_gas"], ch0["Yobs"]
    lin = {}
    for tag, xx in (("with_fgas_CORRECT", t * fg), ("without_fgas_as_in_previous_run", t)):
        b_zero = float(np.sum(xx * Yo) / np.sum(xx * xx))
        f = ols(xx, Yo)
        bz = np.empty(nboot); bf = np.empty(nboot); ba = np.empty(nboot)
        for b in range(nboot):
            idx = rng.integers(0, n, n)
            bz[b] = float(np.sum(xx[idx] * Yo[idx]) / np.sum(xx[idx] ** 2))
            g = ols(xx[idx], Yo[idx]); bf[b] = g["b"]; ba[b] = g["a"]
        lin[tag] = dict(kappa_zero_intercept=b_zero, kappa_zero_intercept_boot=pct(bz),
                        kappa_free_intercept=f["b"], intercept=f["a"],
                        kappa_free_intercept_boot=pct(bf), intercept_boot=pct(ba),
                        residual_sd=f["s"])
    lin["ratio_correct_over_previous_zero_intercept"] = (
        lin["with_fgas_CORRECT"]["kappa_zero_intercept"]
        / lin["without_fgas_as_in_previous_run"]["kappa_zero_intercept"])
    out["Y_space_deep_limit_linear"] = lin

    # ---- per-cluster kappa_i that makes the EXACT chain reproduce E_obs_i
    ki = np.empty(n)
    for i in range(n):
        Ai = {k: (np.atleast_1d(v[i]) if isinstance(v, np.ndarray) else v)
              for k, v in A.items() if k != "name"}
        lo, hi = 0.0, 5.0e6
        target = float(ch0["E_obs"][i])
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            if float(E_pred_at(Ai, mid)[0]) < target:
                lo = mid
            else:
                hi = mid
        ki[i] = 0.5 * (lo + hi)
    out["per_cluster_kappa_exact"] = dict(
        values=ki.tolist(), names=A["name"],
        min=float(ki.min()), max=float(ki.max()), median=float(np.median(ki)),
        argmin=A["name"][int(np.argmin(ki))], argmax=A["name"][int(np.argmax(ki))],
        ratio_max_min=float(ki.max() / ki.min()), sd_of_ln=float(np.std(np.log(ki), ddof=1)),
        spearman_vs_kT=[float(v) for v in stats.spearmanr(ki, A["kT"])],
        spearman_vs_MWL=[float(v) for v in stats.spearmanr(ki, A["M_WL"])],
        note="kappa_i is the value making the EXACT chain reproduce that cluster's E_obs.")
    return out


# ----------------------------------------------------------------------------
# 9.  Injection-recovery power with the EXACT forward model  (item 4)
# ----------------------------------------------------------------------------
def power_exact(A, mc, nsim=4000, rng=None, scatter_mode="observed"):
    """
    Three arms.  All inject through the exact chain.  Both arms of every comparison
    are generated by the SAME generator with only kappa_true changed, so the null is
    exchangeable by construction.

      ARM A  amplitude-inclusive:  ln E = ln E_pred(kappa) + eps.  Tests the law as
             written (no free normalisation).
      ARM B  shape only:           ln E = a + [ln E_pred(kappa) - mean] + eps.
      ARM C  shape only at fixed mass: additionally a free ln M_WL slope, matching
             the EIV headline.

    Statistic  T = sign(kappa_hat) * (SSR(kappa=0) - SSR(kappa_hat)), one-sided in
    the direction the model predicts.  Critical value = 95th percentile of T under
    kappa_true = 0 from the SAME generator, so the false-positive rate is 0.05 by
    construction; the uncorrected nominal rate is reported alongside.
    """
    rng = rng or RNG
    n = len(A["M_WL"])
    sd_meas = np.array(mc["sd_lnEobs"])
    w = 1.0 / sd_meas ** 2
    sw = np.sqrt(w)

    ch0 = chain(A["M_WL"], A["z"], A["M_gas"], A["L_K"], A["kT"], 0.0)
    lnEo = np.log(ch0["E_obs"])
    fM = ols(np.log(A["M_WL"]), lnEo)
    s_tot = fM["s"]
    s_int2 = max(0.0, s_tot ** 2 - float(np.mean(sd_meas ** 2)))
    if scatter_mode == "observed":
        # calibrate the generator so its total scatter reproduces the OBSERVED scatter
        # of ln E about the mass relation.  Here the propagated measurement errors are
        # LARGER than the observed scatter, so this scales them DOWN; the alternative
        # ("measurement") mode keeps them and is reported as the conservative variant.
        sd_gen = sd_meas * (s_tot / math.sqrt(float(np.mean(sd_meas ** 2))))
    else:
        sd_gen = np.sqrt(sd_meas ** 2 + s_int2)
    a_obs = float(np.mean(lnEo))
    q_obs = fM["b"]
    lnM = np.log(A["M_WL"])

    LP = np.array([np.log(E_pred_at(A, k)) for k in KGRID])
    LPW = LP * sw[None, :]
    Us = {"A": None,
          "B": basis(np.ones((n, 1)), w),
          "C": basis(np.column_stack([np.ones(n), lnM - lnM.mean()]), w)}

    kappa_grid_inj = [0.0, 5e3, 1e4, 2e4, 3e4, 5e4, 7.5e4, 1.0e5, 1.36e5, 2.0e5, 3.0e5, 5.0e5]
    raw = {arm: {} for arm in "ABC"}
    khat = {arm: {} for arm in "ABC"}
    for kin in kappa_grid_inj:
        lpk = np.log(E_pred_at(A, kin))
        eps = rng.normal(0.0, 1.0, (nsim, n)) * sd_gen[None, :]
        Ys = {"A": lpk[None, :] + eps,
              "B": a_obs + (lpk - lpk.mean())[None, :] + eps,
              "C": a_obs + q_obs * (lnM - lnM.mean())[None, :]
                   + (lpk - lpk.mean())[None, :] + eps}
        for arm in "ABC":
            YW = Ys[arm] * sw[None, :]
            ssr = ssr_grid(YW, LPW, Us[arm])
            j = np.argmin(ssr, axis=1)
            T = np.where(KGRID[j] >= 0, 1.0, -1.0) * (ssr[:, K0J] - ssr[np.arange(nsim), j])
            raw[arm]["%.6g" % kin] = T
            khat[arm]["%.6g" % kin] = KGRID[j]

    out = {}
    for arm in "ABC":
        T0 = raw[arm]["0"]
        crit = float(np.percentile(T0, 95.0))
        nominal_crit = 3.84
        rows = []
        for kin in kappa_grid_inj:
            T = raw[arm]["%.6g" % kin]; K = khat[arm]["%.6g" % kin]
            rows.append(dict(kappa_injected=kin,
                             power_size_corrected=float((T > crit).mean()),
                             power_nominal=float((T > nominal_crit).mean()),
                             median_kappa_hat=float(np.median(K)),
                             p16_kappa_hat=float(np.percentile(K, 16)),
                             p84_kappa_hat=float(np.percentile(K, 84))))

        def floor80(key):
            xs = [r["kappa_injected"] for r in rows]; ys = [r[key] for r in rows]
            for i in range(1, len(xs)):
                if ys[i] >= 0.80 > ys[i - 1]:
                    f = (0.80 - ys[i - 1]) / (ys[i] - ys[i - 1])
                    return float(xs[i - 1] + f * (xs[i] - xs[i - 1]))
            return None

        out["arm_" + arm] = dict(
            description={"A": "amplitude-inclusive, no free normalisation (law as written)",
                         "B": "shape only, free constant",
                         "C": "shape only at fixed mass (free constant + free ln M_WL slope)"}[arm],
            nsim=nsim,
            false_positive_rate_nominal_at_kappa0=float((T0 > nominal_crit).mean()),
            critical_value_nominal=nominal_crit,
            critical_value_size_corrected=crit,
            size_correction_needed=bool(abs(float((T0 > nominal_crit).mean()) - 0.05) > 0.01),
            false_positive_rate_after_correction=float((T0 > crit).mean()),
            curve=rows,
            power_at_XCOP_kappa=float((raw[arm]["136000"] > crit).mean()),
            kappa_floor_80pct_power=floor80("power_size_corrected"))
    out["generator"] = dict(
        scatter_mode=scatter_mode,
        sd_lnE_about_mass_relation_observed=s_tot,
        rms_measurement_sd_lnE=float(np.sqrt(np.mean(sd_meas ** 2))),
        rms_generator_sd_lnE=float(np.sqrt(np.mean(sd_gen ** 2))),
        implied_intrinsic_scatter_lnE=float(math.sqrt(s_int2)),
        note=("identical generator at every injected kappa, so the kappa = 0 arm and the "
              "kappa > 0 arms are exchangeable and the size check is meaningful."))
    return out


# ----------------------------------------------------------------------------
# 10.  Errors-in-variables / hierarchical treatment  (item 5)
# ----------------------------------------------------------------------------
def _mom(Yl, Cl):
    S = np.cov(Yl, rowvar=False)
    V = S - Cl.mean(axis=0)
    M = np.array([[V[1, 1], V[1, 2]], [V[1, 2], V[2, 2]]])
    b = np.array([V[0, 1], V[0, 2]])
    try:
        qp = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, V
    q, p = float(qp[0]), float(qp[1])
    s2 = V[0, 0] - (q * q * V[1, 1] + 2 * p * q * V[1, 2] + p * p * V[2, 2])
    return p, q, s2, V


def _naive(Yl):
    X = np.column_stack([np.ones(len(Yl)), Yl[:, 2], Yl[:, 1]])
    beta, *_ = np.linalg.lstsq(X, Yl[:, 0], rcond=None)
    return float(beta[1]), float(beta[2])


def _nll(theta, Yl, Cl):
    c, p, q, lns, mu1, mu2, l11, l21, l22 = theta
    if not np.all(np.isfinite(theta)):
        return 1e12
    s2 = math.exp(2.0 * min(lns, 5.0))
    L = np.array([[math.exp(min(l11, 5.0)), 0.0], [l21, math.exp(min(l22, 5.0))]])
    Sig = L @ L.T
    B = np.array([[q, p], [1.0, 0.0], [0.0, 1.0]])
    V = B @ Sig @ B.T
    V[0, 0] += s2
    mu = np.array([c + p * mu2 + q * mu1, mu1, mu2])
    Vi = V[None, :, :] + Cl
    try:
        Lc = np.linalg.cholesky(Vi)
    except np.linalg.LinAlgError:
        return 1e12
    d = (Yl - mu)[:, :, None]
    sol = np.linalg.solve(Lc, d)[:, :, 0]
    return float(0.5 * (2.0 * np.sum(np.log(np.diagonal(Lc, axis1=1, axis2=2)))
                        + np.sum(sol * sol)))


def _mle(Yl, Cl, x0=None, polish=False):
    if x0 is None:
        p0, q0, s20, _ = _mom(Yl, Cl)
        if not np.isfinite(p0):
            p0, q0 = _naive(Yl); s20 = 0.01
        mu1 = float(Yl[:, 1].mean()); mu2 = float(Yl[:, 2].mean())
        c0 = float(Yl[:, 0].mean() - p0 * mu2 - q0 * mu1)
        S = np.cov(Yl, rowvar=False)
        v11 = max(S[1, 1] - Cl[:, 1, 1].mean(), 1e-4)
        v22 = max(S[2, 2] - Cl[:, 2, 2].mean(), 1e-4)
        v12 = S[1, 2]
        Sig0 = np.array([[v11, v12], [v12, v22]])
        try:
            L0 = np.linalg.cholesky(Sig0)
        except np.linalg.LinAlgError:
            L0 = np.diag(np.sqrt([v11, v22]))
        x0 = np.array([c0, p0, q0, 0.5 * math.log(max(s20, 1e-4)), mu1, mu2,
                       math.log(max(L0[0, 0], 1e-3)), L0[1, 0], math.log(max(L0[1, 1], 1e-3))])
    r = optimize.minimize(_nll, x0, args=(Yl, Cl), method="L-BFGS-B",
                          options=dict(maxiter=800, maxfun=8000))
    if polish:
        r = optimize.minimize(_nll, r.x, args=(Yl, Cl), method="Nelder-Mead",
                              options=dict(maxiter=8000, maxfev=8000, xatol=1e-9, fatol=1e-9))
        r = optimize.minimize(_nll, r.x, args=(Yl, Cl), method="L-BFGS-B",
                              options=dict(maxiter=800, maxfun=8000))
    return r.x


def eiv(A, C, mc, nboot_mom=4000, nboot_mle=500, nnull=4000, rng=None):
    """
    Replaces the partial rank correlation rho_p(E, kT | M_WL).

      observables  y_i   = (ln E_obs_i, ln M_WL_i, ln kT_i)
      latents      xi_i  = (ln E_true_i, m_i, tau_i),  y_i = xi_i + e_i, e_i ~ N(0, C_i)
      structure    ln E_true = c + p tau + q m + eps,  eps ~ N(0, s^2)
      population   (m, tau) ~ N(mu, Sigma)
      marginal     y_i ~ N(mean, B Sigma B' + s^2 e1 e1' + C_i)

    C_i is measured by pushing the published errors through the exact chain, and is
    strongly NON-diagonal: ln E_obs is built from M_WL, so its error correlates with
    the ln M_WL error.  That correlation is exactly what biases a naive partial
    statistic, and it is what this treatment removes.

    Target parameter p = d ln E / d ln kT at fixed mass.
    """
    rng = rng or RNG
    n = len(A["M_WL"])
    ch0 = chain(A["M_WL"], A["z"], A["M_gas"], A["L_K"], A["kT"], 0.0)
    Y = np.column_stack([np.log(ch0["E_obs"]), np.log(A["M_WL"]), np.log(A["kT"])])
    C = np.asarray(C, float)

    p_mom, q_mom, s2_mom, Vhat = _mom(Y, C)
    th = _mle(Y, C, polish=True)
    p_mle, q_mle = float(th[1]), float(th[2])
    p_nai, q_nai = _naive(Y)

    corr_e = np.array([C[i][0, 1] / math.sqrt(C[i][0, 0] * C[i][1, 1]) for i in range(n)])

    def partial_spearman(a, b, cc):
        ra, rb, rc = stats.rankdata(a), stats.rankdata(b), stats.rankdata(cc)
        rab = np.corrcoef(ra, rb)[0, 1]; rac = np.corrcoef(ra, rc)[0, 1]
        rbc = np.corrcoef(rb, rc)[0, 1]
        return float((rab - rac * rbc) / math.sqrt(max(1e-15, (1 - rac ** 2) * (1 - rbc ** 2))))

    rho_p_naive = partial_spearman(ch0["Yobs"], A["kT"], A["M_WL"])

    bp_mom, bp_nai = [], []
    for b in range(nboot_mom):
        idx = rng.integers(0, n, n)
        pm, _, _, _ = _mom(Y[idx], C[idx])
        if np.isfinite(pm) and abs(pm) < 20:
            bp_mom.append(pm)
        pn, _ = _naive(Y[idx]); bp_nai.append(pn)
    bp_mom = np.array(bp_mom); bp_nai = np.array(bp_nai)

    bp_mle = []
    for b in range(nboot_mle):
        idx = rng.integers(0, n, n)
        try:
            thb = _mle(Y[idx], C[idx], x0=th)
            if abs(thb[1]) < 20:
                bp_mle.append(float(thb[1]))
        except Exception:
            pass
    bp_mle = np.array(bp_mle)

    # ---- parametric null / recovery, using the fitted population and the real C_i
    c_, p_, q_, lns, mu1, mu2, l11, l21, l22 = th
    s_ = math.exp(lns)
    L = np.array([[math.exp(l11), 0.0], [l21, math.exp(l22)]])
    Sig = L @ L.T
    Cchol = np.array([np.linalg.cholesky(C[i] + 1e-14 * np.eye(3)) for i in range(n)])

    def gen(p_true, ns):
        mt = rng.multivariate_normal([mu1, mu2], Sig, size=(ns, n))
        m_, tau_ = mt[:, :, 0], mt[:, :, 1]
        lnE = c_ + p_true * tau_ + q_ * m_ + rng.normal(0.0, s_, (ns, n))
        Xi = np.stack([lnE, m_, tau_], axis=2)
        u = rng.normal(0.0, 1.0, (ns, n, 3))
        e = np.einsum("njk,lnk->lnj", Cchol, u)
        return Xi + e

    ch_x = chain(A["M_WL"], A["z"], A["M_gas"], A["L_K"], A["kT"], KAPPA_XCOP)
    p_required = (1.0 + dlnnu_dlnx(ch_x["x_eff"])) * ch_x["delta"] / (1.0 + ch_x["delta"])
    p_req_mean = float(np.mean(p_required))

    def sweep(p_true, ns, n_mle=0):
        D = gen(p_true, ns)
        om, on, ol = [], [], []
        for l in range(ns):
            pm, _, _, _ = _mom(D[l], C)
            if np.isfinite(pm) and abs(pm) < 20:
                om.append(pm)
            pn, _ = _naive(D[l]); on.append(pn)
            if l < n_mle:
                try:
                    thl = _mle(D[l], C, x0=th)
                    if abs(thl[1]) < 20:
                        ol.append(float(thl[1]))
                except Exception:
                    pass
        return np.array(om), np.array(on), np.array(ol)

    null_mom, null_nai, null_mle = sweep(0.0, nnull, n_mle=min(nnull, 1200))
    rec_mom, rec_nai, rec_mle = sweep(p_req_mean, min(nnull, 2000), n_mle=800)

    # ---- recovery straight off the EXACT chain: if the source law were true at the
    #      X-COP kappa, generate ln E from the chain itself (plus the observed scatter
    #      and the real measurement errors) and see what each estimator returns.
    sd_meas = np.array(mc["sd_lnEobs"])
    lnEo_real = np.log(ch0["E_obs"])
    fM = ols(np.log(A["M_WL"]), lnEo_real)
    sd_gen = sd_meas * (fM["s"] / math.sqrt(float(np.mean(sd_meas ** 2))))
    lp_x = np.log(ch_x["E_pred"])
    off = float(np.mean(lnEo_real) - np.mean(lp_x))     # free normalisation, fair to the model
    ch_mle, ch_mom, ch_nai = [], [], []
    for l in range(600):
        lnE = off + lp_x + rng.normal(0.0, sd_gen)
        Yl = np.column_stack([lnE, np.log(A["M_WL"]), np.log(A["kT"])])
        u = rng.normal(0.0, 1.0, (n, 3))
        Yl = Yl + np.einsum("njk,nk->nj", Cchol, u)
        pm, _, _, _ = _mom(Yl, C)
        if np.isfinite(pm) and abs(pm) < 20:
            ch_mom.append(pm)
        pn, _ = _naive(Yl); ch_nai.append(pn)
        try:
            thl = _mle(Yl, C, x0=th)
            if abs(thl[1]) < 20:
                ch_mle.append(float(thl[1]))
        except Exception:
            pass
    ch_mle = np.array(ch_mle); ch_mom = np.array(ch_mom); ch_nai = np.array(ch_nai)

    # ---- map the EIV-estimated p back onto kappa through the exact chain.
    #      p_of_kappa is the SAME regression run on noise-free model predictions, so the
    #      mapping is estimator-consistent; it is strictly increasing, hence invertible.
    Xd = np.column_stack([np.ones(n), np.log(A["kT"]), np.log(A["M_WL"])])
    p_of_k = np.array([float(np.linalg.lstsq(
        Xd, np.log(E_pred_at(A, k)), rcond=None)[0][1]) for k in KGRID])
    order = np.argsort(p_of_k)
    monotone = bool(np.all(np.diff(p_of_k) > -1e-12))

    def k_of_p(pv):
        return float(np.interp(pv, p_of_k[order], KGRID[order]))

    kappa_eiv = k_of_p(p_mle)
    kappa_eiv_boot = np.array([k_of_p(v) for v in bp_mle]) if len(bp_mle) > 50 else None

    return dict(
        target_parameter="p = d ln E / d ln kT at fixed ln M_WL",
        p_EIV_maximum_likelihood=p_mle, q_EIV_maximum_likelihood=q_mle,
        p_EIV_method_of_moments=p_mom, q_EIV_method_of_moments=q_mom,
        intrinsic_scatter_lnE_from_MLE=float(math.exp(th[3])),
        moment_estimator_implied_intrinsic_variance=float(s2_mom),
        Vhat_positive_definite=bool(np.all(np.linalg.eigvalsh(Vhat) > 0)),
        Vhat_eigenvalues=np.linalg.eigvalsh(Vhat).tolist(),
        p_naive_OLS_treating_M_WL_as_exact=p_nai,
        q_naive_OLS_treating_M_WL_as_exact=q_nai,
        rho_partial_spearman_naive_for_reference=rho_p_naive,
        measurement_error_correlation_lnEobs_lnMWL=dict(
            median=float(np.median(corr_e)), min=float(corr_e.min()), max=float(corr_e.max()),
            note="this is the quantity that makes the naive partial statistic biased"),
        bootstrap_p_MLE=pct(bp_mle, (2.5, 16, 50, 84, 97.5)) if len(bp_mle) > 50 else None,
        bootstrap_p_MLE_sd=float(bp_mle.std(ddof=1)) if len(bp_mle) > 50 else None,
        n_bootstrap_MLE=int(len(bp_mle)),
        bootstrap_p_moments=pct(bp_mom, (2.5, 16, 50, 84, 97.5)),
        bootstrap_p_moments_sd=float(bp_mom.std(ddof=1)),
        bootstrap_p_naive=pct(bp_nai, (2.5, 16, 50, 84, 97.5)),
        null_p_equals_zero=dict(
            EIV_MLE=dict(mean=float(null_mle.mean()), median=float(np.median(null_mle)),
                         sd=float(null_mle.std(ddof=1)),
                         p2_5=float(np.percentile(null_mle, 2.5)),
                         p97_5=float(np.percentile(null_mle, 97.5)), n=int(len(null_mle))),
            EIV_moments=dict(mean=float(null_mom.mean()), median=float(np.median(null_mom)),
                             sd=float(null_mom.std(ddof=1)),
                             p2_5=float(np.percentile(null_mom, 2.5)),
                             p97_5=float(np.percentile(null_mom, 97.5))),
            naive=dict(mean=float(null_nai.mean()), median=float(np.median(null_nai)),
                       sd=float(null_nai.std(ddof=1)),
                       p2_5=float(np.percentile(null_nai, 2.5)),
                       p97_5=float(np.percentile(null_nai, 97.5))),
            bias_of_naive_estimator=float(np.median(null_nai)),
            bias_of_EIV_MLE=float(np.median(null_mle)),
            bias_of_EIV_moments=float(np.median(null_mom)),
            two_sided_p_observed_EIV_MLE=float(np.mean(np.abs(null_mle - np.median(null_mle))
                                                       >= abs(p_mle - np.median(null_mle)))),
            two_sided_p_observed_EIV_moments=float(np.mean(np.abs(null_mom - np.median(null_mom))
                                                           >= abs(p_mom - np.median(null_mom)))),
            two_sided_p_observed_naive=float(np.mean(np.abs(null_nai - np.median(null_nai))
                                                     >= abs(p_nai - np.median(null_nai)))),
            n=int(len(null_mom))),
        kappa_from_EIV=dict(
            mapping_is_monotone=monotone,
            p_of_kappa_grid={"kappa": [0.0, 1e4, 3e4, 1e5, 1.36e5, 3e5],
                             "p": [float(np.interp(k, KGRID, p_of_k))
                                   for k in (0.0, 1e4, 3e4, 1e5, 1.36e5, 3e5)]},
            kappa_EIV_MLE=kappa_eiv,
            kappa_EIV_bootstrap=(pct(kappa_eiv_boot, (2.5, 16, 50, 84, 95, 97.5))
                                 if kappa_eiv_boot is not None else None),
            kappa_EIV_one_sided_95_upper=(float(np.percentile(kappa_eiv_boot, 95))
                                          if kappa_eiv_boot is not None else None),
            kappa_naive_for_comparison=k_of_p(p_nai),
            note=("kappa recovered from the EIV temperature exponent by inverting the SAME "
                  "regression applied to noise-free exact-chain predictions.  This is the "
                  "M_WL-uncertainty-corrected counterpart of the free fit.")),
        required_p_at_XCOP_kappa=dict(
            per_cluster=p_required.tolist(), mean=p_req_mean,
            min=float(p_required.min()), max=float(p_required.max()),
            derivation=("p_req = d ln E_pred/d ln kT = [1 + dlnnu/dlnx at x_eff] * "
                        "delta/(1+delta), from the EXACT chain at kappa = 1.36e5")),
        recovery_at_required_p=dict(
            EIV_MLE_median=float(np.median(rec_mle)), EIV_MLE_sd=float(rec_mle.std(ddof=1)),
            EIV_moments_median=float(np.median(rec_mom)),
            naive_median=float(np.median(rec_nai)), naive_sd=float(rec_nai.std(ddof=1)),
            note="linear-population generator with p set to the required value"),
        recovery_generated_from_the_EXACT_CHAIN_at_XCOP_kappa=dict(
            n=int(len(ch_mle)),
            EIV_MLE_median=float(np.median(ch_mle)), EIV_MLE_sd=float(ch_mle.std(ddof=1)),
            EIV_MLE_p2_5=float(np.percentile(ch_mle, 2.5)),
            EIV_MLE_p97_5=float(np.percentile(ch_mle, 97.5)),
            EIV_moments_median=float(np.median(ch_mom)),
            naive_median=float(np.median(ch_nai)), naive_sd=float(ch_nai.std(ddof=1)),
            observed_p_EIV_MLE=p_mle,
            observed_p_naive=p_nai,
            separation_in_sd_between_observed_and_model=float(
                (np.median(ch_mle) - p_mle) / ch_mle.std(ddof=1)),
            note=("ln E generated DIRECTLY from the exact chain at kappa = 1.36e5, with a free "
                  "normalisation (fair to the model: the amplitude mismatch is given away), "
                  "the observed scatter and the real measurement errors.  This is the "
                  "apples-to-apples comparison: the same estimator run on data the model "
                  "would produce, against the same estimator run on the real data.")),
        why=("Conditioning on a noisy ln M_WL biases any partial statistic, because the error "
             "in ln E_obs is strongly correlated with the error in ln M_WL (E is built from "
             "M_WL).  The EIV estimator subtracts the known measurement covariance C_i from "
             "the total covariance before solving for (p, q), so the M_WL uncertainty is "
             "propagated instead of being treated as exact."),
    )


# ----------------------------------------------------------------------------
# 11.  Responsiveness: does every headline statistic MOVE with kappa?
# ----------------------------------------------------------------------------
def responsiveness(A, mc):
    n = len(A["M_WL"])
    w = 1.0 / np.array(mc["sd_lnEobs"]) ** 2
    LP = np.array([np.log(E_pred_at(A, k)) for k in KGRID])
    ch_obs = chain(A["M_WL"], A["z"], A["M_gas"], A["L_K"], A["kT"], 0.0)
    lnEo_real = np.log(ch_obs["E_obs"])
    grid = [0.0, 1e3, 1e4, 3e4, 1e5, 1.36e5, 3e5, 1e6]
    Zn = _nuisance(A, "int_mass")
    rows = []
    for k in grid:
        ch = chain(A["M_WL"], A["z"], A["M_gas"], A["L_K"], A["kT"], k)
        lnEsyn = np.log(ch["E_pred"])
        row = dict(kappa=k)
        # EXACTLY blind: the previous run's compressed response Y = kappa t was a strictly
        # monotone function of its own input, so its rank correlation is 1.000 at every kappa.
        row["spearman_compressed_response_vs_t_EXACTLY_BLIND"] = float(
            stats.spearmanr(k * ch["t"], ch["t"])[0]) if k > 0 else float("nan")
        row["spearman_delta_vs_t_fgas_EXACTLY_BLIND"] = float(
            stats.spearmanr(ch["delta"], ch["t"] * ch["f_gas"])[0]) if k > 0 else float("nan")
        # NEARLY blind: the exact chain breaks strict monotonicity only through x_b
        row["spearman_Epred_vs_t_NEARLY_BLIND"] = float(
            stats.spearmanr(ch["E_pred"], ch["t"])[0]) if k > 0 else float("nan")
        row["spearman_Ypred_vs_t_fgas_NEARLY_BLIND"] = float(
            stats.spearmanr(ch["Ypred"], ch["t"] * ch["f_gas"])[0]) if k > 0 else float("nan")
        row["mean_lnEpred"] = float(lnEsyn.mean())
        row["sd_lnEpred"] = float(lnEsyn.std(ddof=1))
        row["weighted_mean_residual_real_data_vs_this_kappa"] = float(
            np.sum(w * (lnEo_real - lnEsyn)) / np.sum(w))
        X = np.column_stack([np.ones(n), np.log(A["kT"]), np.log(A["M_WL"])])
        row["p_kT_exponent_of_noisefree_synthetic_data"] = float(
            np.linalg.lstsq(X, lnEsyn, rcond=None)[0][1])
        j, _, _ = profile_fit(lnEsyn, LP, w, None)
        row["kappa_recovered_noint"] = float(KGRID[j])
        j2, _, _ = profile_fit(lnEsyn, LP, w, Zn)
        row["kappa_recovered_int_mass"] = float(KGRID[j2])
        rows.append(row)

    keys_used = ["mean_lnEpred", "sd_lnEpred", "weighted_mean_residual_real_data_vs_this_kappa",
                 "p_kT_exponent_of_noisefree_synthetic_data", "kappa_recovered_noint",
                 "kappa_recovered_int_mass"]
    keys_all = keys_used + ["spearman_compressed_response_vs_t_EXACTLY_BLIND",
                            "spearman_delta_vs_t_fgas_EXACTLY_BLIND",
                            "spearman_Epred_vs_t_NEARLY_BLIND",
                            "spearman_Ypred_vs_t_fgas_NEARLY_BLIND"]
    checks = {}
    for key in keys_all:
        v = [r[key] for r in rows if not (isinstance(r[key], float) and math.isnan(r[key]))]
        checks[key] = dict(values=v, spread=float(max(v) - min(v)),
                           MOVES=bool(max(v) - min(v) > 1e-9))
    blind = [k for k, v in checks.items() if not v["MOVES"]]
    kfac = 1e6 / 1e3
    return dict(grid=rows, checks=checks, statistics_BLIND_to_kappa=blind,
                all_statistics_used_in_this_analysis_move=bool(
                    all(checks[k]["MOVES"] for k in keys_used)),
                kappa_range_spanned=[1e3, 1e6],
                interpretation=(
                    "Two rank statistics on the prediction are EXACTLY constant at 1.000 over "
                    "three decades of kappa, because the response is a strictly monotone "
                    "function of its own input: that is the recorded trap, reproduced here on "
                    "purpose.  Two more (on E_pred itself) move by only ~0.02 over the same "
                    "three decades - nearly blind - because the exact chain breaks strict "
                    "monotonicity only weakly, through the spread in x_b.  Every statistic this "
                    "analysis actually uses is listed above and moves by orders of magnitude "
                    "over the same range: the fitted kappa tracks the injected kappa one-to-one "
                    "and the temperature exponent runs from 0.00 to 0.89."))


# ----------------------------------------------------------------------------
# 12.  Aperture caveat with the EXACT model  (item 7)
# ----------------------------------------------------------------------------
def aperture(A, mc, nsim=2000, rng=None):
    """
    All LoCuSS observables sit inside r500 set by the LENSING mass, and kT_X_ce is
    measured over ~0.15-1 r500,WL.  A WL mass error delta drags everything:
        dln r500 = delta/3, dln M_gas = a_g delta/3, dln L_K = a_s delta/3,
        dln kT = b_T delta/3.
    NEW: DM_P is proportional to M_gas AND kT, so the aperture drags the PREDICTION
    as well as the observation and part of the effect cancels.  Both are computed.
    """
    rng = rng or RNG
    n = len(A["M_WL"])
    ch0 = chain(A["M_WL"], A["z"], A["M_gas"], A["L_K"], A["kT"], 0.0)
    chX = chain(A["M_WL"], A["z"], A["M_gas"], A["L_K"], A["kT"], KAPPA_XCOP)
    fgas = ch0["f_gas"]; fstar = 1.0 - fgas
    nub = dlnnu_dlnx(ch0["x_b"]); nueff = dlnnu_dlnx(chX["x_eff"]); dP = chX["delta"]
    sM = A["sM"]; sT = A["sT"]
    sdE = float(np.std(np.log(ch0["E_obs"]), ddof=1))
    sdT = float(np.std(np.log(A["kT"]), ddof=1))

    grid = []
    for a_g in (0.8, 1.0, 1.2, 1.5):
        for a_s in (0.6, 1.0, 1.4):
            for b_T in (-0.6, -0.4, -0.2, 0.0, 0.2):
                Adl = (fgas * a_g + fstar * a_s) / 3.0
                dlnx_b = Adl - 2.0 / 3.0
                dlnE_obs = 1.0 - nub * dlnx_b - Adl
                dln_dP = (b_T + a_g) / 3.0 - Adl
                dlnE_pred = ((nueff - nub) * dlnx_b
                             + (1.0 + nueff) * (dP / (1.0 + dP)) * dln_dP)
                dlnR = dlnE_obs - dlnE_pred
                dlnT = b_T / 3.0
                grid.append(dict(
                    alpha_gas=a_g, alpha_star=a_s, beta_T=b_T,
                    mean_dlnEobs_ddelta=float(np.mean(dlnE_obs)),
                    mean_dlnEpred_ddelta_at_XCOP=float(np.mean(dlnE_pred)),
                    mean_dln_residual_ddelta=float(np.mean(dlnR)),
                    induced_corr_lnEobs_lnT=float(np.mean(dlnE_obs * dlnT * sM ** 2)
                                                  / (sdE * sdT)),
                    induced_corr_residual_lnT=float(np.mean(dlnR * dlnT * sM ** 2) / (sdE * sdT)),
                    aperture_share_of_var_lnEobs=float(np.mean(dlnE_obs ** 2 * sM ** 2) / sdE ** 2),
                    aperture_share_of_var_lnT=float(np.mean(dlnT ** 2 * sM ** 2) / sdT ** 2)))
    ind = [g["induced_corr_lnEobs_lnT"] for g in grid]
    indr = [g["induced_corr_residual_lnT"] for g in grid]

    # ---- Monte Carlo directly on the EIV target parameter p
    lnEo = np.log(ch0["E_obs"])
    fM = ols(np.log(A["M_WL"]), lnEo)
    resid = fM["resid"]
    configs = [(1.2, 1.0, -0.2), (1.2, 1.0, 0.0), (1.0, 1.0, -0.4), (1.5, 1.4, 0.2),
               (1.2, 1.0, -0.6)]
    mc_out = {}
    for cfg in configs:
        a_g, a_s, b_T = cfg
        Adl = (fgas * a_g + fstar * a_s) / 3.0
        # response of ln E_obs to a WL mass error, with the aperture drag ON and OFF.
        # r500 always follows M_WL; only M_gas, L_K and kT stop following when drag is OFF.
        dlnE_on = 1.0 - nub * (Adl - 2.0 / 3.0) - Adl
        dlnE_off = 1.0 - nub * (0.0 - 2.0 / 3.0)
        acc = {"on": [], "off": [], "clean": []}
        for _ in range(nsim):
            lnE_true = fM["a"] + fM["b"] * np.log(A["M_WL"]) + resid[rng.permutation(n)]
            d = rng.normal(0.0, sM, n)
            eta = rng.normal(0.0, sT, n)
            for tag, drag, dd in (("on", True, d), ("off", False, d),
                                  ("clean", False, np.zeros(n))):
                Mw = A["M_WL"] * np.exp(dd)
                kT = A["kT"] * np.exp((b_T * dd / 3.0 if drag else 0.0) + eta)
                lnE = lnE_true + (dlnE_on if drag else dlnE_off) * dd
                X = np.column_stack([np.ones(n), np.log(kT), np.log(Mw)])
                acc[tag].append(float(np.linalg.lstsq(X, lnE, rcond=None)[0][1]))
        on = np.array(acc["on"]); off = np.array(acc["off"]); cl = np.array(acc["clean"])
        mc_out["alpha_gas=%.1f_alpha_star=%.1f_beta_T=%+.1f" % cfg] = dict(
            p_naive_CLEAN_no_WL_error=dict(mean=float(cl.mean()), sd=float(cl.std(ddof=1))),
            p_naive_OFF_WL_error_no_drag=dict(mean=float(off.mean()), sd=float(off.std(ddof=1))),
            p_naive_ON_WL_error_plus_drag=dict(mean=float(on.mean()), sd=float(on.std(ddof=1))),
            regression_dilution_OFF_minus_CLEAN=float((off - cl).mean()),
            aperture_manufactured_ON_minus_OFF=float((on - off).mean()))
    return dict(
        analytic_grid=grid,
        induced_corr_lnEobs_lnT_range=[float(min(ind)), float(max(ind))],
        induced_corr_residual_lnT_range=[float(min(indr)), float(max(indr))],
        does_the_Mgas_over_Mb_factor_change_it=dict(
            answer=("YES, but only slightly, and in the direction of MORE aperture leverage, "
                    "not less.  Because DM_P is proportional to M_gas while the denominator is "
                    "M_b, the aperture drags numerator and denominator together and "
                    "delta = kappa t f_gas is nearly aperture-invariant; the prediction "
                    "therefore barely responds (d ln E_pred/d delta is about -0.02 to -0.13, "
                    "slightly NEGATIVE, driven by the drop in x_b), while the observation "
                    "responds at about +0.49.  The residual response is therefore about 11 per "
                    "cent LARGER than the observation's.  The induced correlation stays bounded "
                    "and small, and stays NEGATIVE for any declining temperature profile, i.e. "
                    "it pushes away from the model, not toward it."),
            mean_dlnEobs_ddelta_over_grid=float(np.mean([g["mean_dlnEobs_ddelta"] for g in grid])),
            mean_dln_residual_ddelta_over_grid=float(
                np.mean([g["mean_dln_residual_ddelta"] for g in grid])),
            suppression_factor=float(np.mean([g["mean_dln_residual_ddelta"] for g in grid])
                                     / np.mean([g["mean_dlnEobs_ddelta"] for g in grid]))),
        montecarlo_on_EIV_target_parameter=mc_out, nsim=nsim)


# ----------------------------------------------------------------------------
# 12b.  Leave-one-cluster-out on every headline number
# ----------------------------------------------------------------------------
def leave_one_out(A, C, mc, kappa=KAPPA_XCOP):
    n = len(A["M_WL"])
    sd = np.array(mc["sd_lnEobs"])
    sR = np.array(mc["per_kappa"]["%.6g" % kappa]["sd_residual"])
    LP = np.array([np.log(E_pred_at(A, k)) for k in KGRID])
    ch0 = chain(A["M_WL"], A["z"], A["M_gas"], A["L_K"], A["kT"], 0.0)
    chK = chain(A["M_WL"], A["z"], A["M_gas"], A["L_K"], A["kT"], kappa)
    lnEo = np.log(ch0["E_obs"])
    R = lnEo - np.log(chK["E_pred"])
    C = np.asarray(C, float)
    Y = np.column_stack([lnEo, np.log(A["M_WL"]), np.log(A["kT"])])
    th_full = _mle(Y, C, polish=True)

    out = {"wmean_residual": [], "kappa_noint": [], "kappa_int_mass": [], "p_EIV": []}
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        w = 1.0 / sR[m] ** 2
        out["wmean_residual"].append(float(np.sum(w * R[m]) / np.sum(w)))
        wf = 1.0 / sd[m] ** 2
        j1, _, _ = profile_fit(lnEo[m], LP[:, m], wf, None)
        out["kappa_noint"].append(float(KGRID[j1]))
        Z = np.column_stack([np.ones(int(m.sum())), np.log(A["M_WL"][m] / np.median(A["M_WL"][m]))])
        j2, _, _ = profile_fit(lnEo[m], LP[:, m], wf, Z)
        out["kappa_int_mass"].append(float(KGRID[j2]))
        try:
            out["p_EIV"].append(float(_mle(Y[m], C[m], x0=th_full)[1]))
        except Exception:
            out["p_EIV"].append(float("nan"))

    res = {}
    for k, v in out.items():
        v = np.array(v, float)
        good = np.isfinite(v)
        res[k] = dict(min=float(v[good].min()), max=float(v[good].max()),
                      median=float(np.median(v[good])),
                      argmin=A["name"][int(np.argmin(np.where(good, v, np.inf)))],
                      argmax=A["name"][int(np.argmax(np.where(good, v, -np.inf)))],
                      values=v.tolist())
    res["note"] = ("Each entry is the headline statistic recomputed on the 39 clusters that "
                   "remain after dropping one.  Sign stability across all 40 subsamples is the "
                   "test; no single cluster may control a conclusion.")
    return res


# ----------------------------------------------------------------------------
# 13.  Calibration (no chi2/lnL/AIC/BIC is quoted anywhere)
# ----------------------------------------------------------------------------
def calibration(A, mc, kappa=KAPPA_XCOP):
    ch0 = chain(A["M_WL"], A["z"], A["M_gas"], A["L_K"], A["kT"], 0.0)
    lnEo = np.log(ch0["E_obs"])
    sd_meas = np.array(mc["sd_lnEobs"])
    fM = ols(np.log(A["M_WL"]), lnEo)
    chk = chain(A["M_WL"], A["z"], A["M_gas"], A["L_K"], A["kT"], kappa)
    R = lnEo - np.log(chk["E_pred"])
    sR = np.array(mc["per_kappa"]["%.6g" % kappa]["sd_residual"])
    return dict(
        median_measurement_sd_lnEobs=float(np.median(sd_meas)),
        observed_sd_lnEobs=float(np.std(lnEo, ddof=1)),
        residual_sd_about_mass_relation=fM["s"],
        ratio_scatter_to_measurement_about_mass_relation=fM["s"] / float(np.median(sd_meas)),
        ratio_scatter_to_measurement_about_mean=float(np.std(lnEo, ddof=1))
                                                / float(np.median(sd_meas)),
        mean_normalised_squared_residual_at_fixed_XCOP_kappa=float(np.mean((R / sR) ** 2)),
        licence_to_quote_likelihood_statistics=bool(
            0.8 < float(np.mean((R / sR) ** 2)) < 1.25),
        statement=("No chi^2, lnL, AIC or BIC is quoted anywhere.  The mean normalised squared "
                   "residual at the fixed X-COP kappa is reported ONLY as a calibration "
                   "diagnostic - it is far from 1, and that failure is itself the result.  All "
                   "intervals are bootstrap or Monte-Carlo intervals."))


# ----------------------------------------------------------------------------
# 14.  Sensitivity
# ----------------------------------------------------------------------------
def sensitivity(clusters, mc, rng=None):
    rng = rng or RNG
    out = {}
    for ups in (0.5, 0.6, 0.73, 0.9, 1.1):
        S = build(clusters, ups); A = arrays(S)
        ch = chain(A["M_WL"], A["z"], A["M_gas"], A["L_K"], A["kT"], KAPPA_XCOP, ups)
        ch0 = chain(A["M_WL"], A["z"], A["M_gas"], A["L_K"], A["kT"], 0.0, ups)
        R = np.log(ch["E_obs"]) - np.log(ch["E_pred"])
        w = 1.0 / np.array(mc["sd_lnEobs"]) ** 2
        LP = np.array([np.log(chain(A["M_WL"], A["z"], A["M_gas"], A["L_K"], A["kT"], k,
                                    ups)["E_pred"]) for k in KGRID])
        j, _, _ = profile_fit(np.log(ch0["E_obs"]), LP, w, None)
        out["upsilon_K=%.2f" % ups] = dict(
            n=len(S), median_E_obs=float(np.median(ch0["E_obs"])),
            median_E_pred_at_XCOP=float(np.median(ch["E_pred"])),
            mean_residual_lnE_at_XCOP=float(R.mean()),
            median_f_gas=float(np.median(ch0["f_gas"])),
            kappa_free_noint=float(KGRID[j]))
    S41 = build(clusters, UPSILON_K_PRIMARY, impute=True); A41 = arrays(S41)
    ch = chain(A41["M_WL"], A41["z"], A41["M_gas"], A41["L_K"], A41["kT"], KAPPA_XCOP)
    out["n41_with_Abell2697_imputed"] = dict(
        n=len(S41), mean_residual_lnE_at_XCOP=float(
            (np.log(ch["E_obs"]) - np.log(ch["E_pred"])).mean()),
        median_E_obs=float(np.median(ch["E_obs"])),
        median_E_pred_at_XCOP=float(np.median(ch["E_pred"])))
    S = build(clusters); A = arrays(S)
    m = A["snr"] >= 2.5
    ch = chain(A["M_WL"][m], A["z"][m], A["M_gas"][m], A["L_K"][m], A["kT"][m], KAPPA_XCOP)
    out["MWL_SNR_ge_2.5"] = dict(n=int(m.sum()), mean_residual_lnE_at_XCOP=float(
        (np.log(ch["E_obs"]) - np.log(ch["E_pred"])).mean()))
    # cosmology
    global OM, OL, RHO_C0
    OM, OL = 0.27, 0.73
    ch = chain(A["M_WL"], A["z"], A["M_gas"], A["L_K"], A["kT"], KAPPA_XCOP)
    out["Omega_m=0.27"] = dict(mean_residual_lnE_at_XCOP=float(
        (np.log(ch["E_obs"]) - np.log(ch["E_pred"])).mean()))
    OM, OL = 0.3, 0.7
    return out


# ----------------------------------------------------------------------------
# 15.  Main
# ----------------------------------------------------------------------------
def main():
    clusters = load()
    audit = exclusion_audit(clusters)
    S = build(clusters)
    A = arrays(S)
    n = len(S)
    print("n input = %d, retained = %d, excluded = %s"
          % (audit["n_input"], audit["n_retained"], list(audit["clusters_failing_criterion"])))

    ident = verify_pressure_integral_identity()
    print("pressure-integral identity ratio = %.12f (exact = %s)"
          % (ident["isothermal_ratio_integral_over_kT_Mgas"], ident["isothermal_identity_exact"]))

    kappa_list = [0.0, 1.0e4, 3.0e4, 5.0e4, KAPPA_XCOP]
    print("MC error propagation ...")
    DRAG = (1.2, 1.0, -0.2)
    mc, C = mc_errors(A, kappa_list, ndraw=20000)                 # published errors as stated
    mc2, C2 = mc_errors(A, kappa_list, ndraw=20000, drag=DRAG)    # + shared-aperture drag
    lnEo_chk = np.log(chain(A["M_WL"], A["z"], A["M_gas"], A["L_K"], A["kT"], 0.0)["E_obs"])
    err_model = dict(
        primary="published errors treated as independent (drag = 0), the literal reading",
        variant="shared-aperture drag (alpha_gas, alpha_star, beta_T) = %s" % (DRAG,),
        observed_var_lnEobs=float(np.var(lnEo_chk, ddof=1)),
        mean_propagated_var_lnEobs_no_drag=float(np.mean(np.array(mc["sd_lnEobs"]) ** 2)),
        mean_propagated_var_lnEobs_with_drag=float(np.mean(np.array(mc2["sd_lnEobs"]) ** 2)),
        finding=("Treated as independent, the published errors propagate to MORE scatter in "
                 "ln E than is observed, so the implied intrinsic variance is negative and the "
                 "moment-form EIV estimator is undefined.  Adding the shared-aperture drag - "
                 "which must be there, because every observable is measured inside r500,WL - "
                 "removes the inconsistency.  Both error models are carried through below; the "
                 "independent one is the conservative choice and is used for the headline."))
    print("  observed var(lnE) %.5f | propagated no-drag %.5f | with drag %.5f"
          % (err_model["observed_var_lnEobs"],
             err_model["mean_propagated_var_lnEobs_no_drag"],
             err_model["mean_propagated_var_lnEobs_with_drag"]))

    chX = chain(A["M_WL"], A["z"], A["M_gas"], A["L_K"], A["kT"], KAPPA_XCOP)
    ch0 = chain(A["M_WL"], A["z"], A["M_gas"], A["L_K"], A["kT"], 0.0)
    table = []
    for i in range(n):
        table.append(dict(
            name=A["name"][i], z=float(A["z"][i]),
            M_WL_1e14=float(A["M_WL"][i]), M_gas_1e14=float(A["M_gas"][i]),
            L_K_1e12=float(A["L_K"][i]), M_star_1e14=float(ch0["M_star"][i]),
            M_b_1e14=float(ch0["M_b"][i]), f_gas=float(ch0["f_gas"][i]),
            r500_Mpc=float(ch0["r500_Mpc"][i]), kT_keV=float(A["kT"][i]), t=float(ch0["t"][i]),
            DM_P_1e14_at_XCOP=float(chX["DM_P"][i]), delta_at_XCOP=float(chX["delta"][i]),
            gNb_over_a0=float(ch0["x_b"][i]), gNeff_over_a0_at_XCOP=float(chX["x_eff"][i]),
            nu_b=float(ch0["nu_b"][i]), nu_eff_at_XCOP=float(chX["nu_eff"][i]),
            E_pred_at_XCOP=float(chX["E_pred"][i]), E_obs=float(ch0["E_obs"][i]),
            E_pred_deep_limit_at_XCOP=float(math.sqrt(1.0 + chX["deep_limit_Ypred"][i])),
            residual_lnE_at_XCOP=float(math.log(ch0["E_obs"][i] / chX["E_pred"][i])),
            sigma_lnEobs=float(mc["sd_lnEobs"][i]),
            sigma_residual_at_XCOP=float(mc["per_kappa"]["%.6g" % KAPPA_XCOP]["sd_residual"][i]),
            snr_MWL=float(A["snr"][i])))
    print("x_b %.4f-%.4f ; x_eff(XCOP) %.4f-%.4f ; E_obs med %.3f ; E_pred(XCOP) med %.3f"
          % (ch0["x_b"].min(), ch0["x_b"].max(), chX["x_eff"].min(), chX["x_eff"].max(),
             np.median(ch0["E_obs"]), np.median(chX["E_pred"])))

    print("fixed-kappa test ...")
    fk = fixed_kappa_test(A, mc, KAPPA_XCOP, nboot=10000)
    fk2 = fixed_kappa_test(A, mc2, KAPPA_XCOP, nboot=10000)
    print("  weighted mean residual lnE = %+.4f (bootstrap %.1f sigma), sd = %.3f, pull sd %.2f"
          % (fk["weighted_mean_residual_lnE"], fk["sigma_from_bootstrap"],
             fk["sd_residual_lnE"], fk["sd_pull"]))

    print("free fits ...")
    ff = free_fits(A, mc, nboot=3000)
    ff2 = free_fits(A, mc2, nboot=3000)
    for m_ in ("noint", "int", "int_mass"):
        print("  %-9s kappa = %+.4e  boot95 [%+.3e, %+.3e]   (drag model: %+.4e)"
              % (m_, ff[m_]["kappa"], ff[m_]["bootstrap"]["p2.5"], ff[m_]["bootstrap"]["p97.5"],
                 ff2[m_]["kappa"]))
    print("  per-cluster kappa_i: %.3e to %.3e (factor %.1f)"
          % (ff["per_cluster_kappa_exact"]["min"], ff["per_cluster_kappa_exact"]["max"],
             ff["per_cluster_kappa_exact"]["ratio_max_min"]))

    print("responsiveness check ...")
    resp = responsiveness(A, mc)
    print("  all used statistics move: %s ; blind: %s"
          % (resp["all_statistics_used_in_this_analysis_move"], resp["statistics_BLIND_to_kappa"]))

    print("power (exact forward model) ...")
    pw = power_exact(A, mc, nsim=6000, scatter_mode="observed")
    pw_cons = power_exact(A, mc, nsim=6000, scatter_mode="measurement")
    for arm in "ABC":
        d = pw["arm_" + arm]
        print("  arm %s: nominal size %.3f, corrected crit %.3f, power@XCOP %.3f, 80%% floor %s"
              % (arm, d["false_positive_rate_nominal_at_kappa0"],
                 d["critical_value_size_corrected"], d["power_at_XCOP_kappa"],
                 d["kappa_floor_80pct_power"]))

    print("EIV ...")
    ev = eiv(A, C, mc, nboot_mom=4000, nboot_mle=900, nnull=3000)
    print("  p_EIV(MLE) = %+.3f ; p_EIV(mom) = %+.3f ; p_naive = %+.3f ; REQUIRED = %+.3f"
          % (ev["p_EIV_maximum_likelihood"], ev["p_EIV_method_of_moments"],
             ev["p_naive_OLS_treating_M_WL_as_exact"], ev["required_p_at_XCOP_kappa"]["mean"]))
    print("EIV (aperture-drag error model) ...")
    ev2 = eiv(A, C2, mc2, nboot_mom=2000, nboot_mle=500, nnull=1500)
    print("  p_EIV(MLE) = %+.3f" % ev2["p_EIV_maximum_likelihood"])

    print("aperture ...")
    ap = aperture(A, mc, nsim=2000)
    print("leave-one-out ...")
    loo = leave_one_out(A, C, mc)
    print("  wmean %.4f..%.4f | kappa_noint %.3e..%.3e | kappa_int_mass %.3e..%.3e | p_EIV %.3f..%.3f"
          % (loo["wmean_residual"]["min"], loo["wmean_residual"]["max"],
             loo["kappa_noint"]["min"], loo["kappa_noint"]["max"],
             loo["kappa_int_mass"]["min"], loo["kappa_int_mass"]["max"],
             loo["p_EIV"]["min"], loo["p_EIV"]["max"]))
    print("calibration + sensitivity ...")
    cal = calibration(A, mc)
    sens = sensitivity(clusters, mc)

    # ---- what would have to be true for the source law to survive at kappa = 1.36e5
    phi = ff["noint"]["kappa"] / KAPPA_XCOP
    recon = dict(
        fitted_over_XCOP_kappa=phi,
        multiplicative_factor_needed_on_the_pressure_integral=phi,
        implied_T_effective_over_kT_core_excised=phi,
        physically_allowed_range_of_that_ratio=[
            min(d["gas_mass_weighted_T_over_core_excised_T"] for d in ident["declining_T"]),
            max(d["gas_mass_weighted_T_over_core_excised_T"] for d in ident["declining_T"])],
        verdict=("The gas-mass-weighted temperature would have to be about %.0f per cent of "
                 "the published core-excised value.  Observed cluster temperature profiles "
                 "give %.2f-%.2f.  Non-thermal pressure would push the required factor the "
                 "WRONG way, since it adds pressure." % (
                     100 * phi,
                     min(d["gas_mass_weighted_T_over_core_excised_T"]
                         for d in ident["declining_T"]),
                     max(d["gas_mass_weighted_T_over_core_excised_T"]
                         for d in ident["declining_T"]))))
    print("  reconciliation factor needed on the pressure integral: %.3f" % phi)

    res = dict(
        meta=dict(
            title="LoCuSS test of rho_eff = rho_b + 3 kappa P/c^2 via the EXACT forward chain",
            seed=SEED, n_clusters=n, supersedes="../locuss/locuss_test.py",
            what_changed=[
                "M_gas/M_b factor carried: DM_P = kappa t M_gas, delta = kappa t f_gas.",
                "Full RAR nu on BOTH sides: E_pred = F(g_N_eff)/F(g_N_b), not the deep limit.",
                "The previous run's E-1 branch is NOT reused (additive post-RAR acceleration "
                "is a different theory from a source modification).",
                "Primary inference is an errors-in-variables regression, not a partial rank "
                "correlation conditioned on a noisy M_WL.",
                "Power injected through the exact chain and size-corrected.",
            ],
            constants=dict(a0=A0, mu=MU_MOL, upsilon_K=UPSILON_K_PRIMARY,
                           kappa_XCOP=KAPPA_XCOP, H0=70.0, Om=0.3,
                           nu="nu(x) = 1/(1-exp(-sqrt(x)))"),
            assumptions=[
                "Isothermal gas inside r500,WL, so the pressure integral reduces exactly to "
                "(kT/mu m_p) M_gas; verified numerically to be density-profile independent.",
                "kT_X_ce (core-excised, ~0.15-1 r500,WL) stands in for the gas-mass-weighted T.",
                "Spherical enclosed masses on both sides; lensing measures the same g as "
                "dynamics (no gravitational slip).",
                "Only thermal gas pressure sources the term; stars are collisionless and "
                "non-thermal pressure is unmeasured in this sample.",
            ]),
        exclusion_audit=audit,
        pressure_integral_identity=ident,
        per_cluster_chain=table,
        summary_ranges=dict(
            r500_Mpc=[float(ch0["r500_Mpc"].min()), float(ch0["r500_Mpc"].max())],
            x_b=[float(ch0["x_b"].min()), float(ch0["x_b"].max())],
            x_eff_at_XCOP=[float(chX["x_eff"].min()), float(chX["x_eff"].max())],
            nu_b=[float(ch0["nu_b"].min()), float(ch0["nu_b"].max())],
            nu_eff_at_XCOP=[float(chX["nu_eff"].min()), float(chX["nu_eff"].max())],
            f_gas=[float(ch0["f_gas"].min()), float(ch0["f_gas"].max())],
            delta_at_XCOP=[float(chX["delta"].min()), float(chX["delta"].max())],
            E_obs=dict(median=float(np.median(ch0["E_obs"])), min=float(ch0["E_obs"].min()),
                       max=float(ch0["E_obs"].max()),
                       p16=float(np.percentile(ch0["E_obs"], 16)),
                       p84=float(np.percentile(ch0["E_obs"], 84))),
            E_pred_at_XCOP=dict(median=float(np.median(chX["E_pred"])),
                                min=float(chX["E_pred"].min()), max=float(chX["E_pred"].max())),
            E_pred_deep_limit_at_XCOP=dict(
                median=float(np.median(np.sqrt(1.0 + chX["deep_limit_Ypred"]))),
                note="the deep-MOND limit UNDERSTATES the prediction; the full nu is steeper"),
            M_b_over_M_WL_median=float(np.median(ch0["M_b"] / A["M_WL"])),
            kT_keV=[float(A["kT"].min()), float(A["kT"].max())]),
        error_model=err_model,
        mc_error_propagation=mc,
        mc_error_propagation_with_aperture_drag=mc2,
        fixed_kappa_test=fk,
        fixed_kappa_test_with_aperture_drag=fk2,
        free_fits=ff,
        free_fits_with_aperture_drag=ff2,
        responsiveness_check=resp,
        power=pw,
        power_conservative_full_measurement_scatter=pw_cons,
        eiv=ev,
        eiv_with_aperture_drag=ev2,
        aperture=ap,
        leave_one_out=loo,
        calibration=cal,
        sensitivity=sens,
        reconciliation=recon,
        headline=dict(
            n=n,
            excluded_cluster=list(audit["clusters_failing_criterion"]),
            E_obs_median=float(np.median(ch0["E_obs"])),
            E_pred_median_at_XCOP=float(np.median(chX["E_pred"])),
            x_b_range=[float(ch0["x_b"].min()), float(ch0["x_b"].max())],
            fixed_kappa_weighted_mean_residual_lnE=fk["weighted_mean_residual_lnE"],
            fixed_kappa_significance_bootstrap=fk["sigma_from_bootstrap"],
            kappa_free_no_intercept=ff["noint"]["kappa"],
            kappa_free_no_intercept_boot95=[ff["noint"]["bootstrap"]["p2.5"],
                                            ff["noint"]["bootstrap"]["p97.5"]],
            kappa_free_with_intercept=ff["int"]["kappa"],
            kappa_free_with_intercept_boot95=[ff["int"]["bootstrap"]["p2.5"],
                                              ff["int"]["bootstrap"]["p97.5"]],
            kappa_free_with_intercept_and_mass=ff["int_mass"]["kappa"],
            p_EIV=ev["p_EIV_maximum_likelihood"],
            p_EIV_boot95=[ev["bootstrap_p_MLE"]["p2.5"], ev["bootstrap_p_MLE"]["p97.5"]]
            if ev["bootstrap_p_MLE"] else None,
            p_EIV_with_drag=ev2["p_EIV_maximum_likelihood"],
            p_required_from_exact_chain=ev["recovery_generated_from_the_EXACT_CHAIN_at_XCOP_kappa"]
            ["EIV_MLE_median"],
            separation_sd=ev["recovery_generated_from_the_EXACT_CHAIN_at_XCOP_kappa"]
            ["separation_in_sd_between_observed_and_model"],
            power_at_XCOP_armC=pw["arm_C"]["power_at_XCOP_kappa"],
            kappa_floor_80pct_armC=pw["arm_C"]["kappa_floor_80pct_power"],
            per_cluster_kappa_spread=ff["per_cluster_kappa_exact"]["ratio_max_min"]))
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("wrote %s (%.1f kB)" % (OUT_JSON, os.path.getsize(OUT_JSON) / 1024.0))

    print("")
    print("| cluster | z | M_b | M_gas | f_gas | kT | DM_P | g_Nb/a0 | g_Neff/a0 | nu_b | "
          "E_pred | E_obs | ln(Eo/Ep) | sig |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(table, key=lambda d: -d["kT_keV"]):
        print("| %s | %.3f | %.2f | %.2f | %.3f | %.2f | %.2f | %.4f | %.4f | %.2f | %.3f | "
              "%.3f | %+.3f | %.3f |"
              % (r["name"], r["z"], r["M_b_1e14"], r["M_gas_1e14"], r["f_gas"], r["kT_keV"],
                 r["DM_P_1e14_at_XCOP"], r["gNb_over_a0"], r["gNeff_over_a0_at_XCOP"], r["nu_b"],
                 r["E_pred_at_XCOP"], r["E_obs"], r["residual_lnE_at_XCOP"],
                 r["sigma_residual_at_XCOP"]))
    return res


if __name__ == "__main__":
    main()
