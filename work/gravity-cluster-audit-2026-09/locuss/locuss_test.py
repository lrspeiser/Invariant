#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoCuSS test of the pressure-sourced gravity modification

    rho_eff = rho_b + 3 kappa P / c^2
 => M_eff(<r) = M_b(<r) + (3 kappa / c^2) Int_0^r 4 pi r'^2 P(r') dr'

Decisive feature of this sample: the gravity measurement (M_WL) is weak lensing,
NOT gas hydrostatics.  Temperature therefore does not appear on both sides of the
correlation, which is what made the earlier X-COP version uninterpretable.

Runs start to finish from a fresh process.  Writes locuss_results.json.

Author: analysis for the pressure-model test, 2026-09-03
"""

import json
import math
import os
import sys

import numpy as np
from scipy import stats

# ----------------------------------------------------------------------------
# 0.  Paths
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ACQ = os.path.abspath(os.path.join(HERE, "..", "acquire"))
SAMPLE_TSV = os.path.join(ACQ, "mulroy2019_sample.tsv")
OBS_TSV = os.path.join(ACQ, "mulroy2019_observables.tsv")
OUT_JSON = os.path.join(HERE, "locuss_results.json")

RNG = np.random.default_rng(20260903)

# ----------------------------------------------------------------------------
# 1.  PRE-REGISTRATION  (fixed before any residual was inspected)
# ----------------------------------------------------------------------------
PREREG = {
    "stellar_mass_to_light": {
        "value_Msun_per_Lsun_K": 0.73,
        "band": "rest-frame K (2.2 um), Vega, M_K_sun = 3.39 as used by Mulroy+2019",
        "source": (
            "Old passive cluster-galaxy population with a Chabrier IMF. "
            "Bell et al. (2003) 2MASS+SDSS give log10(M/L_K) = -0.206 + 0.135*(B-V) "
            "on a 'diet Salpeter' scale; for a typical red-sequence B-V = 0.90 this is "
            "M/L_K = 0.82, and subtracting 0.05 dex to go from diet-Salpeter to Chabrier "
            "gives 0.73.  Consistent with the ATLAS-3D / Cappellari+(2013) dynamical "
            "M/L for early types once translated to K band."
        ),
        "status": "GLOBAL nuisance, single value for all clusters, NOT fitted",
        "secondary_sensitivity_grid": [0.5, 0.6, 0.73, 0.9, 1.1],
    },
    "quality_cuts": [
        "Require M_WL, kT_X_ce, M_gas and L_K_tot all present -> primary sample.",
        "No dynamical-state / merger cut (weak lensing does not assume equilibrium; "
        "that is the whole point of using this sample).",
        "No weak-lensing S/N cut in the primary (a S/N cut at 2.0 would sit exactly on "
        "ZwCl0857.9+2107 and would be a knife-edge decision).",
    ],
    "secondary_analyses_declared_in_advance": [
        "(a) n=41 with Abell2697 stellar mass imputed at the sample-median M_star/M_gas.",
        "(b) M_WL signal-to-noise >= 2.5 subsample.",
        "(c) Upsilon_K sensitivity grid (above).",
        "(d) drop the 4 clusters with the largest |residual| in the primary fit.",
        "(e) the physically-derived isothermal form  E - 1 = kappa * t * f_gas  "
        "(see the approximation discussion), reported alongside the mandated compressed form.",
    ],
    "missing_columns": {
        "Abell2697": "no UKIRT/WFCAM NIR data -> L_K_tot and L_K_BCG absent. "
                     "Dropped from the primary sample (n=40); imputed only in secondary (a).",
        "unused_but_incomplete": "Y_SZA missing for 11, lambda for 8, Y_X for 2. "
                                 "None of these enter this analysis.",
    },
    "primary_statistic": {
        "name": "partial Spearman rank correlation rho_p( E^2 - 1 , kT_X_ce | M_WL )",
        "why": (
            "kT and cluster mass are strongly correlated, and E is built from M_WL, so the "
            "raw E-kT correlation is confounded.  The partial correlation at fixed M_WL is "
            "the single number that asks whether temperature carries information about the "
            "gravity excess beyond what mass already carries."
        ),
        "threshold": "two-sided p < 0.05, p from a >=20000-draw permutation null",
        "note": "ONE primary statistic.  Everything else is a required supporting check.",
    },
    "cosmology": "flat LCDM, Omega_m = 0.3, Omega_L = 0.7, H0 = 70 km/s/Mpc "
                 "(exactly Mulroy+2019 Sec.1, so r_500 reproduces their aperture)",
    "fixed_constants": {
        "a0": 1.2e-10,
        "nu": "nu(x) = 1/(1 - exp(-sqrt(x))),  x = g_bar/a0   (McGaugh+2016 RAR)",
        "mu": 0.6,
        "kappa_XCOP_reference": 1.36e5,
    },
}

# ----------------------------------------------------------------------------
# 2.  Physical constants (SI)
# ----------------------------------------------------------------------------
G_SI = 6.67430e-11          # m^3 kg^-1 s^-2
MSUN = 1.98892e30           # kg
MPC = 3.0856775814913673e22  # m
C_LIGHT = 2.99792458e8      # m/s
M_P = 1.67262192369e-27     # kg
KEV_J = 1.602176634e-16     # J per keV
A0 = 1.2e-10                # m/s^2
MU_MOL = 0.6
H0_SI = 70.0 * 1000.0 / MPC  # s^-1
OM, OL = 0.3, 0.7
KAPPA_XCOP = 1.36e5
UPSILON_K_PRIMARY = 0.73

RHO_C0 = 3.0 * H0_SI ** 2 / (8.0 * math.pi * G_SI)   # kg/m^3


def Ez2(z):
    return OM * (1.0 + z) ** 3 + OL


def rho_c(z):
    return RHO_C0 * Ez2(z)


def r500_from_M500(M500_kg, z):
    """Spherical overdensity radius: M = (4/3) pi r^3 * 500 * rho_c(z)."""
    return (3.0 * M500_kg / (4.0 * math.pi * 500.0 * rho_c(z))) ** (1.0 / 3.0)


def nu_rar(x):
    """RAR interpolation function nu(x) = 1/(1 - exp(-sqrt(x)))."""
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 - np.exp(-np.sqrt(x)))


# ----------------------------------------------------------------------------
# 3.  Data loading
# ----------------------------------------------------------------------------
def read_tsv(path):
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln.rstrip("\n").rstrip("\r") for ln in fh if ln.strip() != ""]
    hdr = lines[0].split("\t")
    rows = []
    for ln in lines[1:]:
        parts = ln.split("\t")
        # pad short rows (trailing empty cells can be dropped by some writers)
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
        ))
    return cl


# ----------------------------------------------------------------------------
# 4.  Core derivation:  enclosed mass -> acceleration -> RAR -> enclosed mass
# ----------------------------------------------------------------------------
def derive(M_WL_1e14, z, M_gas_1e14, L_K_1e12, upsilon_K):
    """
    Returns dict with r500 [Mpc], M_star, M_b [1e14 Msun], g_bar [m/s^2],
    x, nu, M_pred [1e14 Msun], E.

    Conversion chain, stated explicitly:
      (1) r_500 is the weak-lensing overdensity radius, r_500 = [3 M_WL /
          (4 pi 500 rho_c(z))]^(1/3).  This is exactly Mulroy+2019's aperture,
          so M_gas and L_K are the baryons INSIDE this same sphere.
      (2) g_bar(r_500) = G M_b(<r_500) / r_500^2.  Spherical Newtonian.  Legitimate
          because M_WL is itself an NFW spherical enclosed mass, so both sides use
          the same geometry.
      (3) g_obs = nu(g_bar/a0) * g_bar   with nu(x) = 1/(1-exp(-sqrt(x))).
      (4) M_pred(<r_500) = g_obs r_500^2 / G = nu(x) * M_b(<r_500).
          The r_500^2/G factors cancel, so M_pred/M_b = nu exactly; but r_500 is
          still needed to evaluate x, and that is where the aperture enters.
      (5) E = M_WL / M_pred.
    """
    M_WL_kg = M_WL_1e14 * 1e14 * MSUN
    r500_m = r500_from_M500(M_WL_kg, z)

    M_star_1e14 = upsilon_K * L_K_1e12 * 1e12 / 1e14      # -> 1e14 Msun
    M_b_1e14 = M_gas_1e14 + M_star_1e14
    M_b_kg = M_b_1e14 * 1e14 * MSUN

    g_bar = G_SI * M_b_kg / r500_m ** 2
    x = g_bar / A0
    nu = float(nu_rar(x))
    M_pred_1e14 = nu * M_b_1e14
    E = M_WL_1e14 / M_pred_1e14
    return dict(
        r500_Mpc=r500_m / MPC,
        M_star=M_star_1e14,
        M_b=M_b_1e14,
        f_gas_of_baryons=M_gas_1e14 / M_b_1e14,
        g_bar=g_bar,
        x=x,
        nu=nu,
        g_obs=nu * g_bar,
        M_pred=M_pred_1e14,
        E=E,
    )


def t_of_kT(kT_keV):
    """t = 3 kT / (mu m_p c^2), dimensionless."""
    return 3.0 * np.asarray(kT_keV, float) * KEV_J / (MU_MOL * M_P * C_LIGHT ** 2)


# ----------------------------------------------------------------------------
# 5.  Statistics helpers
# ----------------------------------------------------------------------------
def ols(x, y):
    """y = a + b x.  Returns a, b, SE_a, SE_b, resid, s (residual sd)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
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


def ols_noint(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    b = float(x @ y / (x @ x))
    resid = y - b * x
    dof = len(x) - 1
    s2 = float(resid @ resid) / dof
    se = float(np.sqrt(s2 / (x @ x)))
    return dict(b=b, se_b=se, resid=resid, s=float(np.sqrt(s2)), dof=dof)


def partial_spearman(a, b, c):
    """rank-based partial correlation of a and b controlling for c."""
    ra = stats.rankdata(a); rb = stats.rankdata(b); rc = stats.rankdata(c)
    rab = np.corrcoef(ra, rb)[0, 1]
    rac = np.corrcoef(ra, rc)[0, 1]
    rbc = np.corrcoef(rb, rc)[0, 1]
    den = math.sqrt(max(1e-15, (1 - rac ** 2) * (1 - rbc ** 2)))
    return (rab - rac * rbc) / den


def partial_p_analytic(r, n, k=1):
    """two-sided p for a partial correlation with k controlled variables."""
    dof = n - 2 - k
    if abs(r) >= 1.0:
        return 0.0
    tstat = r * math.sqrt(dof / (1.0 - r ** 2))
    return float(2.0 * stats.t.sf(abs(tstat), dof))


def sigma_ln(val, ep, em):
    """symmetric log-normal sigma from asymmetric linear errors (Mulroy assume log-normal)."""
    hi = math.log1p(ep / val)
    frac = min(em / val, 0.95)
    lo = -math.log1p(-frac)
    return 0.5 * (hi + lo)


# ----------------------------------------------------------------------------
# 6.  Build the primary sample
# ----------------------------------------------------------------------------
def build(clusters, upsilon_K, impute_missing_LK=False):
    med_ratio = None
    if impute_missing_LK:
        ratios = [c["L_K"] * 1e12 / (c["M_gas"] * 1e14) for c in clusters if c["L_K"] is not None]
        med_ratio = float(np.median(ratios))

    out = []
    for c in clusters:
        L_K = c["L_K"]
        imputed = False
        if L_K is None:
            if not impute_missing_LK:
                continue
            L_K = med_ratio * (c["M_gas"] * 1e14) / 1e12
            imputed = True
        d = derive(c["M_WL"], c["z"], c["M_gas"], L_K, upsilon_K)
        rec = dict(c)
        rec.update(d)
        rec["L_K_used"] = L_K
        rec["L_K_imputed"] = imputed
        rec["t"] = float(t_of_kT(c["kT"]))
        rec["Y"] = d["E"] ** 2 - 1.0            # mandated compressed-form response
        rec["Y_lin"] = d["E"] - 1.0             # physically-derived isothermal response
        rec["kappa_implied_compressed"] = rec["Y"] / rec["t"]
        rec["kappa_implied_isothermal"] = rec["Y_lin"] / (rec["t"] * d["f_gas_of_baryons"])
        rec["snr_MWL"] = c["M_WL"] / (0.5 * (c["M_WL_ep"] + c["M_WL_em"]))
        out.append(rec)
    return out


# ----------------------------------------------------------------------------
# 7.  Headline fits + required checks
# ----------------------------------------------------------------------------
def headline(S, nboot=10000, rng=None):
    rng = rng or RNG
    t = np.array([s["t"] for s in S])
    Y = np.array([s["Y"] for s in S])
    Ylin = np.array([s["Y_lin"] for s in S])
    kT = np.array([s["kT"] for s in S])
    MWL = np.array([s["M_WL"] for s in S])
    fgas = np.array([s["f_gas_of_baryons"] for s in S])
    E = np.array([s["E"] for s in S])
    n = len(S)

    free = ols(t, Y)
    zero = ols_noint(t, Y)
    iso = ols_noint(t * fgas, Ylin)          # physically-derived form, zero intercept
    iso_free = ols(t * fgas, Ylin)

    # bootstrap over clusters
    ba, bb, bz, biso, brho = [], [], [], [], []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        f = ols(t[idx], Y[idx])
        ba.append(f["a"]); bb.append(f["b"])
        bz.append(ols_noint(t[idx], Y[idx])["b"])
        biso.append(ols_noint((t * fgas)[idx], Ylin[idx])["b"])
        jit = rng.normal(0, 1e-9, n)
        brho.append(partial_spearman(Y[idx] + jit, kT[idx] + jit, MWL[idx] + jit))
    q = lambda v, p: float(np.percentile(v, p))

    rho_p = partial_spearman(Y, kT, MWL)
    res = dict(
        n=n,
        fit_free_intercept=dict(
            intercept=free["a"], intercept_se=free["se_a"],
            intercept_boot68=[q(ba, 16), q(ba, 84)], intercept_boot95=[q(ba, 2.5), q(ba, 97.5)],
            kappa=free["b"], kappa_se=free["se_b"],
            kappa_boot68=[q(bb, 16), q(bb, 84)], kappa_boot95=[q(bb, 2.5), q(bb, 97.5)],
            kappa_one_sided_95pct_upper_limit=q(bb, 95),
            kappa_one_sided_99pct_upper_limit=q(bb, 99),
            residual_sd=free["s"],
            intercept_consistent_with_zero=bool(q(ba, 2.5) <= 0.0 <= q(ba, 97.5)),
            intercept_over_se=free["a"] / free["se_a"],
        ),
        fit_zero_intercept=dict(
            kappa=zero["b"], kappa_se=zero["se_b"],
            kappa_boot68=[q(bz, 16), q(bz, 84)], kappa_boot95=[q(bz, 2.5), q(bz, 97.5)],
            residual_sd=zero["s"],
        ),
        fit_isothermal_exact_form=dict(
            model="E - 1 = kappa * t * f_gas   (gas-only isothermal pressure integral)",
            kappa_zero_intercept=iso["b"], kappa_se=iso["se_b"],
            kappa_boot95=[q(biso, 2.5), q(biso, 97.5)],
            intercept_if_free=iso_free["a"], kappa_if_free=iso_free["b"],
            intercept_if_free_se=iso_free["se_a"], kappa_if_free_se=iso_free["se_b"],
        ),
        correlations=dict(
            spearman_Y_kT=list(map(float, stats.spearmanr(Y, kT))),
            spearman_Y_MWL=list(map(float, stats.spearmanr(Y, MWL))),
            spearman_kT_MWL=list(map(float, stats.spearmanr(kT, MWL))),
            pearson_Y_kT=list(map(float, stats.pearsonr(Y, kT))),
            spearman_E_kT=list(map(float, stats.spearmanr(E, kT))),
        ),
        primary_statistic=dict(
            name="partial Spearman rho_p(E^2-1, kT | M_WL)",
            value=rho_p,
            boot68=[q(brho, 16), q(brho, 84)],
            boot95=[q(brho, 2.5), q(brho, 97.5)],
            p_analytic=partial_p_analytic(rho_p, n, 1),
        ),
        E_summary=dict(
            median=float(np.median(E)), mean=float(np.mean(E)),
            min=float(E.min()), max=float(E.max()), sd=float(E.std(ddof=1)),
        ),
        kappa_implied_percluster=dict(
            compressed_median=float(np.median([s["kappa_implied_compressed"] for s in S])),
            compressed_range=[float(np.min([s["kappa_implied_compressed"] for s in S])),
                              float(np.max([s["kappa_implied_compressed"] for s in S]))],
            isothermal_median=float(np.median([s["kappa_implied_isothermal"] for s in S])),
        ),
    )
    return res, dict(t=t, Y=Y, Ylin=Ylin, kT=kT, MWL=MWL, fgas=fgas, E=E)


# ----------------------------------------------------------------------------
# 8.  Check 2 - compatibility with the X-COP kappa
# ----------------------------------------------------------------------------
def compare_xcop(fit_free, fit_zero):
    out = {}
    for tag, f in (("free_intercept", fit_free), ("zero_intercept", fit_zero)):
        k, se = f["kappa"], f["kappa_se"]
        lo, hi = f["kappa_boot95"]
        out[tag] = dict(
            kappa=k, kappa_se=se, boot95=[lo, hi],
            n_sigma_from_XCOP=(k - KAPPA_XCOP) / se,
            XCOP_inside_boot95=bool(lo <= KAPPA_XCOP <= hi),
            ratio_kappa_over_XCOP=k / KAPPA_XCOP,
        )
    return out


# ----------------------------------------------------------------------------
# 9.  Check 4 - leave-one-cluster-out
# ----------------------------------------------------------------------------
def loo(S):
    n = len(S)
    rows = []
    for i in range(n):
        sub = [s for j, s in enumerate(S) if j != i]
        t = np.array([s["t"] for s in sub]); Y = np.array([s["Y"] for s in sub])
        kT = np.array([s["kT"] for s in sub]); MWL = np.array([s["M_WL"] for s in sub])
        f = ols(t, Y); z = ols_noint(t, Y)
        r = partial_spearman(Y, kT, MWL)
        rows.append(dict(dropped=S[i]["name"], intercept=f["a"], kappa_free=f["b"],
                         kappa_zero=z["b"], rho_partial=r,
                         p_partial=partial_p_analytic(r, n - 1, 1),
                         spearman_Y_kT=float(stats.spearmanr(Y, kT)[0])))
    def rng_of(key):
        v = [x[key] for x in rows]
        lo, hi = min(v), max(v)
        return dict(min=lo, max=hi,
                    argmin=rows[int(np.argmin(v))]["dropped"],
                    argmax=rows[int(np.argmax(v))]["dropped"])
    return dict(rows=rows,
                range_intercept=rng_of("intercept"),
                range_kappa_free=rng_of("kappa_free"),
                range_kappa_zero=rng_of("kappa_zero"),
                range_rho_partial=rng_of("rho_partial"),
                max_p_partial=max(x["p_partial"] for x in rows),
                min_p_partial=min(x["p_partial"] for x in rows))


# ----------------------------------------------------------------------------
# 10.  Check 5 - shared-aperture dependence
# ----------------------------------------------------------------------------
def dlnnu_dlnx(x):
    s = math.sqrt(x); u = math.exp(-s)
    return -s * u / (2.0 * (1.0 - u))


def aperture_analytic(S):
    """
    Every LoCuSS observable is measured inside r_500,WL, and r_500,WL propto M_WL^(1/3).
    A WL mass error delta = dln M_WL therefore moves the aperture and drags M_gas,
    L_K and kT with it:
        dln r500   = delta/3
        dln M_gas  = alpha_g * delta/3
        dln L_K    = alpha_s * delta/3
        dln kT     = beta_T  * delta/3
    Propagate analytically to dln E and dln kT and form the induced correlation.
    """
    grid = []
    for alpha_g in (0.8, 1.0, 1.2, 1.5):
        for alpha_s in (0.6, 1.0, 1.4):
            for beta_T in (-0.4, -0.2, 0.0, 0.2):
                dlnE, dlnT, w = [], [], []
                for s in S:
                    fg = s["f_gas_of_baryons"]; fs = 1.0 - fg
                    A = (fg * alpha_g + fs * alpha_s) / 3.0        # dln M_b / d delta
                    dlnx = A - 2.0 / 3.0                            # x propto M_b r500^-2
                    dE = 1.0 - dlnnu_dlnx(s["x"]) * dlnx - A        # dln E / d delta
                    dlnE.append(dE)
                    dlnT.append(beta_T / 3.0)
                    w.append(sigma_ln(s["M_WL"], s["M_WL_ep"], s["M_WL_em"]))
                dlnE = np.array(dlnE); dlnT = np.array(dlnT); w = np.array(w)
                cov = float(np.mean(dlnE * dlnT * w ** 2))
                var_lnE_ap = float(np.mean(dlnE ** 2 * w ** 2))
                var_lnT_ap = float(np.mean(dlnT ** 2 * w ** 2))
                sd_lnE_obs = float(np.std(np.log([s["E"] for s in S]), ddof=1))
                sd_lnT_obs = float(np.std(np.log([s["kT"] for s in S]), ddof=1))
                r_ind = cov / (sd_lnE_obs * sd_lnT_obs)
                grid.append(dict(alpha_gas=alpha_g, alpha_star=alpha_s, beta_T=beta_T,
                                 mean_dlnE_ddelta=float(dlnE.mean()),
                                 induced_cov_lnE_lnT=cov,
                                 induced_corr_vs_observed_spread=r_ind,
                                 aperture_share_of_lnE_variance=var_lnE_ap / sd_lnE_obs ** 2,
                                 aperture_share_of_lnT_variance=var_lnT_ap / sd_lnT_obs ** 2))
    return grid


def solve_Mb(M_WL, z, M_pred_target):
    """Invert  nu(x(M_b)) * M_b = M_pred_target  at fixed r_500(M_WL).  Monotone -> bisect."""
    M_WL_kg = M_WL * 1e14 * MSUN
    r500_m = r500_from_M500(M_WL_kg, z)
    def F(mb):
        g = G_SI * mb * 1e14 * MSUN / r500_m ** 2
        return float(nu_rar(g / A0)) * mb - M_pred_target
    lo, hi = 1e-6, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if F(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_null_world(S, rng):
    """
    A genuine null: E carries NO information about kT once M_WL is fixed, but the
    observed E-M_WL relation and the observed scatter are preserved.

      ln E_null = OLS fit of ln E on ln M_WL, with the residuals PERMUTED.

    The baryons are then solved backwards so that this E_null is exactly what the
    RAR would give: M_b such that nu(x) M_b = M_WL / E_null, split into gas and
    stars at each cluster's own observed gas fraction.  kT_true is left at its
    observed value and is now independent of E_null by construction.
    """
    n = len(S)
    lnE = np.log([s["E"] for s in S])
    lnM = np.log([s["M_WL"] for s in S])
    f = ols(lnM, lnE)
    fit = f["a"] + f["b"] * lnM
    E_null = np.exp(fit + f["resid"][rng.permutation(n)])
    Mg, LK = np.empty(n), np.empty(n)
    for i, s in enumerate(S):
        Mb = solve_Mb(s["M_WL"], s["z"], s["M_WL"] / E_null[i])
        Mg[i] = s["f_gas_of_baryons"] * Mb
        LK[i] = (1.0 - s["f_gas_of_baryons"]) * Mb * 1e14 / (UPSILON_K_PRIMARY * 1e12)
    return Mg, LK


def aperture_montecarlo(S, nsim=2000, rng=None):
    """
    CHECK 5.  How much of the observed E-kT relation can the shared aperture
    manufacture on its own?

    Every LoCuSS observable is measured inside r_500,WL and r_500,WL propto
    M_WL^(1/3), so a weak-lensing mass error delta drags M_gas, L_K and kT along
    with it.  We build a world in which E genuinely has no kT dependence at fixed
    M_WL (build_null_world), then push the WL error through the shared aperture:

        M_WL_meas = M_WL_true exp(delta)
        M_gas_meas = M_gas_true exp(alpha_g delta/3)
        L_K_meas   = L_K_true   exp(alpha_s delta/3)
        kT_meas    = kT_true    exp(beta_T delta/3 + eta),  eta = kT measurement noise

    and measure what correlation appears.  Each configuration is run in THREE arms
    on the same draws, which separates three distinct effects:

      CLEAN : delta = 0.  No WL error at all.  Validates the null construction
              (rho_p must come out at ~0).
      OFF   : WL error present, aperture coupling switched off
              (alpha_g = alpha_s = beta_T = 0).  Isolates REGRESSION DILUTION:
              the partial correlation conditions on the *measured* M_WL, which is
              a noisy version of the true M_WL, so M_WL is only partly controlled.
      ON    : WL error present and dragging M_gas, L_K and kT through the shared
              aperture.  ON minus OFF is the aperture-manufactured amount.
    """
    rng = rng or RNG
    n = len(S)
    sig = np.array([sigma_ln(s["M_WL"], s["M_WL_ep"], s["M_WL_em"]) for s in S])
    sig_kT = np.array([sigma_ln(s["kT"], s["kT_ep"], s["kT_em"]) for s in S])
    M_true = np.array([s["M_WL"] for s in S])
    kT_true = np.array([s["kT"] for s in S])
    z = np.array([s["z"] for s in S])

    configs = [(1.2, 1.0, -0.2), (1.2, 1.0, 0.0), (1.0, 1.0, -0.4),
               (1.5, 1.4, 0.2), (1.2, 1.0, -0.6)]
    acc = {c: dict(on_rho=[], off_rho=[], clean_rho=[], on_raw=[], off_raw=[],
                   on_kap=[], off_kap=[], clean_kap=[]) for c in configs}

    for _ in range(nsim):
        Mg_true, LK_true = build_null_world(S, rng)
        d = rng.normal(0.0, sig, n)
        eta = rng.normal(0.0, sig_kT, n)
        for cfg in configs:
            ag, as_, bT = cfg
            arms = (("on", (ag, as_, bT), d),
                    ("off", (0.0, 0.0, 0.0), d),
                    ("clean", (0.0, 0.0, 0.0), np.zeros(n)))
            for tag, (a1, a2, b1), dd in arms:
                Mw = M_true * np.exp(dd)
                Mg = Mg_true * np.exp(a1 * dd / 3.0)
                LK = LK_true * np.exp(a2 * dd / 3.0)
                kT = kT_true * np.exp(b1 * dd / 3.0 + eta)
                E = np.array([derive(Mw[i], z[i], Mg[i], LK[i], UPSILON_K_PRIMARY)["E"]
                              for i in range(n)])
                Y = E ** 2 - 1.0
                acc[cfg][tag + "_rho"].append(partial_spearman(Y, kT, Mw))
                acc[cfg][tag + "_kap"].append(ols(t_of_kT(kT), Y)["b"])
                if tag in ("on", "off"):
                    acc[cfg][tag + "_raw"].append(float(stats.spearmanr(Y, kT)[0]))

    obs_rho = partial_spearman(np.array([s["Y"] for s in S]),
                               np.array([s["kT"] for s in S]),
                               np.array([s["M_WL"] for s in S]))
    out = {}
    for cfg, v in acc.items():
        on_r = np.array(v["on_rho"]); off_r = np.array(v["off_rho"])
        cl_r = np.array(v["clean_rho"])
        on_k = np.array(v["on_kap"]); off_k = np.array(v["off_kap"])
        cl_k = np.array(v["clean_kap"])
        key = "alpha_gas=%.1f_alpha_star=%.1f_beta_T=%+.1f" % cfg
        out[key] = dict(
            nsim=nsim,
            rho_partial_CLEAN_no_WL_error=dict(
                mean=float(cl_r.mean()), sd=float(cl_r.std(ddof=1)),
                p2_5=float(np.percentile(cl_r, 2.5)), p97_5=float(np.percentile(cl_r, 97.5)),
                note="null-construction validation: should be ~0"),
            kappa_CLEAN_no_WL_error_mean=float(cl_k.mean()),
            regression_dilution_OFF_minus_CLEAN=dict(
                mean=float((off_r - cl_r).mean()), sd=float((off_r - cl_r).std(ddof=1)),
                p2_5=float(np.percentile(off_r - cl_r, 2.5)),
                p97_5=float(np.percentile(off_r - cl_r, 97.5)),
                note="bias from conditioning on a NOISY M_WL rather than the true one"),
            rho_partial_aperture_ON=dict(mean=float(on_r.mean()), sd=float(on_r.std(ddof=1)),
                                         p2_5=float(np.percentile(on_r, 2.5)),
                                         p97_5=float(np.percentile(on_r, 97.5))),
            rho_partial_aperture_OFF=dict(mean=float(off_r.mean()), sd=float(off_r.std(ddof=1)),
                                          p2_5=float(np.percentile(off_r, 2.5)),
                                          p97_5=float(np.percentile(off_r, 97.5))),
            manufactured_rho_ON_minus_OFF=dict(
                mean=float((on_r - off_r).mean()), sd=float((on_r - off_r).std(ddof=1)),
                p2_5=float(np.percentile(on_r - off_r, 2.5)),
                p97_5=float(np.percentile(on_r - off_r, 97.5))),
            manufactured_kappa_ON_minus_OFF=dict(
                mean=float((on_k - off_k).mean()), sd=float((on_k - off_k).std(ddof=1)),
                p2_5=float(np.percentile(on_k - off_k, 2.5)),
                p97_5=float(np.percentile(on_k - off_k, 97.5))),
            kappa_aperture_ON=dict(mean=float(on_k.mean()), sd=float(on_k.std(ddof=1)),
                                   p2_5=float(np.percentile(on_k, 2.5)),
                                   p97_5=float(np.percentile(on_k, 97.5))),
            raw_spearman_aperture_ON_mean=float(np.mean(v["on_raw"])),
            raw_spearman_aperture_OFF_mean=float(np.mean(v["off_raw"])),
            fraction_of_null_reaching_observed_rho=float((on_r <= obs_rho).mean()),
        )
    out["observed_rho_partial_for_reference"] = obs_rho
    return out


# ----------------------------------------------------------------------------
# 11.  Check 6 - permutation nulls
# ----------------------------------------------------------------------------
def permutation_null(S, ndraw=20000, rng=None):
    rng = rng or RNG
    t = np.array([s["t"] for s in S]); Y = np.array([s["Y"] for s in S])
    kT = np.array([s["kT"] for s in S]); MWL = np.array([s["M_WL"] for s in S])
    n = len(S)
    obs_rho = partial_spearman(Y, kT, MWL)
    obs_k = ols(t, Y)["b"]
    obs_raw = float(stats.spearmanr(Y, kT)[0])

    rho_n, kap_n, raw_n = np.empty(ndraw), np.empty(ndraw), np.empty(ndraw)
    for i in range(ndraw):
        p = rng.permutation(n)
        rho_n[i] = partial_spearman(Y, kT[p], MWL)
        kap_n[i] = ols(t[p], Y)["b"]
        raw_n[i] = stats.spearmanr(Y, kT[p])[0]

    # restricted permutation: shuffle kT only WITHIN blocks of similar M_WL,
    # which preserves the kT-M_WL relation that plain shuffling destroys.
    order = np.argsort(MWL)
    blocks = [order[i:i + 5] for i in range(0, n, 5)]
    rho_b = np.empty(ndraw)
    for i in range(ndraw):
        kTb = kT.copy()
        for bl in blocks:
            kTb[bl] = kT[rng.permutation(bl)]
        rho_b[i] = partial_spearman(Y, kTb, MWL)

    def dist(v):
        return dict(
            mean=float(v.mean()), sd=float(v.std(ddof=1)),
            percentiles={str(p): float(np.percentile(v, p))
                         for p in (0.5, 1, 2.5, 5, 10, 16, 25, 50, 75, 84, 90, 95, 97.5, 99, 99.5)},
            min=float(v.min()), max=float(v.max()),
            histogram=dict(zip(["counts", "edges"],
                               [list(map(int, np.histogram(v, bins=40)[0])),
                                list(map(float, np.histogram(v, bins=40)[1]))])),
        )

    return dict(
        ndraw=ndraw,
        observed=dict(rho_partial=obs_rho, kappa_free=obs_k, raw_spearman=obs_raw),
        null_rho_partial=dist(rho_n),
        null_kappa_free=dist(kap_n),
        null_raw_spearman=dist(raw_n),
        null_rho_partial_block_restricted=dist(rho_b),
        p_two_sided_rho_partial=float((np.abs(rho_n) >= abs(obs_rho)).mean()),
        p_two_sided_kappa=float((np.abs(kap_n - kap_n.mean()) >= abs(obs_k - kap_n.mean())).mean()),
        p_two_sided_raw=float((np.abs(raw_n) >= abs(obs_raw)).mean()),
        p_two_sided_rho_block=float((np.abs(rho_b) >= abs(obs_rho)).mean()),
        p_one_sided_rho_partial_positive=float((rho_n >= obs_rho).mean()),
        analytic_vs_permutation=dict(
            analytic_sd_of_rho_under_null=float(1.0 / math.sqrt(len(S) - 3)),
            permutation_sd_of_rho=float(rho_n.std(ddof=1)),
            note="if these agree the analytic t-approximation is safe to use for power",
        ),
    )


# ----------------------------------------------------------------------------
# 12.  Check 7 - power
# ----------------------------------------------------------------------------
def power_curves(S, fit_free, nsim=2000, rng=None):
    rng = rng or RNG
    t = np.array([s["t"] for s in S]); Y = np.array([s["Y"] for s in S])
    kT = np.array([s["kT"] for s in S]); MWL = np.array([s["M_WL"] for s in S])
    n = len(S)
    a_hat, k_hat, s_res = fit_free["intercept"], fit_free["kappa"], fit_free["residual_sd"]
    resid = Y - (a_hat + k_hat * t)

    # Null baseline for the PRIMARY statistic: keep the Y-M_WL relation, destroy
    # only the part of Y that could carry kT information at fixed M_WL.  Removing
    # the OLS trend in t is NOT enough (the rank partial correlation survives it),
    # so we permute the residuals of Y on M_WL.  Verified below: injecting
    # kappa = 0 recovers a ~5 per cent type-I rate.
    fM = ols(MWL, Y)
    baseY = fM["a"] + fM["b"] * MWL
    residM = fM["resid"]
    tc = t - t.mean()

    kappa_grid = [0.0, 5e3, 1e4, 2e4, 3e4, 5e4, 7.5e4, 1.0e5, 1.36e5, 2.0e5, 3.0e5, 5.0e5]
    rows = []
    for kin in kappa_grid:
        hit_slope, hit_rho = 0, 0
        for _ in range(nsim):
            # (P1) slope test: homoscedastic resampled residuals
            eps = resid[rng.integers(0, n, n)]
            ys = a_hat + kin * t + eps
            f = ols(t, ys)
            if abs(f["b"] / f["se_b"]) > 1.96:
                hit_slope += 1
            # (P2) primary statistic on a genuine null baseline + injected signal
            yb = baseY + residM[rng.permutation(n)] + kin * tc
            r = partial_spearman(yb, kT, MWL)
            if partial_p_analytic(r, n, 1) < 0.05:
                hit_rho += 1
        rows.append(dict(kappa_injected=kin,
                         power_slope_test=hit_slope / nsim,
                         power_primary_partial=hit_rho / nsim))

    def k80(key):
        xs = [r["kappa_injected"] for r in rows]
        ys = [r[key] for r in rows]
        for i in range(1, len(xs)):
            if ys[i] >= 0.80 and ys[i - 1] < 0.80:
                f = (0.80 - ys[i - 1]) / (ys[i] - ys[i - 1])
                return xs[i - 1] + f * (xs[i] - xs[i - 1])
        return None

    # analytic cross-check for the slope test
    sx = float(np.std(t, ddof=1)) * math.sqrt(n - 1)
    kappa80_analytic = 2.802 * s_res / sx   # (1.96+0.842) * sigma / sqrt(Sxx)

    return dict(nsim=nsim, curve=rows,
                kappa_at_80pct_power_slope=k80("power_slope_test"),
                kappa_at_80pct_power_primary=k80("power_primary_partial"),
                kappa80_analytic_slope=kappa80_analytic,
                type_I_rate_at_kappa_zero=dict(
                    slope_test=rows[0]["power_slope_test"],
                    primary_partial=rows[0]["power_primary_partial"],
                    note="both should sit near 0.05; this validates the two null baselines"),
                residual_sd_used=s_res,
                spread_of_t=dict(sd=float(np.std(t, ddof=1)),
                                 min=float(t.min()), max=float(t.max())))


# ----------------------------------------------------------------------------
# 13.  Error-model calibration (required before any chi^2-like statement)
# ----------------------------------------------------------------------------
def error_calibration(S, fit_free, nmc=4000, rng=None):
    rng = rng or RNG
    n = len(S)
    z = np.array([s["z"] for s in S])
    sM = np.array([sigma_ln(s["M_WL"], s["M_WL_ep"], s["M_WL_em"]) for s in S])
    sG = np.array([sigma_ln(s["M_gas"], s["M_gas_ep"], s["M_gas_em"]) for s in S])
    sL = np.array([sigma_ln(s["L_K_used"], s["L_K_ep"] or 0.15 * s["L_K_used"],
                            s["L_K_em"] or 0.15 * s["L_K_used"]) for s in S])
    sT = np.array([sigma_ln(s["kT"], s["kT_ep"], s["kT_em"]) for s in S])
    M0 = np.array([s["M_WL"] for s in S]); G0 = np.array([s["M_gas"] for s in S])
    L0 = np.array([s["L_K_used"] for s in S]); T0 = np.array([s["kT"] for s in S])

    Ys = np.empty((nmc, n)); ts = np.empty((nmc, n))
    for j in range(nmc):
        Mw = M0 * np.exp(rng.normal(0, sM, n))
        Mg = G0 * np.exp(rng.normal(0, sG, n))
        LK = L0 * np.exp(rng.normal(0, sL, n))
        kT = T0 * np.exp(rng.normal(0, sT, n))
        for i in range(n):
            Ys[j, i] = derive(Mw[i], z[i], Mg[i], LK[i], UPSILON_K_PRIMARY)["E"] ** 2 - 1.0
        ts[j] = t_of_kT(kT)
    sigY = Ys.std(axis=0, ddof=1)
    kap_mc = np.array([ols(ts[j], Ys[j])["b"] for j in range(nmc)])
    a_mc = np.array([ols(ts[j], Ys[j])["a"] for j in range(nmc)])

    obs_resid_sd = fit_free["residual_sd"]
    ratio = float(obs_resid_sd / np.median(sigY))
    calibrated = bool(0.7 < ratio < 1.4)
    if calibrated:
        verdict = ("Observed residual scatter is %.2f times the scatter predicted by the "
                   "propagated measurement errors alone, i.e. the per-point error model IS "
                   "approximately calibrated and the measurement errors account for the bulk "
                   "of the residual spread (leaving little room for intrinsic scatter). "
                   "Even so, all quoted intervals in this analysis are bootstrap intervals "
                   "and no chi^2 / AIC / BIC is quoted anywhere." % ratio)
    else:
        verdict = ("Observed residual scatter is %.2f times the measurement-only prediction, "
                   "so the per-point error model is NOT calibrated; no chi^2 / AIC / BIC is "
                   "quoted anywhere and all intervals are bootstrap intervals." % ratio)
    return dict(
        per_cluster_sigma_Y_from_measurement=list(map(float, sigY)),
        median_sigma_Y=float(np.median(sigY)),
        observed_residual_sd=obs_resid_sd,
        ratio_observed_scatter_to_measurement_scatter=ratio,
        calibrated=calibrated,
        verdict=verdict,
        kappa_measurement_error_only_sd=float(kap_mc.std(ddof=1)),
        intercept_measurement_error_only_sd=float(a_mc.std(ddof=1)),
        nmc=nmc,
    )


# ----------------------------------------------------------------------------
# 14.  Sensitivity (declared secondaries)
# ----------------------------------------------------------------------------
def sensitivity(clusters, primary_fit):
    out = {}

    # (c) Upsilon_K grid
    grid = []
    for up in PREREG["stellar_mass_to_light"]["secondary_sensitivity_grid"]:
        S = build(clusters, up)
        t = np.array([s["t"] for s in S]); Y = np.array([s["Y"] for s in S])
        kT = np.array([s["kT"] for s in S]); MWL = np.array([s["M_WL"] for s in S])
        f = ols(t, Y)
        r = partial_spearman(Y, kT, MWL)
        grid.append(dict(upsilon_K=up, n=len(S), intercept=f["a"], kappa=f["b"],
                         kappa_se=f["se_b"], rho_partial=r,
                         p_partial=partial_p_analytic(r, len(S), 1),
                         median_E=float(np.median([s["E"] for s in S])),
                         median_f_star=float(np.median([1 - s["f_gas_of_baryons"] for s in S]))))
    out["upsilon_K_grid"] = grid

    # (a) impute Abell2697
    S = build(clusters, UPSILON_K_PRIMARY, impute_missing_LK=True)
    t = np.array([s["t"] for s in S]); Y = np.array([s["Y"] for s in S])
    kT = np.array([s["kT"] for s in S]); MWL = np.array([s["M_WL"] for s in S])
    f = ols(t, Y); r = partial_spearman(Y, kT, MWL)
    out["impute_A2697_n41"] = dict(n=len(S), intercept=f["a"], kappa=f["b"],
                                   kappa_se=f["se_b"], rho_partial=r,
                                   p_partial=partial_p_analytic(r, len(S), 1))

    # (b) WL S/N >= 2.5
    S0 = build(clusters, UPSILON_K_PRIMARY)
    S = [s for s in S0 if s["snr_MWL"] >= 2.5]
    t = np.array([s["t"] for s in S]); Y = np.array([s["Y"] for s in S])
    kT = np.array([s["kT"] for s in S]); MWL = np.array([s["M_WL"] for s in S])
    f = ols(t, Y); r = partial_spearman(Y, kT, MWL)
    out["wl_snr_ge_2p5"] = dict(n=len(S), dropped=[s["name"] for s in S0 if s["snr_MWL"] < 2.5],
                                intercept=f["a"], kappa=f["b"], kappa_se=f["se_b"],
                                rho_partial=r, p_partial=partial_p_analytic(r, len(S), 1))

    # (d) drop 4 largest |residual|
    tt = np.array([s["t"] for s in S0]); YY = np.array([s["Y"] for s in S0])
    res = ols(tt, YY)["resid"]
    keep = np.argsort(np.abs(res))[:-4]
    S = [S0[i] for i in sorted(keep)]
    t = np.array([s["t"] for s in S]); Y = np.array([s["Y"] for s in S])
    kT = np.array([s["kT"] for s in S]); MWL = np.array([s["M_WL"] for s in S])
    f = ols(t, Y); r = partial_spearman(Y, kT, MWL)
    out["drop_4_largest_residuals"] = dict(
        n=len(S), dropped=[S0[i]["name"] for i in np.argsort(np.abs(res))[-4:]],
        intercept=f["a"], kappa=f["b"], kappa_se=f["se_b"],
        rho_partial=r, p_partial=partial_p_analytic(r, len(S), 1))

    # cosmology sensitivity (Om = 0.27 as in Okabe & Smith 2016)
    global OM, OL
    OM, OL = 0.27, 0.73
    S = build(clusters, UPSILON_K_PRIMARY)
    t = np.array([s["t"] for s in S]); Y = np.array([s["Y"] for s in S])
    f = ols(t, Y)
    out["cosmology_Om0p27"] = dict(intercept=f["a"], kappa=f["b"],
                                   median_E=float(np.median([s["E"] for s in S])))
    OM, OL = 0.3, 0.7
    return out


# ----------------------------------------------------------------------------
# 15.  Approximation-bias budget for the compressed form
# ----------------------------------------------------------------------------
def compressed_form_bias(S):
    """
    The compressed form  E^2 - 1 = kappa * 3kT/(mu m_p c^2)  is not what the field
    equation gives.  Two separate discrepancies, quantified here.

    (i) FORM.  For isothermal gas, Int 4 pi r'^2 P dr' = (kT/(mu m_p)) M_gas exactly,
        so the field prediction is
            M_eff/M_b - 1 = kappa * t * (M_gas/M_b),     i.e.  E - 1, not E^2 - 1,
        and with an extra factor f_gas = M_gas/M_b.
        Ratio of responses  (E^2-1)/[(E-1) f_gas] = (E+1)/f_gas  is the multiplicative
        factor by which kappa is inflated by using the compressed form.

    (ii) WEIGHTING.  kT_X_ce is a core-excised spectroscopic temperature in
        [0.15-1] r500; the integral needs the gas-mass-weighted temperature over the
        same sphere.  For observed cluster T(r) profiles the mass-weighted mean is
        ~0.80-0.95 of the core-excised spectroscopic value, so the compressed form
        overstates the pressure integral and therefore UNDERSTATES kappa by that factor.
    """
    E = np.array([s["E"] for s in S]); fg = np.array([s["f_gas_of_baryons"] for s in S])
    ratio = (E + 1.0) / fg
    Y = np.array([s["Y"] for s in S]); Ylin = np.array([s["Y_lin"] for s in S])
    t = np.array([s["t"] for s in S]); MWL = np.array([s["M_WL"] for s in S])
    kT = np.array([s["kT"] for s in S])
    r_comp = partial_spearman(Y, kT, MWL)
    r_exact = partial_spearman(Ylin, t * fg, MWL)
    return dict(
        primary_statistic_is_invariant_to_the_approximation=dict(
            rho_p_compressed_form=r_comp, p_compressed=partial_p_analytic(r_comp, len(S), 1),
            rho_p_isothermal_exact_form=r_exact, p_exact=partial_p_analytic(r_exact, len(S), 1),
            spearman_between_the_two_predictors=float(stats.spearmanr(kT, t * fg)[0]),
            spearman_between_the_two_responses=float(stats.spearmanr(Y, Ylin)[0]),
            note=("E^2-1 is a strictly monotone function of E-1, and kT ranks agree with "
                  "t*f_gas ranks at rho = 0.994, so the rank-based PRIMARY statistic is "
                  "essentially unchanged by switching from the compressed form to the "
                  "isothermal-exact one.  The approximation therefore cannot flip the "
                  "sign or the significance of the primary result; it can only rescale "
                  "the amplitude of kappa."),
        ),
        form_inflation_factor=dict(
            definition="kappa_compressed / kappa_isothermal-exact = (E+1)/f_gas, per cluster",
            median=float(np.median(ratio)), min=float(ratio.min()), max=float(ratio.max()),
            note="a factor ~3, and it is E-dependent, so it does not even act as a "
                 "constant rescaling: it varies systematically across the sample.",
        ),
        weighting_bias=dict(
            T_massweighted_over_T_ce_plausible_range=[0.80, 0.95],
            implied_kappa_multiplier_range=[1.0 / 0.95, 1.0 / 0.80],
            note="kappa from the compressed form must be divided by <T>_mw/T_ce to "
                 "recover the field-equation kappa; this is a 5-25 per cent effect, "
                 "small next to the form factor above.",
        ),
        isothermality_check=dict(
            note="LoCuSS publishes ONE temperature per cluster.  There is no P(r) and no "
                 "T(r) in this dataset, so the isothermal + proportional-density-profile "
                 "assumption cannot be tested here at all.  The exact test needs "
                 "P(r) on radii matched to the lensing aperture, which this sample does "
                 "not provide.",
        ),
        f_gas_of_baryons=dict(median=float(np.median(fg)), min=float(fg.min()), max=float(fg.max())),
    )


# ----------------------------------------------------------------------------
# 15b.  Extra effect sizes
# ----------------------------------------------------------------------------
def extras(S, nboot=10000, rng=None):
    rng = rng or RNG
    n = len(S)
    t = np.array([s["t"] for s in S]); Y = np.array([s["Y"] for s in S])
    kT = np.array([s["kT"] for s in S]); MWL = np.array([s["M_WL"] for s in S])
    E = np.array([s["E"] for s in S])
    kimp = np.array([s["kappa_implied_compressed"] for s in S])

    # log-log multiple regression: ln E = c + p ln kT + q ln M_WL
    X = np.column_stack([np.ones(n), np.log(kT), np.log(MWL)])
    beta, *_ = np.linalg.lstsq(X, np.log(E), rcond=None)
    bs = []
    for _ in range(nboot):
        i = rng.integers(0, n, n)
        b, *_ = np.linalg.lstsq(X[i], np.log(E)[i], rcond=None)
        bs.append(b)
    bs = np.array(bs)
    # the pressure model, in the compressed form, needs d lnE / d ln kT > 0.
    # E^2 = 1 + kappa t  =>  dlnE/dlnkT = (kappa t)/(2 E^2) > 0 always.
    Epred_slope = float(np.mean(KAPPA_XCOP * t / (2.0 * (1.0 + KAPPA_XCOP * t))))

    # per-cluster implied kappa: the model forbids any per-cluster coefficient
    ba = []
    for _ in range(nboot):
        i = rng.integers(0, n, n)
        ba.append(ols(t[i], Y[i])["a"])
    return dict(
        loglog_regression=dict(
            model="ln E = c + p ln kT + q ln M_WL",
            c=float(beta[0]), p_kT=float(beta[1]), q_MWL=float(beta[2]),
            p_kT_boot95=[float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5))],
            q_MWL_boot95=[float(np.percentile(bs[:, 2], 2.5)), float(np.percentile(bs[:, 2], 97.5))],
            p_kT_required_by_model_at_kappa_XCOP=Epred_slope,
        ),
        intercept_boot99=[float(np.percentile(ba, 0.5)), float(np.percentile(ba, 99.5))],
        implied_kappa_per_cluster=dict(
            median=float(np.median(kimp)), min=float(kimp.min()), max=float(kimp.max()),
            max_over_min=float(kimp.max() / kimp.min()),
            sd_of_ln=float(np.std(np.log(kimp), ddof=1)),
            spearman_with_kT=list(map(float, stats.spearmanr(kimp, kT))),
            spearman_with_MWL=list(map(float, stats.spearmanr(kimp, MWL))),
            note=("The model forbids a per-cluster coefficient.  If a single universal kappa "
                  "held, every cluster's implied kappa would agree within its errors."),
        ),
        raw_ratio_summary=dict(
            median_E=float(np.median(E)),
            median_Mb_over_MWL=float(np.median([s["M_b"] / s["M_WL"] for s in S])),
            cosmic_baryon_fraction_for_reference=0.157,
        ),
    )


# ----------------------------------------------------------------------------
# 16.  Main
# ----------------------------------------------------------------------------
def main():
    clusters = load()
    S = build(clusters, UPSILON_K_PRIMARY)
    assert len(S) == 40, "primary sample should be 40 clusters (Abell2697 has no L_K)"

    print("primary sample n = %d" % len(S))
    fit, arr = headline(S, nboot=10000)
    print("  intercept = %+.3f  (boot95 %.3f .. %.3f)"
          % (fit["fit_free_intercept"]["intercept"],
             *fit["fit_free_intercept"]["intercept_boot95"]))
    print("  kappa     = %.3e (boot95 %.3e .. %.3e)"
          % (fit["fit_free_intercept"]["kappa"],
             *fit["fit_free_intercept"]["kappa_boot95"]))
    print("  primary rho_p = %+.3f  p_analytic = %.4f"
          % (fit["primary_statistic"]["value"], fit["primary_statistic"]["p_analytic"]))

    xcop = compare_xcop(fit["fit_free_intercept"], fit["fit_zero_intercept"])
    print("  kappa/kappa_XCOP (free) = %.3f" % xcop["free_intercept"]["ratio_kappa_over_XCOP"])

    print("running leave-one-out ...")
    L = loo(S)
    print("running error-model calibration ...")
    cal = error_calibration(S, fit["fit_free_intercept"], nmc=3000)
    print("  observed/measurement scatter ratio = %.2f" %
          cal["ratio_observed_scatter_to_measurement_scatter"])
    print("running aperture analytics + Monte Carlo ...")
    ap_a = aperture_analytic(S)
    ap_m = aperture_montecarlo(S, nsim=1500)
    print("running permutation null (20000) ...")
    perm = permutation_null(S, ndraw=20000)
    print("  permutation p (two-sided, rho_p) = %.4f" % perm["p_two_sided_rho_partial"])
    print("running power curves ...")
    pw = power_curves(S, fit["fit_free_intercept"], nsim=1500)
    print("  kappa at 80%% power (slope) = %s" % pw["kappa_at_80pct_power_slope"])
    print("running sensitivity ...")
    sens = sensitivity(clusters, fit)
    bias = compressed_form_bias(S)
    ex = extras(S, nboot=10000)

    percluster = []
    for s in S:
        percluster.append({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                           for k, v in s.items()
                           if k not in ("L_K_imputed",)} | {"L_K_imputed": s["L_K_imputed"]})

    results = dict(
        preregistration=PREREG,
        constants=dict(G_SI=G_SI, MSUN=MSUN, MPC=MPC, c=C_LIGHT, m_p=M_P, keV_J=KEV_J,
                       a0=A0, mu=MU_MOL, H0_km_s_Mpc=70.0, Omega_m=0.3, Omega_L=0.7,
                       upsilon_K=UPSILON_K_PRIMARY, kappa_XCOP=KAPPA_XCOP,
                       rho_c0_kg_m3=RHO_C0),
        conversion_chain=derive.__doc__,
        n_primary=len(S),
        per_cluster=percluster,
        check1_fit_with_free_intercept=fit,
        check2_compare_to_XCOP=xcop,
        check3_partial_correlation=fit["primary_statistic"],
        check4_leave_one_out=L,
        check5_aperture_analytic=ap_a,
        check5_aperture_montecarlo=ap_m,
        check6_permutation_null=perm,
        check7_power=pw,
        error_model_calibration=cal,
        sensitivity=sens,
        compressed_form_approximation_bias=bias,
        extra_effect_sizes=ex,
    )

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, sort_keys=False, default=float)
    print("wrote %s (%d bytes)" % (OUT_JSON, os.path.getsize(OUT_JSON)))
    return results


if __name__ == "__main__":
    main()
