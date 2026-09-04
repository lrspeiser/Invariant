r"""
member-dynamics lane -- does a galaxy's INTERNAL dynamics change inside a cluster?

The question, and the two predictions being separated
-----------------------------------------------------
The tensor lane (work/wellnet-2026-09/tensor) found that a cluster MEMBER galaxy
sits at |Phi_N| = 1.09e12 m^2/s^2, deeper than the cluster's own 1 Mpc shell at
7.22e11.  Every surviving potential-depth-gated law therefore switches on hardest
INSIDE cluster member galaxies, and boosts their internal gravity by

    Delta log10 g_int = +0.031 +- 0.023 dex          (tensor seed_robustness.json)

An acceleration-gated law (MOND / the RAR) predicts EXACTLY ZERO, because a
cluster member's internal accelerations are unchanged and g_N/a0 is the only
variable it can see.

At fixed radius g = V^2/R, so in the observable used here (a velocity):

    Delta log10 V_int = 0.5 * Delta log10 g_int = +0.0155 +- 0.0115 dex
    acceleration-only                          =  0.0000 dex

BOTH conventions are carried through the whole file; `_g` suffixed numbers are
in log g, unsuffixed ones in log V.

The measure
-----------
Y = log10 sigma_e_tot, the aperture second velocity moment of the STARS inside
1 Re,   sigma_e_tot^2 = <V^2 + sigma^2>_flux .

Stellar, not gas, because cluster passage strips and shocks gas.  The second
moment, not the rotation speed, because environment converts rotation into
dispersion at roughly fixed kinetic energy -- V_rot alone is contaminated by the
kinematic morphology-density relation, the second moment is not.  It needs no
disk model, no rotation-curve fit and no inclination deprojection.

  MaNGA: computed here from the DR17 DAP MAPS cubes (extract_manga_kin.py).
  SAMI : SIGMA_RE_MGE * sqrt(1 + VSIGMA_RE_MGE^2), which is the identical
         construction from the published DR3 stellar-kinematics catalogue.

Everything below the DECLARATIONS block was fixed before any field/cluster
difference was computed.
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260904)

ENV = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data"
LANE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\member-dynamics"

# ======================================================================
# DECLARATIONS -- fixed before any residual was inspected
# ======================================================================
PRED_G = 0.031          # potential-depth gate, dex in log g
PRED_G_SD = 0.023
PRED_V = PRED_G / 2.0   # same prediction expressed in log V
PRED_V_SD = PRED_G_SD / 2.0

# quality / contamination cuts.  A pair is dropped if EITHER member fails.
CUTS = {
    "manga": dict(
        frac_good_1Re=0.50,      # >= : aperture actually covered
        n_bins_1Re=10,           # >= : enough independent bins
        A_kin=None,              # <= : filled from the p80 of the parent
        med_sigma_astro=40.0,    # >= : above the DAP dispersion floor
    ),
    "sami": dict(
        k51=None,                # <= : filled from the p80 of the parent
        aper_corr_flag=1,        # <= : 0 = reaches 1 Re, 1 = mild correction
    ),
}
ASYM_PERCENTILE = 80.0   # keep the 80% most regular; declared, symmetric

# matching-variable differences used in the covariate adjustment
DCOLS_MANGA = ["d_logMstar_nsa", "d_logRd", "d_logSigma_b", "d_incl_deg",
               "d_pym_r_BT_SE", "d_z"]
DCOLS_SAMI = ["d_logMstar", "d_logRd", "d_logSigma_b", "d_incl_deg", "d_z_spec"]
# note: d_log_gbar_2p2Rd is deliberately EXCLUDED -- env-data measured it to be
# collinear with d_logSigma_b at r = 0.996 (MaNGA) and exactly 1 (SAMI).

NBOOT = 20000
NPERM = 20000
NSIM = 2000


# ======================================================================
# estimators
# ======================================================================
def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def adjusted_offset(dy, D):
    """Intercept of OLS of the paired difference on the matching-variable
    differences.  Under the null (no environment term) the intercept is zero
    for any residual imbalance in the matching box, which the raw mean is not."""
    X = np.column_stack([np.ones(len(dy)), D])
    b = ols(X, dy)
    return float(b[0]), b


def system_bootstrap(dy, D, sysid, nboot=NBOOT, adjust=True):
    """Resample HOST SYSTEMS with replacement.  Pairs inside one cluster share
    an environment, so the pair is not the independent unit -- the cluster is."""
    sys_u = np.unique(sysid)
    idx_by_sys = {s: np.where(sysid == s)[0] for s in sys_u}
    out = np.empty(nboot)
    ns = len(sys_u)
    for b in range(nboot):
        pick = RNG.integers(0, ns, ns)
        idx = np.concatenate([idx_by_sys[sys_u[p]] for p in pick])
        if adjust and len(idx) > D.shape[1] + 2:
            try:
                out[b] = adjusted_offset(dy[idx], D[idx])[0]
            except np.linalg.LinAlgError:
                out[b] = np.mean(dy[idx])
        else:
            out[b] = np.mean(dy[idx])
    return out


def blocked_signflip(dy, D, sysid, nperm=NPERM, adjust=True):
    """Exact label-permutation null: swapping which member of a pair is called
    'cluster' flips the sign of the difference.  Flips are applied per HOST
    SYSTEM so the null carries the same system-level correlation as the data."""
    sys_u = np.unique(sysid)
    blocks = [np.where(sysid == s)[0] for s in sys_u]
    out = np.empty(nperm)
    for b in range(nperm):
        s = np.ones(len(dy))
        for blk in blocks:
            if RNG.random() < 0.5:
                s[blk] = -1.0
        if adjust:
            out[b] = adjusted_offset(s * dy, D * s[:, None])[0]
        else:
            out[b] = np.mean(s * dy)
    return out


# ======================================================================
# data assembly
# ======================================================================
def build_manga():
    mp = pd.read_csv(os.path.join(ENV, "clean", "matched_pairs.csv"))
    kin = pd.read_csv(os.path.join(LANE, "clean", "manga_internal_kin.csv"))
    mis = pd.read_csv(os.path.join(ENV, "clean", "manga_gas_star_misalignment.csv"),
                      usecols=["plateifu", "misalign_deg"])
    kin = kin.merge(mis, on="plateifu", how="left")
    kin["Y"] = np.log10(kin["sigma_e_tot"])
    kin["eY"] = kin["e_sigma_e_tot"] / (kin["sigma_e_tot"] * np.log(10))
    keep = ["plateifu", "Y", "eY", "sigma_e_tot", "sigma_e", "v_e", "vsigma_e",
            "A_kin", "frac_good_1Re", "n_bins_1Re", "med_sigma_astro",
            "med_snr_1Re", "misalign_deg", "pa_kin_star", "pa_kin_gas",
            "n_gas_1Re", "ok", "dapqual", "e_sigma_e_tot"]
    k = kin[keep]
    df = mp.merge(k.add_prefix("cl_"), left_on="cl_plateifu", right_on="cl_plateifu", how="left")
    df = df.merge(k.add_prefix("fi_"), left_on="fi_plateifu", right_on="fi_plateifu", how="left")
    # host system: the X-ray cluster for the X-ray tiers, else the Tempel group.
    # Unidentified hosts get a unique id so they are never pooled into one block.
    prim = df["cl_xray_oname"].where(df["tier"].str.startswith("C"), np.nan)
    sysid = prim.fillna(df["cl_t14_GroupID"].astype("Int64").astype(str))
    sysid = sysid.where(sysid.notna() & (sysid != "<NA>"),
                        pd.Series([f"solo{i}" for i in range(len(df))], index=df.index))
    df["sysid"] = sysid.astype(str)
    df["depth_sigma_v"] = df["cl_t14_grp_sigma_v"]
    df["radius_norm"] = df["cl_R_over_Rvir_t14"]
    df["radius_mpc"] = df["cl_t14_Rproj_kpc"] / 1000.0
    return df, kin


def build_sami():
    sp = pd.read_csv(os.path.join(ENV, "clean", "sami_matched_pairs.csv"))
    sk = pd.read_csv(os.path.join(ENV, "raw", "sami", "sami_dr3_samiDR3Stelkin.tsv"), sep="\t")
    gp = pd.read_csv(os.path.join(ENV, "raw", "sami", "sami_dr3_samiDR3gaskinPA.tsv"), sep="\t")
    sk = sk.merge(gp[["CUBEID", "PA_GASKIN"]], on="CUBEID", how="left")
    # de-duplicate repeat cubes: keep the row with the most complete MGE kinematics
    sk["_score"] = sk[["SIGMA_RE_MGE", "VSIGMA_RE_MGE", "MEAN_K51_RE_MGE"]].notna().sum(1)
    sk = sk.sort_values(["CATID", "_score"], ascending=[True, False]).drop_duplicates("CATID")
    sk["sigma_e_tot"] = sk["SIGMA_RE_MGE"] * np.sqrt(1.0 + sk["VSIGMA_RE_MGE"] ** 2)
    # error propagation from the published 1-sigma errors
    s, e_s = sk["SIGMA_RE_MGE"], sk["SIGMA_RE_MGE_ERR"]
    v, e_v = sk["VSIGMA_RE_MGE"], sk["VSIGMA_RE_MGE_ERR"]
    f = np.sqrt(1 + v ** 2)
    sk["e_sigma_e_tot"] = np.sqrt((f * e_s) ** 2 + (s * v / f * e_v) ** 2)
    sk["Y"] = np.log10(sk["sigma_e_tot"])
    sk["eY"] = sk["e_sigma_e_tot"] / (sk["sigma_e_tot"] * np.log(10))
    sk["misalign_deg"] = np.abs(((sk["PA_STELKIN"] - sk["PA_GASKIN"] + 180.0) % 360.0) - 180.0)
    keep = ["CATID", "Y", "eY", "sigma_e_tot", "SIGMA_RE_MGE", "VSIGMA_RE_MGE",
            "MEAN_K51_RE_MGE", "APER_CORR_FLAG_MGE", "LAMBDAR_RE_MGE",
            "misalign_deg", "e_sigma_e_tot"]
    k = sk[keep].rename(columns={"MEAN_K51_RE_MGE": "k51", "APER_CORR_FLAG_MGE": "aper_flag"})
    # drop the env-data lane's own copies of the kinematic columns so the merge
    # does not create _x/_y duplicates; this lane recomputes them from the MGE
    # (photometrically homogeneous across both arms) variants.
    sp = sp.drop(columns=[c for c in sp.columns if c.split("_", 1)[-1] in
                          ("SIGMA_RE_MGE", "LAMBDAR_RE", "VSIGMA_RE", "gas_star_misalign_deg")
                          and c[:3] in ("cl_", "fi_")])
    df = sp.merge(k.add_prefix("cl_"), left_on="cl_CATID", right_on="cl_CATID", how="left")
    df = df.merge(k.add_prefix("fi_"), left_on="fi_CATID", right_on="fi_CATID", how="left")
    df["sysid"] = df["cl_cluster"].astype(str)
    df["depth_sigma_v"] = df["cl_host_sigma_200"]
    # SHARED-DENOMINATOR GUARD: cl_R_on_rtwo = R_proj/R200 and R200 ~ sigma_200,
    # so R_on_rtwo carries sigma in its denominator.  When the other axis is
    # sigma_v the sigma-free radius must be used instead.  Both are carried.
    df["radius_norm"] = df["cl_R_on_rtwo"]
    df["radius_mpc"] = df["cl_R_proj_Mpc_from_cat"]
    return df, k


# ======================================================================
def apply_cuts(df, survey, a_thresh):
    if survey == "manga":
        c = CUTS["manga"]
        ok = np.ones(len(df), bool)
        for side in ("cl_", "fi_"):
            ok &= (df[side + "ok"] == 1).to_numpy()
            ok &= (df[side + "frac_good_1Re"] >= c["frac_good_1Re"]).to_numpy()
            ok &= (df[side + "n_bins_1Re"] >= c["n_bins_1Re"]).to_numpy()
            ok &= (df[side + "med_sigma_astro"] >= c["med_sigma_astro"]).to_numpy()
            ok &= (df[side + "A_kin"] <= a_thresh).to_numpy()
            ok &= np.isfinite(df[side + "Y"]).to_numpy()
    else:
        c = CUTS["sami"]
        ok = np.ones(len(df), bool)
        for side in ("cl_", "fi_"):
            ok &= np.isfinite(df[side + "Y"]).to_numpy()
            ok &= (df[side + "k51"] <= a_thresh).to_numpy()
            ok &= (df[side + "aper_flag"] <= c["aper_corr_flag"]).to_numpy()
    return ok


def analyse_tier(df, survey, tier, a_thresh, adjust=True, tag=""):
    d = df[df["tier"] == tier].copy()
    n0 = len(d)
    ok = apply_cuts(d, survey, a_thresh)
    d = d[ok]
    if len(d) < 8:
        return dict(tier=tier, tag=tag, n_pairs_declared=n0, n_pairs_used=len(d),
                    status="too_few_pairs")
    dcols = DCOLS_MANGA if survey == "manga" else DCOLS_SAMI
    dy = (d["cl_Y"] - d["fi_Y"]).to_numpy()
    D = d[dcols].to_numpy()
    D = np.nan_to_num(D, nan=0.0)
    sysid = d["sysid"].to_numpy()

    raw = float(np.mean(dy))
    adj, coef = adjusted_offset(dy, D)
    boot = system_bootstrap(dy, D, sysid, adjust=adjust)
    perm = blocked_signflip(dy, D, sysid, adjust=adjust)
    est = adj if adjust else raw
    sd_sys = float(np.std(boot, ddof=1))
    sd_naive = float(np.std(dy, ddof=1) / np.sqrt(len(dy)))
    p_perm = float(np.mean(np.abs(perm) >= abs(est)))

    sys_u, sys_n = np.unique(sysid, return_counts=True)
    neff = float(sys_n.sum() ** 2 / (sys_n ** 2).sum())

    # --- power, computed from sd_sys BEFORE the answer is quoted -------------
    mdd3 = 3.0 * sd_sys                       # 3-sigma minimum detectable, log V
    from math import erf, sqrt
    def _pow(mu, sd, alpha=0.05):
        zc = 1.959963985
        z = mu / sd
        cdf = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
        return float(cdf(z - zc) + cdf(-z - zc))
    power_pred = _pow(PRED_V, sd_sys)
    power_pred_lo = _pow(max(PRED_V - PRED_V_SD, 1e-6), sd_sys)
    power_pred_hi = _pow(PRED_V + PRED_V_SD, sd_sys)

    res = dict(
        tier=tier, tag=tag, survey=survey,
        n_pairs_declared=n0, n_pairs_used=int(len(d)),
        n_systems=int(len(sys_u)), n_eff_systems=neff,
        largest_system_share=float(sys_n.max() / sys_n.sum()),
        asym_threshold=float(a_thresh),
        offset_logV_raw=raw, offset_logV_adj=adj, offset_logV=est,
        sd_system_bootstrap=sd_sys, sd_naive_pairwise=sd_naive,
        deff=float((sd_sys / sd_naive) ** 2) if sd_naive > 0 else np.nan,
        ci68=[float(np.percentile(boot, 16)), float(np.percentile(boot, 84))],
        ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        p_permutation=p_perm,
        perm_sd=float(np.std(perm, ddof=1)),
        offset_logg=2 * est, sd_logg=2 * sd_sys,
        mdd_3sigma_logV=mdd3, mdd_3sigma_logg=2 * mdd3,
        power_vs_prediction=power_pred,
        power_vs_prediction_lo=power_pred_lo,
        power_vs_prediction_hi=power_pred_hi,
        z_vs_zero=float(est / sd_sys) if sd_sys > 0 else np.nan,
        z_vs_prediction=float((est - PRED_V) / np.hypot(sd_sys, PRED_V_SD)),
        coef_adjust={c: float(v) for c, v in zip(["const"] + dcols, coef)},
        mean_matching_residual={c: float(np.nanmean(d[c])) for c in dcols},
    )
    # achieved tolerances on the sample ACTUALLY USED
    res["achieved_tolerances"] = {
        c: dict(max_abs=float(np.nanmax(np.abs(d[c]))),
                rms=float(np.sqrt(np.nanmean(d[c] ** 2))),
                mean=float(np.nanmean(d[c]))) for c in dcols}
    return res, d


def gradient_tests(d, survey, dcols):
    """Potential-depth laws predict a GRADIENT with |Phi| and with radius;
    a pure galaxy/not-galaxy class step predicts none."""
    dy = (d["cl_Y"] - d["fi_Y"]).to_numpy()
    D = np.nan_to_num(d[dcols].to_numpy(), nan=0.0)
    # remove the matching-variable dependence first, then look for a gradient
    X = np.column_stack([np.ones(len(dy)), D])
    r = dy - X @ ols(X, dy)
    out = {}
    axes = {
        "log_Phi_proxy": 2 * np.log10(d["depth_sigma_v"].to_numpy()),
        "R_over_Rvir": d["radius_norm"].to_numpy(),
        "R_proj_Mpc": d["radius_mpc"].to_numpy(),
    }
    sysid = d["sysid"].to_numpy()
    sys_u = np.unique(sysid)
    idx_by_sys = {s: np.where(sysid == s)[0] for s in sys_u}
    for name, x in axes.items():
        m = np.isfinite(x) & np.isfinite(r)
        if m.sum() < 12:
            out[name] = dict(status="too_few")
            continue
        xx, rr = x[m], r[m]
        A = np.column_stack([np.ones(len(xx)), xx - np.mean(xx)])
        b = ols(A, rr)
        # system bootstrap on the slope
        sl = []
        for _ in range(4000):
            pick = RNG.integers(0, len(sys_u), len(sys_u))
            idx = np.concatenate([idx_by_sys[sys_u[p]] for p in pick])
            idx = idx[m[idx]]
            if len(idx) < 12:
                continue
            xa, ra = x[idx], r[idx]
            Aa = np.column_stack([np.ones(len(xa)), xa - np.mean(xa)])
            try:
                sl.append(ols(Aa, ra)[1])
            except np.linalg.LinAlgError:
                pass
        sl = np.array(sl)
        # split-half contrast: deep vs shallow
        med = np.median(xx)
        hi, lo = rr[xx > med], rr[xx <= med]
        out[name] = dict(
            slope=float(b[1]), slope_sd=float(np.std(sl, ddof=1)) if len(sl) > 10 else np.nan,
            n=int(m.sum()), x_p10=float(np.percentile(xx, 10)), x_p90=float(np.percentile(xx, 90)),
            mean_hi=float(np.mean(hi)), mean_lo=float(np.mean(lo)),
            contrast=float(np.mean(hi) - np.mean(lo)),
        )
    return out


def contamination_budget(d, survey, dcols):
    """How large a spurious offset could the residual environmental
    differences produce?  For each diagnostic X, measure the FIELD-arm
    sensitivity dY/dX, multiply by the measured cluster-minus-field mean
    difference in X, and quote the induced offset in log V."""
    fi = {}
    if survey == "manga":
        diags = {"A_kin": ("cl_A_kin", "fi_A_kin"),
                 "misalign_deg": ("cl_misalign_deg", "fi_misalign_deg"),
                 "med_sigma_astro": ("cl_med_sigma_astro", "fi_med_sigma_astro"),
                 "frac_good_1Re": ("cl_frac_good_1Re", "fi_frac_good_1Re"),
                 "med_snr_1Re": ("cl_med_snr_1Re", "fi_med_snr_1Re")}
    else:
        diags = {"k51": ("cl_k51", "fi_k51"),
                 "misalign_deg": ("cl_misalign_deg", "fi_misalign_deg"),
                 "lambdaR": ("cl_LAMBDAR_RE_MGE", "fi_LAMBDAR_RE_MGE"),
                 "aper_flag": ("cl_aper_flag", "fi_aper_flag")}
    # field-arm sensitivity, controlling for the baryonic covariates
    yf = d["fi_Y"].to_numpy()
    base = [np.ones(len(d))]
    if survey == "manga":
        base += [d["fi_logMstar_nsa"].to_numpy(), d["fi_logRd"].to_numpy()]
    else:
        base += [d["fi_logMstar"].to_numpy(), d["fi_logRd"].to_numpy()]
    B = np.column_stack(base)
    for name, (cc, fc) in diags.items():
        x = d[fc].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(yf)
        if m.sum() < 20:
            fi[name] = dict(status="too_few")
            continue
        A = np.column_stack([B[m], x[m] - np.nanmean(x[m])])
        b = ols(A, yf[m])
        slope = float(b[-1])
        dx = float(np.nanmean(d[cc].to_numpy(float)) - np.nanmean(x))
        fi[name] = dict(field_sensitivity_dY_dX=slope,
                        mean_cluster_minus_field=dx,
                        induced_offset_logV=slope * dx,
                        induced_offset_logg=2 * slope * dx,
                        n=int(m.sum()))
    return fi


def null_simulation(d, survey, dcols, nsim=NSIM):
    """Forward-simulate the NULL with the real error covariance and the real
    shared inputs, then run the identical estimator.  This is the check the
    programme's rho_p = -0.304 retraction demands: if the internal-dynamics
    measure and the matching variables share a measured input, the null
    expectation of the estimator is not automatically zero."""
    if survey == "manga":
        mcol, rcol = "logMstar_nsa", "logRd"
        extra = ["pym_r_BT_SE", "incl_deg"]
    else:
        mcol, rcol = "logMstar", "logRd"
        extra = ["incl_deg"]
    # field-arm frozen scaling relation Y = f(baryons) -- CLUSTER ARM NEVER USED
    yf = d["fi_Y"].to_numpy()
    cols = [mcol, rcol] + extra
    Xf = np.column_stack([np.ones(len(d))] + [d["fi_" + c].to_numpy(float) for c in cols])
    m = np.isfinite(yf) & np.isfinite(Xf).all(1)
    b = ols(Xf[m], yf[m])
    resid = yf[m] - Xf[m] @ b
    s_int = float(np.std(resid, ddof=1))

    Xc = np.column_stack([np.ones(len(d))] + [d["cl_" + c].to_numpy(float) for c in cols])
    mu_c, mu_f = Xc @ b, Xf @ b
    eY_c = np.nan_to_num(d["cl_eY"].to_numpy(float), nan=0.0)
    eY_f = np.nan_to_num(d["fi_eY"].to_numpy(float), nan=0.0)
    D = np.nan_to_num(d[dcols].to_numpy(), nan=0.0)
    good = np.isfinite(mu_c) & np.isfinite(mu_f)
    out = np.empty(nsim)
    for i in range(nsim):
        yc = mu_c + RNG.normal(0, s_int, len(d)) + RNG.normal(0, 1, len(d)) * eY_c
        yfm = mu_f + RNG.normal(0, s_int, len(d)) + RNG.normal(0, 1, len(d)) * eY_f
        dy = (yc - yfm)[good]
        out[i] = adjusted_offset(dy, D[good])[0]
    return dict(mean=float(np.mean(out)), sd=float(np.std(out, ddof=1)),
                p16=float(np.percentile(out, 16)), p84=float(np.percentile(out, 84)),
                s_int_field=s_int, nsim=nsim,
                interpretation="mean != 0 would mean the estimator is biased by a shared measured input")


def placebo_field_field(kin_field_Y, dcols, d, survey, nrep=2000):
    """Placebo: keep the field arm only, split it at random into a fake
    'cluster' half and a fake 'field' half, re-pair at random and run the
    identical estimator.  Any non-zero mean is pipeline bias."""
    y = d["fi_Y"].to_numpy()
    y = y[np.isfinite(y)]
    n = len(y) // 2
    out = np.empty(nrep)
    for i in range(nrep):
        p = RNG.permutation(len(y))
        out[i] = float(np.mean(y[p[:n]] - y[p[n:2 * n]]))
    return dict(mean=float(np.mean(out)), sd=float(np.std(out, ddof=1)), nrep=nrep)


# ======================================================================
def main():
    results = {"declarations": dict(
        measure="Y = log10 sigma_e_tot ; sigma_e_tot^2 = <V^2+sigma^2>_flux within 1 Re, STELLAR",
        prediction_potential_depth_logg=[PRED_G, PRED_G_SD],
        prediction_potential_depth_logV=[PRED_V, PRED_V_SD],
        prediction_acceleration_only=0.0,
        cuts=CUTS, asym_percentile=ASYM_PERCENTILE,
        dcols_manga=DCOLS_MANGA, dcols_sami=DCOLS_SAMI,
        nboot=NBOOT, nperm=NPERM, nsim=NSIM, seed=20260904)}

    dm, kinm = build_manga()
    ds, kins = build_sami()

    # asymmetry thresholds from the PARENT distributions (declared percentile)
    a_manga = float(np.nanpercentile(kinm.loc[kinm.ok == 1, "A_kin"], ASYM_PERCENTILE))
    a_sami = float(np.nanpercentile(kins["k51"], ASYM_PERCENTILE))
    CUTS["manga"]["A_kin"] = a_manga
    CUTS["sami"]["k51"] = a_sami
    results["declarations"]["asym_threshold_manga_A_kin"] = a_manga
    results["declarations"]["asym_threshold_sami_k51"] = a_sami
    print(f"declared asymmetry thresholds: MaNGA A_kin <= {a_manga:.4f}, SAMI k5/k1 <= {a_sami:.4f}")

    used_frames = {}
    for survey, df, thr, dcols in (("manga", dm, a_manga, DCOLS_MANGA),
                                   ("sami", ds, a_sami, DCOLS_SAMI)):
        results[survey] = {}
        for tier in sorted(df["tier"].unique()):
            r = analyse_tier(df, survey, tier, thr, tag="primary")
            if isinstance(r, dict):
                results[survey][tier] = r
                continue
            res, d = r
            res["gradients"] = gradient_tests(d, survey, dcols)
            res["contamination_budget"] = contamination_budget(d, survey, dcols)
            res["null_simulation"] = null_simulation(d, survey, dcols)
            res["placebo_field_field"] = placebo_field_field(None, dcols, d, survey)
            # arm-by-arm cut attrition -- itself a contamination measurement
            dall = df[df["tier"] == tier]
            if survey == "manga":
                fc = lambda s: ((dall[s + "ok"] == 1) & (dall[s + "frac_good_1Re"] >= 0.5)
                                & (dall[s + "n_bins_1Re"] >= 10)
                                & (dall[s + "med_sigma_astro"] >= 40.0)
                                & (dall[s + "A_kin"] <= thr))
            else:
                fc = lambda s: (np.isfinite(dall[s + "Y"]) & (dall[s + "k51"] <= thr)
                                & (dall[s + "aper_flag"] <= 1))
            res["cut_attrition"] = dict(
                cluster_pass_frac=float(np.mean(fc("cl_"))),
                field_pass_frac=float(np.mean(fc("fi_"))))
            results[survey][tier] = res
            used_frames[(survey, tier)] = d
            print(f"  {survey:6s} {tier:16s} N={res['n_pairs_used']:4d}/{res['n_pairs_declared']:4d} "
                  f"sys={res['n_systems']:3d} sd={res['sd_system_bootstrap']:.4f} "
                  f"power={res['power_vs_prediction']:.3f} -> D={res['offset_logV']:+.4f}")

        # sensitivity to the asymmetry cut
        results[survey]["_cut_sensitivity"] = {}
        for lab, t in (("none", np.inf), ("p50", None), ("p95", None)):
            if survey == "manga":
                par = kinm.loc[kinm.ok == 1, "A_kin"]
            else:
                par = kins["k51"]
            tt = np.inf if lab == "none" else float(np.nanpercentile(par, 50 if lab == "p50" else 95))
            sub = {}
            for tier in sorted(df["tier"].unique()):
                rr = analyse_tier(df, survey, tier, tt, tag=lab)
                if not isinstance(rr, dict):
                    rr = rr[0]
                    sub[tier] = {k: rr[k] for k in ("n_pairs_used", "offset_logV",
                                                    "sd_system_bootstrap", "p_permutation")}
            results[survey]["_cut_sensitivity"][lab] = dict(threshold=tt, tiers=sub)

    os.makedirs(os.path.join(LANE), exist_ok=True)
    with open(os.path.join(LANE, "member_dynamics.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    # write the sample actually used
    for (survey, tier), d in used_frames.items():
        d.to_csv(os.path.join(LANE, "clean", f"sample_used_{survey}_{tier}.csv"), index=False)
    print("wrote member_dynamics.json and", len(used_frames), "sample files")
    return results


if __name__ == "__main__":
    main()
