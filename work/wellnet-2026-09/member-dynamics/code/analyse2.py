r"""
member-dynamics lane -- main analysis.

Adds to analyse.py:
  * the THREE competing predictions, not two;
  * aperture robustness (1 Re, 0.5 Re, fixed 3 kpc, fixed 5 kpc) -- a fixed
    physical aperture is immune to any Re mismatch between the two arms;
  * a quantitative environmental-contamination budget in the units of the
    signal;
  * the combined MaNGA + SAMI estimate;
  * decomposition of the offset into its dispersion and rotation parts.

THE THREE PREDICTIONS
---------------------
H1  potential-depth gate (tensor lane, all surviving well-network points):
        Delta log g_int = +0.031 +- 0.023  ->  Delta log V = +0.0155 +- 0.0115
H2  algebraic RAR / acceleration-only, no external-field effect:
        Delta log V = 0 EXACTLY.  A cluster member's internal g_N/a0 is
        unchanged, and g_N/a0 is the only variable the law can see.
H3  MOND with the external-field effect (AQUAL/QUMOND).  This is NOT the same
    as H2: a real MOND theory is not local in g_N, and an external field
    g_ext SUPPRESSES the internal boost.  The sample's own median
    |g_ext|/a0 = 0.17 is squarely in the regime where this bites, so H3 is
    computed per pair from the measured g_ext and g_bar and is NEGATIVE.
    Two standard prescriptions bracket it (argument shift and quadrature).
"""
from __future__ import annotations

import os
import json
from math import erf, sqrt

import numpy as np
import pandas as pd

import analyse as A

A0_MS2 = 1.2e-10          # MOND acceleration scale, m/s^2
LOG_A0 = np.log10(A0_MS2)

RNG = np.random.default_rng(7654321)
LANE = A.LANE
ENV = A.ENV

NBOOT = 8000
NPERM = 8000


def cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def power(mu, sd, alpha=0.05):
    zc = 1.959963984540054
    z = mu / sd
    return cdf(z - zc) + cdf(-z - zc)


# ----------------------------------------------------------------------
# H3: MOND external-field effect, evaluated on the sample's own numbers
# ----------------------------------------------------------------------
def nu_rar(x):
    """RAR / McGaugh interpolating function: g_obs = g_bar * nu(g_bar/a0)."""
    x = np.asarray(x, float)
    return 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-12))))


def efe_offset(y, e, mode="shift"):
    """Delta log10 V for a galaxy with internal g_bar/a0 = y sitting in an
    external field e = g_ext/a0, relative to the same galaxy isolated.

    g_obs = g_bar nu(arg);  V ~ sqrt(g_obs);  so
        Delta log V = 0.5 [ log nu(arg_with_efe) - log nu(y) ].
    """
    y = np.asarray(y, float)
    e = np.asarray(e, float)
    if mode == "shift":
        arg = y + e
    elif mode == "quad":
        arg = np.sqrt(y ** 2 + e ** 2)
    else:
        raise ValueError(mode)
    return 0.5 * (np.log10(nu_rar(arg)) - np.log10(nu_rar(y)))


# ----------------------------------------------------------------------
def load_extra_manga():
    """Photometric / stellar-population diagnostics not carried by the pair file."""
    cols = ["plateifu", "nsa_absmag_g", "nsa_absmag_i", "nsa_absmag_r",
            "nsa_sersic_n", "nsa_elpetro_th50_r", "pym_r_FLAG_FIT",
            "pym_r_N_SE_DISK", "SNR_MED_r", "hi_detected"]
    m = pd.read_csv(os.path.join(ENV, "clean", "manga_env_master.csv"), usecols=cols)
    m["gi"] = m["nsa_absmag_g"] - m["nsa_absmag_i"]
    # hi_detected is already carried by matched_pairs.csv -- do not duplicate it
    return m[["plateifu", "gi", "nsa_sersic_n", "nsa_elpetro_th50_r",
              "pym_r_FLAG_FIT", "pym_r_N_SE_DISK", "SNR_MED_r"]]


def build_manga2():
    df, kin = A.build_manga()
    kin2 = pd.read_csv(os.path.join(LANE, "clean", "manga_internal_kin.csv"))
    extra = load_extra_manga()
    # columns analyse.build_manga already merged in; do not merge them twice
    already = {c[3:] for c in df.columns if c.startswith("cl_")}
    apcols = sorted({c for c in kin2.columns
                     if c.startswith(("sigma_e_tot_", "sigma_e_", "v_e_", "n_bins_"))}
                    - already)
    keep = ["plateifu", "reff_arcsec", "dlogS_dlogAp", "Rmax_kpc", "Rmax_Re",
            "ecoo_ell"] + apcols
    k = kin2[keep].merge(extra, on="plateifu", how="left")
    df = df.merge(k.add_prefix("cl_"), left_on="cl_plateifu", right_on="cl_plateifu", how="left")
    df = df.merge(k.add_prefix("fi_"), left_on="fi_plateifu", right_on="fi_plateifu", how="left")
    return df, kin


def build_sami2():
    return A.build_sami()


# ----------------------------------------------------------------------
def estimate(d, dcols, ycol_cl, ycol_fi, nboot=NBOOT, nperm=NPERM):
    dy = (d[ycol_cl] - d[ycol_fi]).to_numpy(float)
    D = np.nan_to_num(d[dcols].to_numpy(float), nan=0.0)
    m = np.isfinite(dy)
    dy, D = dy[m], D[m]
    sysid = d["sysid"].to_numpy()[m]
    if len(dy) < 8:
        return dict(status="too_few", n=int(len(dy)))
    est, coef = A.adjusted_offset(dy, D)
    boot = A.system_bootstrap(dy, D, sysid, nboot=nboot)
    perm = A.blocked_signflip(dy, D, sysid, nperm=nperm)
    sd = float(np.std(boot, ddof=1))
    sdn = float(np.std(perm, ddof=1))
    return dict(n=int(len(dy)), n_systems=int(len(np.unique(sysid))),
                offset_logV=float(est), raw=float(np.mean(dy)),
                sd_boot=sd, sd_null=sdn,
                p_perm=float(np.mean(np.abs(perm) >= abs(est))),
                ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                offset_logg=2 * float(est), sd_logg=2 * sd)


def contamination_budget2(d, survey):
    """Field-arm sensitivity dY/dX, times the cluster-minus-field mean
    difference in X, gives the spurious offset X could produce.  Reported in
    the same units as the signal.  This is the answer to 'how far did the
    environmental control get'."""
    yf = d["fi_Y"].to_numpy(float)
    if survey == "manga":
        base = ["fi_logMstar_nsa", "fi_logRd", "fi_pym_r_BT_SE", "fi_incl_deg"]
        diags = {
            "g_minus_i_colour": ("cl_gi", "fi_gi"),
            "log_Re_NSA_aperture": ("cl_logReff", "fi_logReff"),
            "kinematic_asymmetry_A_kin": ("cl_A_kin", "fi_A_kin"),
            "gas_star_PA_misalignment_deg": ("cl_misalign_deg", "fi_misalign_deg"),
            "median_sigma_astro": ("cl_med_sigma_astro", "fi_med_sigma_astro"),
            "continuum_SNR": ("cl_med_snr_1Re", "fi_med_snr_1Re"),
            "sersic_n_NSA": ("cl_nsa_sersic_n", "fi_nsa_sersic_n"),
            "aperture_coverage": ("cl_frac_good_1Re", "fi_frac_good_1Re"),
            "HI_detected": ("cl_hi_detected", "fi_hi_detected"),
        }
    else:
        base = ["fi_logMstar", "fi_logRd", "fi_incl_deg"]
        diags = {
            "kinemetry_k5_k1": ("cl_k51", "fi_k51"),
            "gas_star_PA_misalignment_deg": ("cl_misalign_deg", "fi_misalign_deg"),
            "lambda_R": ("cl_LAMBDAR_RE_MGE", "fi_LAMBDAR_RE_MGE"),
            "aperture_correction_flag": ("cl_aper_flag", "fi_aper_flag"),
            "surface_density_env": (None, "fi_SurfaceDensity"),
        }
    B = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in base])
    out = {}
    tot2 = 0.0
    for name, (cc, fc) in diags.items():
        if fc not in d.columns or (cc is not None and cc not in d.columns):
            out[name] = dict(status="column_absent")
            continue
        x = d[fc].to_numpy(float)
        mm = np.isfinite(x) & np.isfinite(yf) & np.isfinite(B).all(1)
        if mm.sum() < 25 or np.nanstd(x[mm]) == 0:
            out[name] = dict(status="too_few_or_degenerate", n=int(mm.sum()))
            continue
        X = np.column_stack([B[mm], x[mm] - np.nanmean(x[mm])])
        b = A.ols(X, yf[mm])
        slope = float(b[-1])
        if cc is None:
            out[name] = dict(field_sensitivity_dY_dX=slope, note="no cluster-arm counterpart")
            continue
        xc = d[cc].to_numpy(float)
        dx = float(np.nanmean(xc) - np.nanmean(x))
        ind = slope * dx
        # bootstrap the induced offset over host systems
        out[name] = dict(field_sensitivity_dY_dX=slope,
                         mean_cluster=float(np.nanmean(xc)),
                         mean_field=float(np.nanmean(x)),
                         mean_cluster_minus_field=dx,
                         induced_offset_logV=ind, induced_offset_logg=2 * ind,
                         n_field=int(mm.sum()))
        tot2 += ind ** 2
    out["_quadrature_total_logV"] = float(np.sqrt(tot2))
    out["_quadrature_total_logg"] = float(2 * np.sqrt(tot2))
    return out


def main():
    dm, kinm = build_manga2()
    ds, kins = build_sami2()
    a_manga = float(np.nanpercentile(kinm.loc[kinm.ok == 1, "A_kin"], A.ASYM_PERCENTILE))
    a_sami = float(np.nanpercentile(kins["k51"], A.ASYM_PERCENTILE))
    A.CUTS["manga"]["A_kin"] = a_manga
    A.CUTS["sami"]["k51"] = a_sami
    dm["cl_logReff"] = np.log10(dm["cl_reff_arcsec"])
    dm["fi_logReff"] = np.log10(dm["fi_reff_arcsec"])

    R = {"predictions": dict(
        H1_potential_depth=dict(logg=[A.PRED_G, A.PRED_G_SD],
                                logV=[A.PRED_V, A.PRED_V_SD],
                                source="tensor lane seed_robustness.json, surviving well-network points"),
        H2_acceleration_only=dict(logV=0.0, logg=0.0,
                                  source="algebraic RAR: internal g_N/a0 unchanged"),
        H3_mond_efe=dict(note="computed per pair below from the measured g_ext and g_bar")),
        "thresholds": dict(manga_A_kin=a_manga, sami_k51=a_sami)}

    # ---------- H3, on the sample's own numbers --------------------------
    h3 = {}
    for tier in ["B4_disk_wide", "C2_xray_disk", "B3_late_wide", "B2_disk_strict"]:
        g = dm[dm["tier"] == tier]
        y = 10 ** (g["cl_log_gbar_2p2Rd"].to_numpy(float) - LOG_A0)  # g_bar/a0
        e = g["cl_gext_over_a0"].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(e)
        h3[f"manga:{tier}"] = dict(
            n=int(ok.sum()),
            median_gbar_over_a0=float(np.nanmedian(y)),
            median_gext_over_a0=float(np.nanmedian(e)),
            mean_offset_logV_shift=float(np.nanmean(efe_offset(y[ok], e[ok], "shift"))),
            mean_offset_logV_quad=float(np.nanmean(efe_offset(y[ok], e[ok], "quad"))))
    for tier in ["S1_latetype", "S2_diskbearing"]:
        g = ds[ds["tier"] == tier]
        y = 10 ** (g["cl_log_gbar_2p2Rd"].to_numpy(float) - LOG_A0)
        e = g["cl_gext_over_a0"].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(e)
        h3[f"sami:{tier}"] = dict(
            n=int(ok.sum()),
            median_gbar_over_a0=float(np.nanmedian(y)),
            median_gext_over_a0=float(np.nanmedian(e)),
            mean_offset_logV_shift=float(np.nanmean(efe_offset(y[ok], e[ok], "shift"))),
            mean_offset_logV_quad=float(np.nanmean(efe_offset(y[ok], e[ok], "quad"))))
    R["predictions"]["H3_mond_efe"]["per_tier"] = h3
    print("H3 (MOND EFE) predicted offsets, log V:")
    for k, v in h3.items():
        print(f"   {k:24s} shift {v['mean_offset_logV_shift']:+.4f}  quad {v['mean_offset_logV_quad']:+.4f}"
              f"   (median g_ext/a0 = {v['median_gext_over_a0']:.3f})")

    # ---------- primary + robustness ------------------------------------
    R["tiers"] = {}
    frames = {}
    for survey, df, thr, dcols in (("manga", dm, a_manga, A.DCOLS_MANGA),
                                   ("sami", ds, a_sami, A.DCOLS_SAMI)):
        for tier in sorted(df["tier"].unique()):
            d = df[df["tier"] == tier].copy()
            n0 = len(d)
            d = d[A.apply_cuts(d, survey, thr)]
            if len(d) < 8:
                R["tiers"][f"{survey}:{tier}"] = dict(n_declared=n0, n_used=len(d), status="too_few")
                continue
            frames[(survey, tier)] = d
            rec = dict(survey=survey, tier=tier, n_declared=n0)
            rec["primary"] = estimate(d, dcols, "cl_Y", "fi_Y")
            sd = rec["primary"]["sd_boot"]
            rec["power"] = dict(
                sd=sd, mdd_3sigma_logV=3 * sd, mdd_3sigma_logg=6 * sd,
                power_H1=power(A.PRED_V, sd),
                sigma_from_H1=float((rec["primary"]["offset_logV"] - A.PRED_V) / np.hypot(sd, A.PRED_V_SD)),
                sigma_from_H2=float(rec["primary"]["offset_logV"] / sd))
            h3v = h3.get(f"{survey}:{tier}")
            if h3v:
                rec["power"]["sigma_from_H3_shift"] = float(
                    (rec["primary"]["offset_logV"] - h3v["mean_offset_logV_shift"]) / sd)
                rec["power"]["sigma_from_H3_quad"] = float(
                    (rec["primary"]["offset_logV"] - h3v["mean_offset_logV_quad"]) / sd)
            # ---- robustness: apertures (MaNGA only; SAMI has one aperture)
            if survey == "manga":
                rec["apertures"] = {}
                for ap in ["1Re", "0p5Re", "3kpc", "5kpc"]:
                    d[f"cl_Yap"] = np.log10(d[f"cl_sigma_e_tot_{ap}"])
                    d[f"fi_Yap"] = np.log10(d[f"fi_sigma_e_tot_{ap}"])
                    rec["apertures"][ap] = estimate(d, dcols, "cl_Yap", "fi_Yap",
                                                    nboot=3000, nperm=3000)
                # ---- decomposition: dispersion part vs rotation part
                d["cl_Ys"] = np.log10(d["cl_sigma_e_1Re"])
                d["fi_Ys"] = np.log10(d["fi_sigma_e_1Re"])
                rec["dispersion_only"] = estimate(d, dcols, "cl_Ys", "fi_Ys", nboot=3000, nperm=3000)
                d["cl_Yv"] = np.log10(d["cl_v_e_1Re"])
                d["fi_Yv"] = np.log10(d["fi_v_e_1Re"])
                rec["rotation_only"] = estimate(d, dcols, "cl_Yv", "fi_Yv", nboot=3000, nperm=3000)
            else:
                d["cl_Ys"] = np.log10(d["cl_SIGMA_RE_MGE"])
                d["fi_Ys"] = np.log10(d["fi_SIGMA_RE_MGE"])
                rec["dispersion_only"] = estimate(d, dcols, "cl_Ys", "fi_Ys", nboot=3000, nperm=3000)
            rec["contamination_budget"] = contamination_budget2(d, survey)
            rec["gradients"] = A.gradient_tests(d, survey, dcols)
            rec["null_simulation"] = A.null_simulation(d, survey, dcols)
            R["tiers"][f"{survey}:{tier}"] = rec
            p = rec["primary"]
            print(f"{survey:6s} {tier:16s} N={p['n']:4d} D={p['offset_logV']:+.4f} "
                  f"+-{p['sd_boot']:.4f}  (log g {2*p['offset_logV']:+.4f}) "
                  f"p={p['p_perm']:.3f} contam={rec['contamination_budget']['_quadrature_total_logV']:.4f}")

    # ---------- combined MaNGA B4 + SAMI S2 ------------------------------
    a = R["tiers"]["manga:B4_disk_wide"]["primary"]
    b = R["tiers"]["sami:S2_diskbearing"]["primary"]
    wa, wb = 1 / a["sd_boot"] ** 2, 1 / b["sd_boot"] ** 2
    comb = (wa * a["offset_logV"] + wb * b["offset_logV"]) / (wa + wb)
    sdc = 1 / np.sqrt(wa + wb)
    # disjointness check
    dmb = frames[("manga", "B4_disk_wide")]
    dsb = frames[("sami", "S2_diskbearing")]
    cc = pd.read_csv(os.path.join(ENV, "clean", "manga_sami_crosscal.csv"))
    mp_ids = set(dmb["cl_plateifu"]) | set(dmb["fi_plateifu"])
    sa_ids = set(dsb["cl_CATID"]) | set(dsb["fi_CATID"])
    shared = cc[(cc["manga_plateifu"].isin(mp_ids)) & (cc["sami_CATID"].isin(sa_ids))]
    R["combined"] = dict(
        components=["manga:B4_disk_wide", "sami:S2_diskbearing"],
        offset_logV=float(comb), sd=float(sdc),
        offset_logg=float(2 * comb), sd_logg=float(2 * sdc),
        n_galaxies_in_both_surveys=int(len(shared)),
        mdd_3sigma_logV=float(3 * sdc), mdd_3sigma_logg=float(6 * sdc),
        power_H1=power(A.PRED_V, sdc),
        sigma_from_H1_stat_only=float((comb - A.PRED_V) / sdc),
        sigma_from_H1_with_theory_scatter=float((comb - A.PRED_V) / np.hypot(sdc, A.PRED_V_SD)),
        sigma_from_H2=float(comb / sdc),
    )
    print(f"\nCOMBINED  D = {comb:+.4f} +- {sdc:.4f} (log V)  = {2*comb:+.4f} +- {2*sdc:.4f} (log g)"
          f"   overlap galaxies = {len(shared)}")

    with open(os.path.join(LANE, "member_dynamics2.json"), "w") as fh:
        json.dump(R, fh, indent=2, default=float)
    for (s, t), d in frames.items():
        d.to_csv(os.path.join(LANE, "clean", f"sample_used_{s}_{t}.csv"), index=False)
    print("wrote member_dynamics2.json")
    return R


if __name__ == "__main__":
    main()
