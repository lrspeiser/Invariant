r"""
member-dynamics lane -- final pass.

Fixes and additions over analyse2.py:
  1. The contamination budget no longer double-counts.  Two of the diagnostics
     analyse2 priced -- the median stellar dispersion and lambda_R -- ARE
     components of the observable sigma_e_tot, not independent contaminants, so
     including them in the quadrature sum prices the signal against itself.
     They are kept as diagnostics and excluded from the total.
  2. Every budget slope now carries a system-level bootstrap error, so a slope
     that is pure small-sample noise (SAMI S1's misalignment term) can be told
     from one that is real (the g-i colour term).
  3. The rotation-only tracer is promoted to a headline, because it is the
     tracer a naive version of this test would have used and it gives a large,
     highly significant, and ASTROPHYSICAL answer.
  4. Cross-survey validation of sigma_e_tot on the galaxies observed by both
     MaNGA and SAMI.
  5. The combined number with the systematic budget folded in.
  6. A high-purity subsample re-measurement.
"""
from __future__ import annotations

import os
import json
from math import erf, sqrt

import numpy as np
import pandas as pd

import analyse as A
import analyse2 as A2

RNG = np.random.default_rng(13571113)
LANE = A.LANE
ENV = A.ENV

# diagnostics that are PART of the observable and must not be priced as
# independent contamination
CIRCULAR = {"median_sigma_astro", "lambda_R"}


def budget_with_errors(d, survey, nboot=2000):
    """As analyse2.contamination_budget2, but every slope carries a
    system-level bootstrap error and the circular terms are excluded from the
    quadrature total."""
    base_out = A2.contamination_budget2(d, survey)
    sysid = d["sysid"].to_numpy()
    sys_u = np.unique(sysid)
    idx_by_sys = {s: np.where(sysid == s)[0] for s in sys_u}
    yf = d["fi_Y"].to_numpy(float)
    if survey == "manga":
        base = ["fi_logMstar_nsa", "fi_logRd", "fi_pym_r_BT_SE", "fi_incl_deg"]
        pairs = {"g_minus_i_colour": ("cl_gi", "fi_gi"),
                 "log_Re_NSA_aperture": ("cl_logReff", "fi_logReff"),
                 "kinematic_asymmetry_A_kin": ("cl_A_kin", "fi_A_kin"),
                 "gas_star_PA_misalignment_deg": ("cl_misalign_deg", "fi_misalign_deg"),
                 "median_sigma_astro": ("cl_med_sigma_astro", "fi_med_sigma_astro"),
                 "continuum_SNR": ("cl_med_snr_1Re", "fi_med_snr_1Re"),
                 "sersic_n_NSA": ("cl_nsa_sersic_n", "fi_nsa_sersic_n"),
                 "aperture_coverage": ("cl_frac_good_1Re", "fi_frac_good_1Re"),
                 "HI_detected": ("cl_hi_detected", "fi_hi_detected")}
    else:
        base = ["fi_logMstar", "fi_logRd", "fi_incl_deg"]
        pairs = {"kinemetry_k5_k1": ("cl_k51", "fi_k51"),
                 "gas_star_PA_misalignment_deg": ("cl_misalign_deg", "fi_misalign_deg"),
                 "lambda_R": ("cl_LAMBDAR_RE_MGE", "fi_LAMBDAR_RE_MGE"),
                 "aperture_correction_flag": ("cl_aper_flag", "fi_aper_flag")}
    B = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in base])
    tot2 = 0.0
    for name, (cc, fc) in pairs.items():
        rec = base_out.get(name, {})
        if "induced_offset_logV" not in rec:
            continue
        x = d[fc].to_numpy(float)
        xc = d[cc].to_numpy(float)
        vals = []
        for _ in range(nboot):
            pick = RNG.integers(0, len(sys_u), len(sys_u))
            idx = np.concatenate([idx_by_sys[sys_u[p]] for p in pick])
            mm = np.isfinite(x[idx]) & np.isfinite(yf[idx]) & np.isfinite(B[idx]).all(1)
            if mm.sum() < 25:
                continue
            ii = idx[mm]
            X = np.column_stack([B[ii], x[ii] - np.mean(x[ii])])
            try:
                b = A.ols(X, yf[ii])
            except np.linalg.LinAlgError:
                continue
            vals.append(b[-1] * (np.nanmean(xc[idx]) - np.nanmean(x[idx])))
        vals = np.array(vals)
        rec["induced_offset_logV_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 20 else np.nan
        rec["significant"] = bool(len(vals) > 20 and
                                  abs(rec["induced_offset_logV"]) > 2 * np.std(vals, ddof=1))
        rec["circular_component_of_observable"] = name in CIRCULAR
        if name not in CIRCULAR:
            tot2 += rec["induced_offset_logV"] ** 2
        base_out[name] = rec
    base_out["_quadrature_total_logV"] = float(np.sqrt(tot2))
    base_out["_quadrature_total_logg"] = float(2 * np.sqrt(tot2))
    base_out["_excluded_as_circular"] = sorted(CIRCULAR)
    return base_out


def cross_survey_validation():
    """The 103 galaxies observed by BOTH surveys: is sigma_e_tot the same
    number?  A zero-point offset between the surveys would appear as a
    field/cluster signal if the two contributed unequally to the two arms."""
    cc = pd.read_csv(os.path.join(ENV, "clean", "manga_sami_crosscal.csv"))
    km = pd.read_csv(os.path.join(LANE, "clean", "manga_internal_kin.csv"))
    _, ks = A.build_sami()
    m = cc.merge(km[["plateifu", "sigma_e_tot", "ok", "A_kin"]],
                 left_on="manga_plateifu", right_on="plateifu", how="inner")
    m = m.merge(ks[["CATID", "sigma_e_tot", "k51"]].rename(
        columns={"sigma_e_tot": "sigma_e_tot_sami"}),
        left_on="sami_CATID", right_on="CATID", how="inner")
    m = m[(m["ok"] == 1) & np.isfinite(m["sigma_e_tot"]) & np.isfinite(m["sigma_e_tot_sami"])]
    if len(m) < 5:
        return dict(n=int(len(m)), status="too_few_with_both_measurements")
    dl = np.log10(m["sigma_e_tot"]) - np.log10(m["sigma_e_tot_sami"])
    return dict(n=int(len(m)),
                median_offset_manga_minus_sami_logV=float(np.median(dl)),
                mean_offset=float(np.mean(dl)), sd=float(np.std(dl, ddof=1)),
                sem=float(np.std(dl, ddof=1) / np.sqrt(len(dl))),
                note="log10 sigma_e_tot(MaNGA, this lane, 1 Re from the DAP cubes) "
                     "minus log10 sigma_e_tot(SAMI DR3, SIGMA_RE_MGE sqrt(1+VSIGMA_RE_MGE^2))")


def main():
    dm, kinm = A2.build_manga2()
    ds, kins = A2.build_sami2()
    a_manga = float(np.nanpercentile(kinm.loc[kinm.ok == 1, "A_kin"], A.ASYM_PERCENTILE))
    a_sami = float(np.nanpercentile(kins["k51"], A.ASYM_PERCENTILE))
    A.CUTS["manga"]["A_kin"] = a_manga
    A.CUTS["sami"]["k51"] = a_sami
    dm["cl_logReff"] = np.log10(dm["cl_reff_arcsec"])
    dm["fi_logReff"] = np.log10(dm["fi_reff_arcsec"])

    prev = json.load(open(os.path.join(LANE, "member_dynamics2.json")))
    pre = json.load(open(os.path.join(LANE, "power_prestatement.json")))

    R = {
        "lane": "work/wellnet-2026-09/member-dynamics",
        "question": "Does a galaxy's internal dynamics change when it sits inside a cluster?",
        "measure": ("Y = log10 sigma_e_tot, the flux-weighted aperture second velocity "
                    "moment of the STARS inside 1 Re: sigma_e_tot^2 = <V^2 + sigma^2>_F"),
        "predictions": prev["predictions"],
        "power_prestatement": pre,
        "sealed_holdouts": "KiDS and wide binaries were never loaded, listed, queried or referenced.",
        "thresholds": prev["thresholds"],
        "tiers": {},
    }

    frames = {}
    for survey, df, thr, dcols in (("manga", dm, a_manga, A.DCOLS_MANGA),
                                   ("sami", ds, a_sami, A.DCOLS_SAMI)):
        for tier in sorted(df["tier"].unique()):
            d = df[df["tier"] == tier].copy()
            n0 = len(d)
            d = d[A.apply_cuts(d, survey, thr)]
            if len(d) < 8:
                R["tiers"][f"{survey}:{tier}"] = dict(n_declared=n0, n_used=int(len(d)),
                                                      status="too_few_pairs_after_cuts")
                continue
            frames[(survey, tier)] = d
            rec = dict(prev["tiers"][f"{survey}:{tier}"])
            rec["contamination_budget"] = budget_with_errors(d, survey)

            # ---- the naive tracer, for contrast -----------------------------
            if survey == "manga":
                d["cl_Yv"] = np.log10(d["cl_v_e_1Re"])
                d["fi_Yv"] = np.log10(d["fi_v_e_1Re"])
                rec["rotation_only"] = A2.estimate(d, dcols, "cl_Yv", "fi_Yv",
                                                   nboot=4000, nperm=4000)

            # ---- high-purity subsample --------------------------------------
            if survey == "manga":
                hp = ((d["cl_A_kin"] <= np.nanpercentile(d["cl_A_kin"], 50)) &
                      (d["fi_A_kin"] <= np.nanpercentile(d["fi_A_kin"], 50)) &
                      (d["cl_med_sigma_astro"] >= 60) & (d["fi_med_sigma_astro"] >= 60) &
                      (d["cl_frac_good_1Re"] >= 0.9) & (d["fi_frac_good_1Re"] >= 0.9))
            else:
                hp = ((d["cl_k51"] <= np.nanpercentile(d["cl_k51"], 50)) &
                      (d["fi_k51"] <= np.nanpercentile(d["fi_k51"], 50)) &
                      (d["cl_aper_flag"] == 0) & (d["fi_aper_flag"] == 0))
            dh = d[hp.fillna(False)]
            rec["high_purity"] = (A2.estimate(dh, dcols, "cl_Y", "fi_Y", nboot=3000, nperm=3000)
                                  if len(dh) >= 12 else dict(status="too_few", n=int(len(dh))))

            # ---- statistical + systematic ------------------------------------
            p = rec["primary"]
            syst = rec["contamination_budget"]["_quadrature_total_logV"]
            rec["total_uncertainty"] = dict(
                stat=p["sd_boot"], syst=syst,
                total=float(np.hypot(p["sd_boot"], syst)),
                offset_logV=p["offset_logV"], offset_logg=2 * p["offset_logV"],
                total_logg=float(2 * np.hypot(p["sd_boot"], syst)))
            R["tiers"][f"{survey}:{tier}"] = rec

    # ---------- combined, with systematics ---------------------------------
    a = R["tiers"]["manga:B4_disk_wide"]
    b = R["tiers"]["sami:S2_diskbearing"]
    for tag, key in (("stat_only", "sd_boot"),):
        pass
    ea = a["total_uncertainty"]["total"]
    eb = b["total_uncertainty"]["total"]
    wa, wb = 1 / ea ** 2, 1 / eb ** 2
    comb = (wa * a["primary"]["offset_logV"] + wb * b["primary"]["offset_logV"]) / (wa + wb)
    sdc = 1 / np.sqrt(wa + wb)
    sa, sb = a["primary"]["sd_boot"], b["primary"]["sd_boot"]
    wsa, wsb = 1 / sa ** 2, 1 / sb ** 2
    comb_stat = (wsa * a["primary"]["offset_logV"] + wsb * b["primary"]["offset_logV"]) / (wsa + wsb)
    sd_stat = 1 / np.sqrt(wsa + wsb)

    h3 = R["predictions"]["H3_mond_efe"]["per_tier"]
    h3s = (wsa * h3["manga:B4_disk_wide"]["mean_offset_logV_shift"] +
           wsb * h3["sami:S2_diskbearing"]["mean_offset_logV_shift"]) / (wsa + wsb)
    h3q = (wsa * h3["manga:B4_disk_wide"]["mean_offset_logV_quad"] +
           wsb * h3["sami:S2_diskbearing"]["mean_offset_logV_quad"]) / (wsa + wsb)

    R["combined"] = dict(
        components=["manga:B4_disk_wide", "sami:S2_diskbearing"],
        n_pairs=int(a["primary"]["n"] + b["primary"]["n"]),
        n_systems=int(a["primary"]["n_systems"] + b["primary"]["n_systems"]),
        galaxies_in_both_surveys=int(prev["combined"]["n_galaxies_in_both_surveys"]),
        offset_logV=float(comb_stat), stat=float(sd_stat),
        syst=float(np.sqrt(max(sdc ** 2 - sd_stat ** 2, 0.0))),
        offset_logV_systweighted=float(comb), sd_systweighted=float(sdc),
        offset_logg=float(2 * comb_stat), stat_logg=float(2 * sd_stat),
        syst_budget_logV=float(np.hypot(
            a["contamination_budget"]["_quadrature_total_logV"] * wsa / (wsa + wsb),
            b["contamination_budget"]["_quadrature_total_logV"] * wsb / (wsa + wsb))),
        H1=dict(value=A.PRED_V, sd=A.PRED_V_SD),
        H2=dict(value=0.0),
        H3=dict(shift=float(h3s), quad=float(h3q)),
    )
    tot = np.hypot(sd_stat, R["combined"]["syst_budget_logV"])
    R["combined"]["total_uncertainty_logV"] = float(tot)
    R["combined"]["sigma_from_H1_stat_only"] = float((comb_stat - A.PRED_V) / sd_stat)
    R["combined"]["sigma_from_H1_stat_plus_syst"] = float((comb_stat - A.PRED_V) / tot)
    R["combined"]["sigma_from_H1_stat_syst_theory"] = float(
        (comb_stat - A.PRED_V) / np.sqrt(tot ** 2 + A.PRED_V_SD ** 2))
    R["combined"]["sigma_from_H2_stat_only"] = float(comb_stat / sd_stat)
    R["combined"]["sigma_from_H2_stat_plus_syst"] = float(comb_stat / tot)
    R["combined"]["sigma_from_H3_shift"] = float((comb_stat - h3s) / tot)
    R["combined"]["sigma_from_H3_quad"] = float((comb_stat - h3q) / tot)

    R["cross_survey_validation"] = cross_survey_validation()

    with open(os.path.join(LANE, "member_dynamics.json"), "w") as fh:
        json.dump(R, fh, indent=2, default=float)

    c = R["combined"]
    print("=" * 78)
    print(f"COMBINED  Delta log V = {c['offset_logV']:+.4f} +- {c['stat']:.4f} (stat) "
          f"+- {c['syst_budget_logV']:.4f} (syst)")
    print(f"          Delta log g = {c['offset_logg']:+.4f} +- {2*c['stat']:.4f} (stat) "
          f"+- {2*c['syst_budget_logV']:.4f} (syst)")
    print(f"  vs H1 potential depth (+{A.PRED_V:.4f}): {c['sigma_from_H1_stat_plus_syst']:+.2f} sigma "
          f"(stat+syst), {c['sigma_from_H1_stat_syst_theory']:+.2f} sigma incl. theory scatter")
    print(f"  vs H2 acceleration-only (0):       {c['sigma_from_H2_stat_plus_syst']:+.2f} sigma")
    print(f"  vs H3 MOND EFE ({h3s:+.4f}/{h3q:+.4f}): {c['sigma_from_H3_shift']:+.2f} / "
          f"{c['sigma_from_H3_quad']:+.2f} sigma")
    print("\ncross-survey validation:", json.dumps(R["cross_survey_validation"], default=float))
    for k in ["manga:B4_disk_wide", "sami:S2_diskbearing", "manga:C2_xray_disk"]:
        t = R["tiers"][k]
        print(f"\n{k}: D={t['primary']['offset_logV']:+.4f} stat {t['primary']['sd_boot']:.4f} "
              f"syst {t['contamination_budget']['_quadrature_total_logV']:.4f}"
              f"   high-purity {t['high_purity'].get('offset_logV', float('nan')):+.4f} "
              f"(n={t['high_purity'].get('n', 0)})")
        if "rotation_only" in t:
            r = t["rotation_only"]
            print(f"   rotation-only tracer: {r['offset_logV']:+.4f} +- {r['sd_boot']:.4f} "
                  f"({r['offset_logV']/r['sd_boot']:+.1f} sigma)")
    print("\nwrote member_dynamics.json")
    return R


if __name__ == "__main__":
    main()
