r"""
member-dynamics lane -- definitive pass, with a correction to the declared cuts.

WHAT WENT WRONG WITH THE DECLARED CUTS, AND WHY IT MATTERS
----------------------------------------------------------
The cut set declared in analyse.py included `med_sigma_astro >= 40 km/s`,
imposed on BOTH members of a pair, as a guard against the MaNGA DAP dispersion
floor.  That is a cut on the OBSERVABLE.  Requiring both members of a matched
pair to exceed a threshold in a quantity that is monotonically related to Y
truncates the two arms unequally whenever their Y distributions differ, and
biases the paired difference.  The direction is predictable: cluster members
are dispersion-dominated more often, so fewer of them are removed, the field
arm is truncated harder, its surviving Y is raised, and the measured
cluster-minus-field difference is pushed POSITIVE.

This is demonstrated below by scanning the threshold: the offset marches
monotonically upward as the sigma cut tightens, in a sample where the
underlying answer cannot depend on it.

Four cut sets are therefore run and all four are reported:
  A  declared      -- exactly as pre-registered, including the sigma floor
  B  outcome-free  -- the same minus the sigma floor          <- CORRECTED PRIMARY
  C  B + mass floor-- B plus log M* >= 9.8 on both members.  M* is a MATCHED
                      variable, so a cut on it is balanced by construction and
                      does not select on the outcome; it removes the galaxies
                      near the dispersion floor without the collider bias.
  D  B + sigma 60  -- the artefact, exaggerated, for display only
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd

import analyse as A
import analyse2 as A2
import analyse3 as A3

RNG = np.random.default_rng(2718281)
LANE, ENV = A.LANE, A.ENV

EXTRA_COV_MANGA = ["dc_gi", "dc_sersic_n", "dc_logReff", "dc_snr", "dc_Akin"]
EXTRA_COV_SAMI = ["dc_gi", "dc_k51", "dc_lambda"]


def cutset(d, survey, thr, which):
    ok = np.ones(len(d), bool)
    for side in ("cl_", "fi_"):
        ok &= np.isfinite(d[side + "Y"]).to_numpy()
        if survey == "manga":
            ok &= (d[side + "ok"] == 1).to_numpy()
            ok &= (d[side + "frac_good_1Re"] >= 0.50).to_numpy()
            ok &= (d[side + "n_bins_1Re"] >= 10).to_numpy()
            ok &= (d[side + "A_kin"] <= thr).to_numpy()
            if which == "A":
                ok &= (d[side + "med_sigma_astro"] >= 40.0).to_numpy()
            elif which == "D":
                ok &= (d[side + "med_sigma_astro"] >= 60.0).to_numpy()
            if which == "C":
                ok &= (d[side + "logMstar_nsa"] >= 9.8).to_numpy()
        else:
            ok &= (d[side + "k51"] <= thr).to_numpy()
            ok &= (d[side + "aper_flag"] <= 1).to_numpy()
            if which == "C":
                ok &= (d[side + "logMstar"] >= 9.8).to_numpy()
            if which == "D":
                ok &= (d[side + "SIGMA_RE_MGE"] >= 90.0).to_numpy()
    return ok


def add_extra_cov(d, survey):
    if survey == "manga":
        d["dc_gi"] = d["cl_gi"] - d["fi_gi"]
        d["dc_sersic_n"] = d["cl_nsa_sersic_n"] - d["fi_nsa_sersic_n"]
        d["dc_logReff"] = d["cl_logReff"] - d["fi_logReff"]
        d["dc_snr"] = d["cl_med_snr_1Re"] - d["fi_med_snr_1Re"]
        d["dc_Akin"] = d["cl_A_kin"] - d["fi_A_kin"]
    else:
        d["dc_gi"] = d["cl_g_i"] - d["fi_g_i"]
        d["dc_k51"] = d["cl_k51"] - d["fi_k51"]
        d["dc_lambda"] = d["cl_LAMBDAR_RE_MGE"] - d["fi_LAMBDAR_RE_MGE"]
    return d


def main():
    dm, kinm = A2.build_manga2()
    ds, kins = A2.build_sami2()
    # SAMI colour, for a budget as complete as MaNGA's
    inv = pd.read_csv(os.path.join(ENV, "raw", "sami", "sami_dr3_master_galaxy_inventory.tsv"),
                      sep="\t", usecols=["CATID", "g_i", "mu_within_1re", "r_e"])
    ds = ds.merge(inv.add_prefix("cl_"), left_on="cl_CATID", right_on="cl_CATID", how="left")
    ds = ds.merge(inv.add_prefix("fi_"), left_on="fi_CATID", right_on="fi_CATID", how="left")

    a_manga = float(np.nanpercentile(kinm.loc[kinm.ok == 1, "A_kin"], A.ASYM_PERCENTILE))
    a_sami = float(np.nanpercentile(kins["k51"], A.ASYM_PERCENTILE))
    dm["cl_logReff"] = np.log10(dm["cl_reff_arcsec"])
    dm["fi_logReff"] = np.log10(dm["fi_reff_arcsec"])
    dm = add_extra_cov(dm, "manga")
    ds = add_extra_cov(ds, "sami")

    prev = json.load(open(os.path.join(LANE, "member_dynamics.json")))
    R = dict(prev)
    R["cut_sets"] = {
        "A_declared": "pre-registered, includes med_sigma_astro >= 40 km/s (a cut on the observable)",
        "B_outcome_free": "A minus the sigma floor -- CORRECTED PRIMARY",
        "C_mass_floor": "B plus log M* >= 9.8 in both arms; M* is a matched variable",
        "D_sigma60": "B plus a 60 km/s sigma floor; shown only to display the artefact",
    }
    R["results_by_cutset"] = {}
    R["sigma_cut_artefact_scan"] = {}

    for survey, df, thr, dcols, xcov in (("manga", dm, a_manga, A.DCOLS_MANGA, EXTRA_COV_MANGA),
                                         ("sami", ds, a_sami, A.DCOLS_SAMI, EXTRA_COV_SAMI)):
        for tier in sorted(df["tier"].unique()):
            base = df[df["tier"] == tier].copy()
            key = f"{survey}:{tier}"
            R["results_by_cutset"][key] = {}
            for which in ("A", "B", "C", "D"):
                d = base[cutset(base, survey, thr, which)]
                if len(d) < 10:
                    R["results_by_cutset"][key][which] = dict(status="too_few", n=int(len(d)))
                    continue
                est = A2.estimate(d, dcols, "cl_Y", "fi_Y", nboot=4000, nperm=4000)
                if which == "B":
                    # adjustment extended with the nuisance covariates the
                    # matching never controlled
                    est_x = A2.estimate(d, dcols + xcov, "cl_Y", "fi_Y",
                                        nboot=4000, nperm=4000)
                    est["extended_covariate_adjustment"] = est_x
                    est["contamination_budget"] = A3.budget_with_errors(d, survey)
                    est["gradients"] = A.gradient_tests(d, survey, dcols)
                    est["null_simulation"] = A.null_simulation(d, survey, dcols)
                    if survey == "manga":
                        for ap in ["1Re", "0p5Re", "3kpc", "5kpc"]:
                            d["cl_Yap"] = np.log10(d[f"cl_sigma_e_tot_{ap}"])
                            d["fi_Yap"] = np.log10(d[f"fi_sigma_e_tot_{ap}"])
                            est.setdefault("apertures", {})[ap] = A2.estimate(
                                d, dcols, "cl_Yap", "fi_Yap", nboot=2500, nperm=2500)
                        d["cl_Yv"] = np.log10(d["cl_v_e_1Re"])
                        d["fi_Yv"] = np.log10(d["fi_v_e_1Re"])
                        est["rotation_only"] = A2.estimate(d, dcols, "cl_Yv", "fi_Yv",
                                                           nboot=2500, nperm=2500)
                        d["cl_Ys"] = np.log10(d["cl_sigma_e_1Re"])
                        d["fi_Ys"] = np.log10(d["fi_sigma_e_1Re"])
                        est["dispersion_only"] = A2.estimate(d, dcols, "cl_Ys", "fi_Ys",
                                                             nboot=2500, nperm=2500)
                    else:
                        d["cl_Ys"] = np.log10(d["cl_SIGMA_RE_MGE"])
                        d["fi_Ys"] = np.log10(d["fi_SIGMA_RE_MGE"])
                        est["dispersion_only"] = A2.estimate(d, dcols, "cl_Ys", "fi_Ys",
                                                             nboot=2500, nperm=2500)
                    d.to_csv(os.path.join(LANE, "clean", f"sample_used_{survey}_{tier}.csv"),
                             index=False)
                R["results_by_cutset"][key][which] = est

            # ---- the artefact scan ------------------------------------------
            scan = []
            grid = ([0, 30, 40, 50, 60, 70, 80] if survey == "manga"
                    else [0, 60, 75, 90, 105, 120])
            col = "med_sigma_astro" if survey == "manga" else "SIGMA_RE_MGE"
            for s in grid:
                sel = cutset(base, survey, thr, "B")
                sel &= (base["cl_" + col] >= s).to_numpy() & (base["fi_" + col] >= s).to_numpy()
                d = base[sel]
                if len(d) < 12:
                    continue
                dy = (d["cl_Y"] - d["fi_Y"]).to_numpy()
                D = np.nan_to_num(d[dcols].to_numpy(float), nan=0.0)
                mfin = np.isfinite(dy)
                e, _ = A.adjusted_offset(dy[mfin], D[mfin])
                scan.append(dict(threshold=s, n=int(mfin.sum()), offset_logV=float(e)))
            R["sigma_cut_artefact_scan"][key] = scan

    # ---------- corrected combined ----------------------------------------
    a = R["results_by_cutset"]["manga:B4_disk_wide"]["B"]
    b = R["results_by_cutset"]["sami:S2_diskbearing"]["B"]
    sa, sb = a["sd_boot"], b["sd_boot"]
    wa, wb = 1 / sa ** 2, 1 / sb ** 2
    comb = (wa * a["offset_logV"] + wb * b["offset_logV"]) / (wa + wb)
    sd = 1 / np.sqrt(wa + wb)
    bud = float(np.hypot(a["contamination_budget"]["_quadrature_total_logV"] * wa / (wa + wb),
                         b["contamination_budget"]["_quadrature_total_logV"] * wb / (wa + wb)))
    tot = float(np.hypot(sd, bud))
    h3 = R["predictions"]["H3_mond_efe"]["per_tier"]
    h3s = (wa * h3["manga:B4_disk_wide"]["mean_offset_logV_shift"] +
           wb * h3["sami:S2_diskbearing"]["mean_offset_logV_shift"]) / (wa + wb)
    h3q = (wa * h3["manga:B4_disk_wide"]["mean_offset_logV_quad"] +
           wb * h3["sami:S2_diskbearing"]["mean_offset_logV_quad"]) / (wa + wb)
    R["combined_corrected"] = dict(
        cut_set="B_outcome_free", components=["manga:B4_disk_wide", "sami:S2_diskbearing"],
        n_pairs=int(a["n"] + b["n"]), n_systems=int(a["n_systems"] + b["n_systems"]),
        offset_logV=float(comb), stat=float(sd), syst=bud, total=tot,
        offset_logg=float(2 * comb), stat_logg=float(2 * sd),
        syst_logg=float(2 * bud), total_logg=float(2 * tot),
        H1=[A.PRED_V, A.PRED_V_SD], H2=0.0, H3=[float(h3s), float(h3q)],
        sigma_from_H1_stat=float((comb - A.PRED_V) / sd),
        sigma_from_H1_total=float((comb - A.PRED_V) / tot),
        sigma_from_H1_total_plus_theory=float((comb - A.PRED_V) / np.hypot(tot, A.PRED_V_SD)),
        sigma_from_H2_stat=float(comb / sd),
        sigma_from_H2_total=float(comb / tot),
        sigma_from_H3_shift_total=float((comb - h3s) / tot),
        sigma_from_H3_quad_total=float((comb - h3q) / tot),
        mdd_3sigma_logV_stat=float(3 * sd), mdd_3sigma_logV_total=float(3 * tot),
        power_H1_stat=A2.power(A.PRED_V, sd), power_H1_total=A2.power(A.PRED_V, tot),
    )
    R["cross_survey_validation"] = A3.cross_survey_validation()

    with open(os.path.join(LANE, "member_dynamics.json"), "w") as fh:
        json.dump(R, fh, indent=2, default=float)

    print("sigma-cut artefact scan (MaNGA B4):")
    for r in R["sigma_cut_artefact_scan"]["manga:B4_disk_wide"]:
        print(f"   sigma >= {r['threshold']:3d} km/s  n={r['n']:4d}  D={r['offset_logV']:+.4f}")
    print("\noffsets by cut set:")
    for k, v in R["results_by_cutset"].items():
        s = "  ".join(f"{w}:{v[w].get('offset_logV', float('nan')):+.4f}(n={v[w].get('n',0)})"
                      for w in "ABCD" if w in v)
        print(f"   {k:24s} {s}")
    c = R["combined_corrected"]
    print(f"\nCORRECTED COMBINED  Delta log V = {c['offset_logV']:+.4f} "
          f"+- {c['stat']:.4f}(stat) +- {c['syst']:.4f}(syst)")
    print(f"                    Delta log g = {c['offset_logg']:+.4f} "
          f"+- {c['stat_logg']:.4f}(stat) +- {c['syst_logg']:.4f}(syst)")
    print(f"   H1 {c['sigma_from_H1_total']:+.2f} sigma (total), "
          f"{c['sigma_from_H1_total_plus_theory']:+.2f} incl theory scatter")
    print(f"   H2 {c['sigma_from_H2_total']:+.2f} sigma;  "
          f"H3 {c['sigma_from_H3_shift_total']:+.2f}/{c['sigma_from_H3_quad_total']:+.2f} sigma")
    return R


if __name__ == "__main__":
    main()
