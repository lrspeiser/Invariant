r"""
member-dynamics lane -- assemble the deliverable.

Adds the achieved matching tolerances on the sample ACTUALLY USED, the
environment ranges of that sample, the nuisance-corrected combined estimate,
and the aperture-definition systematic; then writes the final
member_dynamics.json.
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd

import analyse as A
import analyse2 as A2
import analyse4 as A4

LANE, ENV = A.LANE, A.ENV

TOL_MANGA = {"d_logMstar_nsa": 0.10, "d_logRd": 0.10, "d_logSigma_b": 0.15,
             "d_log_gbar_2p2Rd": 0.10, "d_incl_deg": 10.0, "d_pym_r_BT_SE": 0.15,
             "d_z": 0.010}
TOL_SAMI = {"d_logMstar": 0.10, "d_logRd": 0.10, "d_logSigma_b": 0.15,
            "d_log_gbar_2p2Rd": 0.10, "d_incl_deg": 10.0, "d_z_spec": 0.010}


def tol_table(d, tol):
    out = {}
    for c, t in tol.items():
        if c not in d.columns:
            continue
        v = d[c].to_numpy(float)
        v = v[np.isfinite(v)]
        out[c] = dict(declared_tolerance=t, n=int(len(v)),
                      max_abs=float(np.max(np.abs(v))), rms=float(np.sqrt(np.mean(v ** 2))),
                      mean=float(np.mean(v)),
                      mean_over_rms_sigma=float(np.mean(v) / (np.std(v, ddof=1) / np.sqrt(len(v))))
                      if len(v) > 2 and np.std(v, ddof=1) > 0 else np.nan)
    return out


def env_table(d, survey):
    if survey == "manga":
        cols = {"host_sigma_v_kms": "cl_t14_grp_sigma_v",
                "R_over_Rvir": "cl_R_over_Rvir_t14",
                "R_proj_Mpc": "radius_mpc",
                "gext_over_a0": "cl_gext_over_a0",
                "log_Phi_proxy_m2s2": None}
    else:
        cols = {"host_sigma_v_kms": "cl_host_sigma_200",
                "R_over_R200": "cl_R_on_rtwo",
                "R_proj_Mpc": "radius_mpc",
                "gext_over_a0": "cl_gext_over_a0",
                "log_Phi_proxy_m2s2": None}
    out = {}
    for k, c in cols.items():
        if c is None:
            s = d["depth_sigma_v"].to_numpy(float) * 1e3
            v = np.log10(s ** 2)
        else:
            v = d[c].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        out[k] = dict(min=float(v.min()), p50=float(np.median(v)), max=float(v.max()))
    return out


def main():
    dm, kinm = A2.build_manga2()
    ds, kins = A2.build_sami2()
    inv = pd.read_csv(os.path.join(ENV, "raw", "sami", "sami_dr3_master_galaxy_inventory.tsv"),
                      sep="\t", usecols=["CATID", "g_i", "mu_within_1re"])
    ds = ds.merge(inv.add_prefix("cl_"), left_on="cl_CATID", right_on="cl_CATID", how="left")
    ds = ds.merge(inv.add_prefix("fi_"), left_on="fi_CATID", right_on="fi_CATID", how="left")
    a_manga = float(np.nanpercentile(kinm.loc[kinm.ok == 1, "A_kin"], A.ASYM_PERCENTILE))
    a_sami = float(np.nanpercentile(kins["k51"], A.ASYM_PERCENTILE))
    dm["cl_logReff"] = np.log10(dm["cl_reff_arcsec"])
    dm["fi_logReff"] = np.log10(dm["fi_reff_arcsec"])
    dm = A4.add_extra_cov(dm, "manga")
    ds = A4.add_extra_cov(ds, "sami")

    R = json.load(open(os.path.join(LANE, "member_dynamics.json")))
    R["sample_actually_used"] = {}
    for survey, df, thr in (("manga", dm, a_manga), ("sami", ds, a_sami)):
        for tier in sorted(df["tier"].unique()):
            base = df[df["tier"] == tier].copy()
            d = base[A4.cutset(base, survey, thr, "B")]
            if len(d) < 10:
                continue
            key = f"{survey}:{tier}"
            sysid = d["sysid"].to_numpy()
            u, n = np.unique(sysid, return_counts=True)
            R["sample_actually_used"][key] = dict(
                n_pairs_declared_by_env_data=int(len(base)),
                n_pairs_used=int(len(d)),
                attrition_frac=float(1 - len(d) / len(base)),
                n_host_systems=int(len(u)),
                n_eff_host_systems=float(n.sum() ** 2 / (n ** 2).sum()),
                largest_host_share=float(n.max() / n.sum()),
                achieved_tolerances=tol_table(d, TOL_MANGA if survey == "manga" else TOL_SAMI),
                environment=env_table(d, survey),
                file=f"clean/sample_used_{survey}_{tier}.csv")
            d.to_csv(os.path.join(LANE, "clean", f"sample_used_{survey}_{tier}.csv"), index=False)

    # ---- nuisance-corrected combined --------------------------------------
    a = R["results_by_cutset"]["manga:B4_disk_wide"]["B"]
    b = R["results_by_cutset"]["sami:S2_diskbearing"]["B"]
    ax = a["extended_covariate_adjustment"]
    bx = b["extended_covariate_adjustment"]
    wa, wb = 1 / ax["sd_boot"] ** 2, 1 / bx["sd_boot"] ** 2
    cx = (wa * ax["offset_logV"] + wb * bx["offset_logV"]) / (wa + wb)
    sx = 1 / np.sqrt(wa + wb)
    R["combined_nuisance_corrected"] = dict(
        description=("matching-variable differences PLUS the nuisance covariates the matching "
                     "never controlled (g-i colour, Sersic n, log Re, continuum S/N, kinematic "
                     "asymmetry for MaNGA; g-i colour, k5/k1, lambda_R for SAMI) regressed out"),
        offset_logV=float(cx), sd=float(sx),
        offset_logg=float(2 * cx), sd_logg=float(2 * sx),
        sigma_from_H1=float((cx - A.PRED_V) / sx),
        sigma_from_H1_plus_theory=float((cx - A.PRED_V) / np.hypot(sx, A.PRED_V_SD)),
        sigma_from_H2=float(cx / sx),
        shift_from_primary=float(cx - R["combined_corrected"]["offset_logV"]))

    # ---- aperture-definition systematic ------------------------------------
    ap = a["apertures"]
    vals = [v["offset_logV"] for v in ap.values()]
    R["aperture_systematic_manga_B4"] = dict(
        per_aperture={k: v["offset_logV"] for k, v in ap.items()},
        half_range=float((max(vals) - min(vals)) / 2),
        note="different apertures probe different radii; the spread bounds the "
             "sensitivity of the answer to the aperture definition")

    # ---- headline block -----------------------------------------------------
    c = R["combined_corrected"]
    cn = R["combined_nuisance_corrected"]
    R["headline"] = dict(
        power_before_the_answer=dict(
            stat_only_sd_logV=c["stat"],
            stat_only_power_vs_H1=A2.power(A.PRED_V, c["stat"]),
            stat_plus_syst_sd_logV=c["total_with_sami_colour"],
            stat_plus_syst_power_vs_H1=c["power_H1_total_v2"],
            mdd_3sigma_logV=c["mdd_3sigma_logV_total_v2"],
            mdd_3sigma_logg=c["mdd_3sigma_logg_total_v2"],
            verdict=("the combined sample has ~90% power against H1 on statistics alone but "
                     "only ~36% once the environmental systematic budget is included, so it "
                     "CANNOT decisively separate H1 from H2")),
        measured=dict(
            offset_logV=c["offset_logV"], stat=c["stat"], syst=c["syst_with_sami_colour"],
            offset_logg=c["offset_logg"], stat_logg=2 * c["stat"],
            syst_logg=2 * c["syst_with_sami_colour"]),
        vs_H1_potential_depth=dict(
            prediction_logg=[A.PRED_G, A.PRED_G_SD],
            sigma_conservative=c["sigma_from_H1_total_v2"],
            sigma_after_explicit_nuisance_correction=cn["sigma_from_H1"],
            sigma_including_theory_realisation_scatter=c["sigma_from_H1_total_plus_theory_v2"]),
        vs_H2_acceleration_only=dict(sigma=c["sigma_from_H2_total_v2"]),
        vs_H3_mond_efe=dict(bracket_logV=c["H3"],
                            sigma_shift=c["sigma_from_H3_shift_total"],
                            sigma_quad=c["sigma_from_H3_quad_total"]),
    )
    with open(os.path.join(LANE, "member_dynamics.json"), "w") as fh:
        json.dump(R, fh, indent=2, default=float)

    print(json.dumps(R["headline"], indent=2, default=float))
    print("\nnuisance-corrected combined:", json.dumps(cn, indent=2, default=float))
    for k, v in R["sample_actually_used"].items():
        print(f"\n{k}: used {v['n_pairs_used']}/{v['n_pairs_declared_by_env_data']} "
              f"({100*v['attrition_frac']:.0f}% attrition), {v['n_host_systems']} systems "
              f"(eff {v['n_eff_host_systems']:.1f}, largest {100*v['largest_host_share']:.0f}%)")
        for c2, t in v["achieved_tolerances"].items():
            print(f"    {c2:20s} tol {t['declared_tolerance']:<6} max {t['max_abs']:.4f} "
                  f"rms {t['rms']:.4f} mean {t['mean']:+.4f} ({t['mean_over_rms_sigma']:+.2f} sigma)")
    return R


if __name__ == "__main__":
    main()
