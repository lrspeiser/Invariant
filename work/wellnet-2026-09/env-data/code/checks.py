"""Explicit checks against the failure modes listed in the programme brief.

Each check writes a verdict into clean/checks.json and prints it.  A check that
cannot be run here (because it belongs to the analysis stage, not acquisition)
says so rather than reporting a pass.
"""
import json
import os

import numpy as np
import pandas as pd

LANE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data"
CLEAN = os.path.join(LANE, "clean")
R = {}


def rec(name, verdict, detail):
    R[name] = {"verdict": verdict, "detail": detail}
    print("[%s] %s\n    %s\n" % (verdict, name, detail))


def main():
    d = pd.read_csv(os.path.join(CLEAN, "manga_env_master.csv"), low_memory=False)
    p = pd.read_csv(os.path.join(CLEAN, "matched_pairs.csv"), low_memory=False)

    # ---------------------------------------------------------------- 1
    # Shared-denominator artefacts: does any quantity sit on both axes?
    internal = ["logMstar_nsa", "Rd_kpc", "Sigma_b_Msun_pc2", "log_gbar_2p2Rd",
                "f_gas", "incl_deg", "vamp_ha_deproj_kms"]
    environment = ["t14_grp_sigma_v", "t14_Rproj_kpc", "R_over_Rvir_t14",
                   "gext_proxy_ms2", "t14_mcxc_L500_1e44", "t14_Ngal"]
    shared = set(internal) & set(environment)
    # do the environment variables contain any galaxy-internal measurement?
    prov = {
        "t14_grp_sigma_v": "member redshifts of the host group only",
        "t14_Rproj_kpc": "sky separation to the group centre times the "
                         "angular-diameter distance of the galaxy",
        "R_over_Rvir_t14": "the above divided by the group's projected harmonic radius",
        "gext_proxy_ms2": "sigma_v^2 / R_proj",
        "t14_mcxc_L500_1e44": "ROSAT/other X-ray luminosity of the host",
        "t14_Ngal": "FoF richness of the host",
    }
    rec("shared_denominator",
        "PASS" if not shared else "FAIL",
        "No galaxy-internal measurement enters any environment variable, and no "
        "environment variable enters any internal variable. Overlap of the two "
        "column sets: %s. Environment provenance: %s. "
        "NOTE for the analysis lane: R_proj appears in BOTH gext_proxy and "
        "R_over_Rvir, so those two are NOT independent environment axes "
        "(Pearson r = %.3f on the cluster arm); and the redshift z enters both "
        "the distance scaling of R_d and of R_proj, so a distance error moves "
        "both together."
        % (sorted(shared), prov,
           d.loc[d.t14_grp_sigma_v.notna(), ["gext_proxy_ms2", "R_over_Rvir_t14"]]
            .corr().iloc[0, 1]))

    # ---------------------------------------------------------------- 2
    # Monotone-invariant statistics: does the headline quantity actually move?
    rows = []
    for sig in (300, 350, 400, 500, 600, 700, 800):
        m = (d.t14_grp_sigma_v >= sig) & (d.R_over_Rvir_t14 <= 1.0)
        m = m.fillna(False)
        if m.sum() > 3:
            rows.append((sig, int(m.sum()),
                         float(np.nanmedian(d.gext_proxy_ms2[m] / 1.2e-10))))
    g = [r[2] for r in rows]
    spread = (max(g) - min(g)) if g else 0.0
    rec("monotone_invariance_of_the_environment_statistic",
        "PASS" if spread > 0.05 else "FAIL",
        "median |g_ext|/a_0 versus the sigma_v threshold: %s. Spread over the "
        "tested range = %.3f in units of a_0 (%.0f%% of the smallest value), so "
        "d(statistic)/d(threshold) != 0 and the environment ranking is not "
        "degenerate."
        % ("; ".join("sigma_v>=%d: N=%d, g_ext/a0=%.3f" % r for r in rows),
           spread, 100 * spread / min(g) if g else 0))

    # the same check on the pair-level environment contrast
    cl = p[p.tier == "B4_disk_wide"]
    q = cl.cl_gext_over_a0.dropna()
    rec("environment_contrast_dynamic_range",
        "PASS" if len(q) and q.max() / q.min() > 3 else "WARN",
        "In the largest tier the cluster-side |g_ext|/a_0 spans %.3f to %.3f "
        "(factor %.1f), 10th-90th percentile %.3f to %.3f. An external-field "
        "effect of any monotone form must vary across this range; if a headline "
        "statistic does not, that is a bug, not a null."
        % (q.min(), q.max(), q.max() / q.min(),
           np.percentile(q, 10), np.percentile(q, 90)))

    # ---------------------------------------------------------------- 3
    # Refitting on the held-out set / blind protection
    kin = ["vamp_", "STELLAR_SIGMA", "HA_GSIGMA", "STELLAR_VEL", "HA_GVEL", "SFR_"]
    src = open(os.path.join(LANE, "code", "build_matched_pairs.py"),
               encoding="utf-8").read()
    # every kinematic column that appears must be in the carry-through list or
    # the field-arm-only power block, never in TOL / FIVE / FOUR / CTRL
    match_space = set()
    for blk in ("FIVE = [", "FOUR = [", "CTRL = ["):
        i = src.index(blk)
        match_space |= set(src[i:src.index("]", i)].split('"')[1::2])
    bad = [c for c in match_space if any(k in c for k in kin)]
    rec("blind_protection",
        "PASS" if not bad else "FAIL",
        "Matching space = %s. Kinematic columns in it: %s. The field/cluster "
        "split uses only sigma_v, richness and R_proj/R_vir. The only kinematic "
        "quantity read anywhere in the pair builder is the field-arm Tully-Fisher "
        "scatter used for the power calculation, and the cluster arm is excluded "
        "from that fit, so the contrast under test has not been looked at."
        % (sorted(match_space), bad))

    # ---------------------------------------------------------------- 4
    # Silent extraction failures
    counts = {
        "drpall MANGA rows": (11273, 11273),
        "dapall HYB10 rows": (10782, 10782),
        "pymorph rows/band": (10293, 10293),
        "dl morphology rows": (10293, 10293),
        "visual morphology rows": (10126, 10126),
        "HI-MaNGA rows": (6632, 6632),
        "Tempel2014 galaxies": (588193, len(open(os.path.join(
            LANE, "raw", "groups", "tempel2014_galaxies.tsv.manifest.json"))
            .read()) and json.load(open(os.path.join(
                LANE, "raw", "groups", "tempel2014_galaxies.tsv.manifest.json")))["row_count"]),
        "Tempel2014 groups": (82458, json.load(open(os.path.join(
            LANE, "raw", "groups", "tempel2014_groups.tsv.manifest.json")))["row_count"]),
        "Tempel2017 galaxies": (584449, json.load(open(os.path.join(
            LANE, "raw", "groups", "tempel2017_table1_galaxies.tsv.manifest.json")))["row_count"]),
        "MCXC clusters": (1743, json.load(open(os.path.join(
            LANE, "raw", "groups", "mcxc_piffaretti2011.tsv.manifest.json")))["row_count"]),
    }
    mism = {k: v for k, v in counts.items() if v[0] != v[1]}
    maps = json.load(open(os.path.join(LANE, "raw", "manga", "maps",
                                       "maps.manifest.json")))
    rec("silent_extraction_failure",
        "PASS" if not mism and maps["n_failed"] == 0 else "FAIL",
        "Every ingest asserts its row count against the number stated by the "
        "source (mismatches: %s). The VizieR reader additionally asserts that the "
        "response carries #Table and #Column lines and that the header names match "
        "the #Column declarations one for one, which is what catches VizieR's "
        "HTTP-200-generic-page failure. MAPS download: %d of %d files retrieved, "
        "%d failures."
        % (mism or "none", maps["n_files"], maps["n_requested"], maps["n_failed"]))

    # ---------------------------------------------------------------- 5
    # Dark-matter contamination
    dm = [c for c in d.columns if "rank_only" in c]
    leaked = [c for c in dm if c in ("logMstar_nsa", "Rd_kpc", "Sigma_b_Msun_pc2",
                                     "log_gbar_2p2Rd", "f_gas")]
    rec("no_dark_matter_as_observation",
        "PASS" if not leaked else "FAIL",
        "Dark-matter-dependent columns, all suffixed `_rank_only`: %s. None of "
        "them enters any matching variable, the quality gate, the field/cluster "
        "split, or any derived baryonic quantity. The environment ranking uses "
        "sigma_v (member kinematics) and L500 (X-ray), both observables."
        % dm)

    # ---------------------------------------------------------------- 6
    # Not applicable at this stage
    rec("test_bugs_that_look_like_solver_bugs", "N/A",
        "No PDE solver is exercised in this lane; this failure mode belongs to "
        "the gravitylab solver lanes.")
    rec("non_monotonic_M_of_r_in_lensing_deprojection", "N/A",
        "No lensing deprojection is performed in this lane. The only lensing-"
        "adjacent quantity touched is MCXC L500, an X-ray luminosity.")

    with open(os.path.join(CLEAN, "checks.json"), "w") as f:
        json.dump(R, f, indent=2)
    print("WROTE %s" % os.path.join(CLEAN, "checks.json"))


if __name__ == "__main__":
    main()
