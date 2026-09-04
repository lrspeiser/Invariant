"""Emit MATCHED_PAIRS.md from clean/matched_pairs.csv + matched_pairs_summary.json."""
import json
import os

import numpy as np
import pandas as pd

LANE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data"
CLEAN = os.path.join(LANE, "clean")
OUT = os.path.join(LANE, "MATCHED_PAIRS.md")

TIER_ORDER = ["B1_primary", "A1_gas_matched", "B2_disk_strict", "B3_late_wide",
              "B4_disk_wide", "C1_xray_late", "C2_xray_disk"]
FIVE = ["logMstar_nsa", "logRd", "logSigma_b", "log_gbar_2p2Rd", "f_gas_or_nan"]
CTRL = ["incl_deg", "pym_r_BT_SE", "z"]
LBL = {"logMstar_nsa": "log10 M_b (stellar)", "logRd": "log10 R_d",
       "logSigma_b": "log10 Sigma_b", "log_gbar_2p2Rd": "log10 g_bar(2.2 R_d)",
       "f_gas_or_nan": "f_gas", "incl_deg": "inclination", "pym_r_BT_SE": "B/T",
       "z": "redshift"}
UNIT = {"logMstar_nsa": "dex", "logRd": "dex", "logSigma_b": "dex",
        "log_gbar_2p2Rd": "dex", "f_gas_or_nan": "abs", "incl_deg": "deg",
        "pym_r_BT_SE": "abs", "z": "abs"}


def main():
    p = pd.read_csv(os.path.join(CLEAN, "matched_pairs.csv"), low_memory=False)
    s = json.load(open(os.path.join(CLEAN, "matched_pairs_summary.json")))
    L = []
    w = L.append

    w("# MATCHED_PAIRS.md — field/cluster matched galaxy pairs (Test 1)")
    w("")
    w("Lane: `work/wellnet-2026-09/env-data/`.  "
      "Machine-readable version: `clean/matched_pairs.csv` "
      "(one row per pair, cluster member prefixed `cl_`, field member `fi_`).  "
      "Per-tier statistics: `clean/matched_pairs_summary.json`.")
    w("")
    w("## What a pair is")
    w("")
    w("Each pair is one MaNGA DR17 galaxy sitting inside the virial radius of a "
      "Tempel+2014 friends-of-friends group, matched one-to-one against one MaNGA "
      "galaxy that the same friends-of-friends run left as a singleton.  The "
      "assignment is a global optimum (`scipy.optimize.linear_sum_assignment`) over "
      "the normalised distance in the matching space, restricted to pairs that lie "
      "inside a hard tolerance box on **every** matching variable simultaneously.  "
      "No galaxy is used twice within a tier.")
    w("")
    w("**Blind protection.**  The matching space, the quality gate and the "
      "field/cluster split contain no kinematic quantity of any kind.  This is "
      "asserted at run time in `code/build_matched_pairs.py` (`MATCH_FORBIDDEN`).  "
      "The cluster-versus-field kinematic contrast — the thing this sample exists to "
      "measure — has not been evaluated in this lane.")
    w("")

    w("## Declared tolerances")
    w("")
    w("Declared in code before any residual was inspected:")
    w("")
    w("| variable | symbol | declared tolerance |")
    w("|---|---|---|")
    tol = {}
    for t in TIER_ORDER:
        for k, v in s.get(t, {}).get("achieved", {}).items():
            tol[k] = v["declared_tolerance"]
    for k in FIVE + CTRL:
        if k in tol:
            w("| %s | `%s` | %g %s |" % (LBL[k], k, tol[k], UNIT[k]))
    w("")
    w("The first five are the five matching variables the brief asks for; the last "
      "three are nuisance controls (projection, bulge fraction, and physical "
      "resolution via redshift).")
    w("")

    w("## Tiers, sample sizes, and the tolerances actually achieved")
    w("")
    w("`B1_primary` is the primary sample.  The remaining tiers were declared after "
      "seeing that the primary cluster arm held only 48 galaxies — a sample-size "
      "observation, not a residual observation — and are reported separately.  They "
      "are **not** to be merged: they trade environmental contrast or morphological "
      "purity for sample size, and each answers a slightly different question.")
    w("")
    for t in TIER_ORDER:
        if t not in s:
            continue
        x = s[t]
        w("### %s — %d pairs" % (t, x["n_pairs"]))
        w("")
        w("- morphology gate: %s" % x["morphology_gate"])
        w("- environment gate: %s" % x["environment_gate"])
        w("- cluster arm %d galaxies, field arm %d galaxies"
          % (x["cluster_arm_size"], x["field_arm_size"]))
        if x["n_pairs"] == 0:
            w("")
            w("**Zero pairs.**  See the note on gas matching below.")
            w("")
            continue
        w("")
        w("| matching variable | declared tol | max abs(delta) | median abs(delta) | rms delta |")
        w("|---|---|---|---|---|")
        for k in FIVE + CTRL:
            a = x["achieved"].get(k)
            if a:
                w("| %s | %g | **%.4f** | %.4f | %.4f |"
                  % (LBL[k], a["declared_tolerance"], a["max_abs"],
                     a["median_abs"], a["rms"]))
        w("")
        bits = []
        for k, nm in (("host_sigma_v_kms", "host sigma_v [km/s]"),
                      ("R_over_Rvir", "R_proj / R_vir"),
                      ("gext_over_a0", "|g_ext| / a_0"),
                      ("psi_normal_to_host_deg", "angle(disk normal, host dir) [deg]")):
            if k in x:
                q = x[k]
                bits.append("| %s | %.3f | %.3f | %.3f |"
                            % (nm, q["min"], q["median"], q["max"]))
        if bits:
            w("Cluster-side environment of these pairs:")
            w("")
            w("| quantity | min | median | max |")
            w("|---|---|---|---|")
            for b in bits:
                w(b)
            w("")

    w("## The gas-matched tier returns zero pairs, and that is a result")
    w("")
    w("`A1_gas_matched` adds f_gas to the matching box and yields **0 pairs**.  The "
      "reason is not a catalogue gap.  Of the MaNGA galaxies covered by HI-MaNGA "
      "that sit in hosts with sigma_v >= 400 km/s, **17 of 494 are HI detections**; "
      "among late types, 14 of 156.  In the field arm the late-type detection rate "
      "is 572 of 1603.  Neutral hydrogen has been stripped out of the cluster "
      "galaxies, so the gas fraction cannot be matched between the two arms.")
    w("")
    w("This is a structural obstacle to the experiment as the brief frames it: the "
      "environment whose effect is under test has removed one of the five variables "
      "that were supposed to be held fixed.  Every tier other than A1 therefore drops "
      "f_gas and matches on stellar baryons only, carrying the HI detection flag and "
      "the HI upper limit for each galaxy so the residual gas contribution can be "
      "bounded downstream (`cl_logMHI_use`, `cl_logMHI_limit`, `cl_hi_detected` and "
      "the `fi_` equivalents).")
    w("")

    w("## The five matching variables are not five independent directions")
    w("")
    w("Correlation matrix of the matching variables across the quality-passing "
      "parent sample:")
    w("")
    c = s["_matching_variable_correlation"]
    w("| | " + " | ".join("`%s`" % k for k in FIVE) + " |")
    w("|---|" + "---|" * len(FIVE))
    for a in FIVE:
        w("| `%s` | " % a + " | ".join("%.3f" % c[a][b] for b in FIVE) + " |")
    w("")
    w("`log Sigma_b` and `log g_bar(2.2 R_d)` correlate at **r = %.4f**.  They are "
      "one matching direction, not two: g_bar of an exponential disk is Sigma_b "
      "times a shape factor that barely varies across the sample.  `f_gas` "
      "correlates with `log M_star` at r = %.3f.  The effective number of "
      "independent matching directions is about **three** (M_star, R_d, and a "
      "partially independent f_gas), not five."
      % (c["logSigma_b"]["log_gbar_2p2Rd"], c["f_gas_or_nan"]["logMstar_nsa"]))
    w("")
    w("This matters for how the result is stated.  Reporting five tight tolerances "
      "would overstate how completely the internal structure has been controlled.  "
      "Two of the five are near-deterministic functions of the other three.")
    w("")

    w("## How the host mass was derived, and what may be used as an observation")
    w("")
    w("| quantity | column | provenance | admissible as |")
    w("|---|---|---|---|")
    w("| member rms velocity | `cl_t14_grp_sigma_v` | Tempel+2014, rms radial "
      "velocity deviation of the FoF members | **observation** |")
    w("| projected clustercentric radius | `cl_t14_Rproj_kpc` | angular separation "
      "from the group luminosity centre times the angular-diameter distance | "
      "**observation** (geometry + redshift) |")
    w("| X-ray luminosity of the host | `cl_t14_mcxc_L500_1e44` | MCXC, "
      "[0.1-2.4] keV luminosity inside R500 | **observation** |")
    w("| potential-depth proxy | `log_Phi_proxy` = log sigma_v^2 | member kinematics "
      "only; no mass model | **observation up to the assumption that the galaxies "
      "trace the potential** |")
    w("| external field proxy | `cl_gext_over_a0` = (sigma_v^2 / R_proj)/a_0 | same | "
      "same |")
    w("| R_vir | `t14_grp_Rvir_Mpc` | projected harmonic mean radius of the members | "
      "geometric, but its interpretation as a virial radius is model-laden |")
    w("| M_NFW | `cl_t14_grp_MNFW_rank_only` | Tempel+2014, assumed NFW profile | "
      "**ranking only — dark-matter dependent** |")
    w("| M200 | `cl_t17_grp_M200_rank_only` | Tempel+2017, assumed NFW profile | "
      "**ranking only — dark-matter dependent** |")
    w("| M500, R500 | `*_mcxc_M500_rank_only`, `*_mcxc_R500_rank_only` | MCXC L-M "
      "scaling relation calibrated on hydrostatic masses | **ranking only — dark-"
      "matter dependent** |")
    w("")
    w("Every dark-matter-dependent column in this lane carries the literal suffix "
      "`_rank_only` in its name, in the master table and in the pair table, so it "
      "cannot be picked up as an observable by accident.")
    w("")

    w("## Disk orientation relative to the host")
    w("")
    w("`cl_t14_psi_norm_host_deg` is the angle between the galaxy's disk normal and "
      "the direction to its host centre, and `cl_t14_theta_sky_deg` is the sky-plane "
      "angle between the disk major axis and that direction.")
    w("")
    w("What is genuinely observable is the sky-projected geometry plus the "
      "inclination.  The line-of-sight offset between a galaxy and its host centre "
      "is not measurable (redshift differences are dominated by peculiar velocity, "
      "not distance), and the near side of a disk is not determined by the "
      "photometry.  psi is therefore computed as")
    w("")
    w("```")
    w("psi = arccos( sin(i) * |sin(PA_host - PA_disk)| )")
    w("```")
    w("")
    w("which assumes the galaxy-to-host offset lies in the plane of the sky, and is "
      "folded onto [0, 90] deg to absorb the unknown near side.  It is a projected "
      "lower bound on the true 3-D angle, not the 3-D angle.  Anyone using it to test "
      "a tidal-eigenvector alignment must propagate that.")
    w("")

    w("## Primary tier: the 19 pairs in full")
    w("")
    sub = p[p.tier == "B1_primary"].sort_values("cl_gext_over_a0", ascending=False)
    w("| # | cluster plateifu | field plateifu | log M* (cl/fi) | R_d kpc (cl/fi) | "
      "i deg (cl/fi) | sigma_v | R/Rvir | g_ext/a0 | psi deg | X-ray host |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, (_, r) in enumerate(sub.iterrows(), 1):
        xr = r.get("cl_t14_mcxc_name", "")
        xr = "" if (not isinstance(xr, str) or not xr.strip()) else xr
        w("| %d | %s | %s | %.2f / %.2f | %.2f / %.2f | %.0f / %.0f | %.0f | %.2f | "
          "%.2f | %.0f | %s |"
          % (i, r.cl_plateifu, r.fi_plateifu,
             r.cl_logMstar_nsa, r.fi_logMstar_nsa,
             10 ** r.cl_logRd, 10 ** r.fi_logRd,
             r.cl_incl_deg, r.fi_incl_deg,
             r.cl_t14_grp_sigma_v, r.cl_R_over_Rvir_t14,
             r.cl_gext_over_a0, r.cl_t14_psi_norm_host_deg, xr))
    w("")
    w("The other tiers are in `clean/matched_pairs.csv`; filter on the `tier` column.")
    w("")

    w("## Resolved kinematics for every galaxy in every pair")
    w("")
    w("`raw/manga/maps/` holds the DAP `MAPS-HYB10-MILESHC-MASTARSSP` file for each "
      "of the %d distinct galaxies appearing in any tier: stellar velocity field, "
      "stellar velocity dispersion, H-alpha velocity field, H-alpha dispersion, "
      "emission-line fluxes, and the elliptical polar radius map, with the matching "
      "inverse-variance and mask extensions.  See `raw/manga/maps/maps.manifest.json` "
      "for the per-file SHA-256 list."
      % len(set(p.cl_plateifu) | set(p.fi_plateifu)))
    w("")

    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("WROTE %s (%d lines)" % (OUT, len(L)))


if __name__ == "__main__":
    main()
