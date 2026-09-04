"""
JOB 2.1 + JOB 3 -- the provenance table, and the dark-matter-presupposition check.

The question Run AT left open: which mass enters the excess numerator, which
enters the x-axis radius, and do they share inputs?

The answer is stronger than "share inputs", and this module PROVES it rather
than asserting it:

  Tian+2020 Sec. 2.1 (arXiv:2001.08340v2, lines 313-357):
    "we use the CLASH lensing constraints on the total mass profile M_tot(<r) of
     each individual CLASH cluster ASSUMING A SPHERICAL NFW PROFILE ...
     For each cluster, Umetsu+2016 extracted the posterior probability
     distributions of (M_200, c_200) from the observed surface mass density
     profile assuming a spherical NFW halo ...
     In our analysis, we use these posterior distributions of the NFW parameters
     to obtain well-characterized inference of M_tot(<r|M_200,c_200)"

  Umetsu+2016 Table 3 caption (arXiv:1507.04385v4):
    "Cluster mass estimates M_3D(<r) from SINGLE SPHERICAL NFW FITS to individual
     surface mass density profiles"

So per cluster there are exactly TWO free numbers, (M200_i, c200_i), and

    g_tot(r)  = G M_NFW(<r | M200_i, c200_i) / r^2      <- the numerator
    M500c_i   = M_NFW(<R500_i | M200_i, c200_i)          <- the x-axis mass
    R500_i    = [3 M500c_i / (4 pi 500 rho_c(z_i))]^(1/3) <- the x-axis radius

The numerator and the normaliser are not correlated measurements.  They are two
functionals of ONE two-parameter fit.  This module verifies that numerically by
regenerating Tian's published log g_tot from Umetsu's Table 2 alone.
"""
from __future__ import annotations
import json
import math

import numpy as np

import ingest as I
import stats as S

KPC, MPC, MSUN, G = I.KPC, I.MPC, I.MSUN, I.G
OUT = {}


def main():
    D = I.load_all(verbose=True)
    T = I.points_table(D)
    C = D["clusters"]

    # ------------------------------------------------------------------ (1)
    # Regenerate the numerator from (M200, c200) alone.
    print("\n=== 1. does (M200,c200) alone regenerate Tian's published g_tot? ===")
    pred = np.array([np.log10(G * float(I.nfw_mass(r, C[n]["M200"], C[n]["c200"],
                                                   C[n]["z"])) * MSUN / r ** 2)
                     for n, r in zip(T["name"], T["r"])])
    obs = np.log10(T["go"])
    d = pred - obs
    lev = np.round(T["r"] / KPC, 1)
    by_level = {}
    for L in sorted(set(np.where(lev < 40, 0.0, lev))):
        m = (lev < 40) if L == 0 else (np.abs(lev - L) < 1e-6)
        by_level["BCG" if L == 0 else f"{L:.0f}kpc"] = dict(
            n=int(m.sum()), mean=float(d[m].mean()), sd=float(d[m].std()))
    OUT["numerator_regeneration"] = dict(
        n=len(d), mean_dex=float(d.mean()), sd_dex=float(d.std()),
        max_abs_dex=float(np.abs(d).max()), median_abs_dex=float(np.median(np.abs(d))),
        published_error_min_dex=float(T["e_lgt"].min()),
        published_error_median_dex=float(np.median(T["e_lgt"])),
        published_error_max_dex=float(T["e_lgt"].max()),
        ratio_sd_to_median_published_error=float(d.std() / np.median(T["e_lgt"])),
        by_level=by_level)
    print(f"  n = {len(d)}, mean {d.mean():+.4f} dex, sd {d.std():.4f} dex, "
          f"max |d| {np.abs(d).max():.4f} dex")
    print(f"  published e_log(gtot): min {T['e_lgt'].min():.3f}, median "
          f"{np.median(T['e_lgt']):.3f}, max {T['e_lgt'].max():.3f} dex")
    print(f"  -> the regeneration residual is {np.median(T['e_lgt'])/d.std():.0f}x "
          f"SMALLER than the quoted error.  The numerator IS the NFW fit.")

    # ------------------------------------------------------------------ (2)
    # R500 from the published M500c vs R500 solved off the NFW profile.
    print("\n=== 2. is R500 the same NFW fit? ===")
    rl = np.array([C[n]["R500_lens"] for n in sorted(C)])
    rn = np.array([C[n]["R500_nfw"] for n in sorted(C)])
    rel = np.abs(rn / rl - 1)
    OUT["R500_closure"] = dict(
        max_rel_diff=float(rel.max()), median_rel_diff=float(np.median(rel)),
        note="R500 from published M500c (overdensity definition) vs R500 solved "
             "directly on M_NFW(<r|M200,c200).  Agreement proves M500c is the "
             "NFW mass at R500, i.e. the SAME two numbers.")
    print(f"  R500(from published M500c) vs R500(solved on NFW): "
          f"median {100*np.median(rel):.2f}%, max {100*rel.max():.2f}%")

    # M500 closure with the overdensity definition (must be exact -- it is a
    # definition, so this is a units/cosmology check, not a physics check)
    clos = []
    for n in sorted(C):
        c = C[n]
        A = (4 / 3) * math.pi * 500 * I.rhoc(c["z"])
        clos.append(c["M500"] * MSUN / (A * c["R500_lens"] ** 3))
    OUT["R500_closure"]["overdensity_identity_max_dev"] = float(
        np.max(np.abs(np.array(clos) - 1)))
    print(f"  M500 / [(4/3)pi 500 rho_c R500^3] = 1 to "
          f"{np.max(np.abs(np.array(clos)-1)):.2e}  (definition, so this checks "
          f"units and cosmology only)")

    # ------------------------------------------------------------------ (3)
    # degrees of freedom: 84 rows carry at most 40 numbers
    print("\n=== 3. how many independent numbers are in the 84 rows? ===")
    OUT["degrees_of_freedom"] = dict(
        published_rows=84, clusters=20,
        free_numbers_in_the_lensing_chain=40,
        per_cluster_free_numbers=2,
        rows_per_cluster_min=int(min((T["name"] == n).sum() for n in C)),
        rows_per_cluster_max=int(max((T["name"] == n).sum() for n in C)),
        within_cluster_radial_shape_dof=1,
        note="Every g_tot value for cluster i is M_NFW(<r|M200_i,c200_i).  With "
             "a per-cluster level absorbing M200_i, the ENTIRE within-cluster "
             "radial shape is set by the single number c200_i.  84 rows, 20 "
             "independent radial-shape numbers.")
    print("  84 rows <- 40 numbers (20 M200 + 20 c200).")
    print("  With per-cluster levels free, the within-cluster radial shape of the")
    print("  numerator is a ONE-parameter family per cluster (c200_i).")
    print("  Effective df for the radial trend = 20, not 64.")

    # is the measured radial trend predicted by c200 alone?
    y = S.excess_y(T["gb"], T["go"])
    a = S.excess_a0(T["gb"], T["go"])
    m = T["r"] / KPC > 50
    names = sorted(C)
    per = {}
    for stat, lab in ((y, "y"), (a, "a0")):
        sl = []
        for n in names:
            mm = m & (T["name"] == n)
            if mm.sum() >= 3:
                sl.append((n, S.ols_slope(np.log10(T["r"][mm] / KPC), stat[mm])))
        c2 = np.array([C[n]["c200"] for n, _ in sl])
        sv = np.array([s for _, s in sl])
        per[lab] = dict(n=len(sl), pearson_slope_vs_c200=S.pear(np.log10(c2), sv),
                        spearman=S.spear(c2, sv), mean_slope=float(sv.mean()))
        print(f"  per-cluster radial slope of {lab}: corr with log c200 = "
              f"{S.pear(np.log10(c2), sv):+.4f} (n={len(sl)})")
    OUT["degrees_of_freedom"]["per_cluster_slope_vs_c200"] = per

    # ------------------------------------------------------------------ (4)
    # JOB 3: can the numerator be reconstructed from RAW SHEAR?
    print("\n=== 4. JOB 3 -- can the CLASH numerator be rebuilt from raw shear? ===")
    OUT["job3_admissibility"] = dict(
        numerator_is="G * M_NFW(<r | M200_i, c200_i) / r^2",
        numerator_source="Tian+2020 Sec 2.1; Umetsu+2016 Table 2 posteriors",
        quote_tian=("we use the CLASH lensing constraints on the total mass "
                    "profile M_tot(<r) of each individual CLASH cluster assuming "
                    "a spherical NFW profile"),
        quote_umetsu_t2=("Cluster parameters derived from single spherical NFW "
                         "fits to individual surface mass density profiles "
                         "reconstructed from combined strong-lensing, "
                         "weak-lensing shear and magnification measurements"),
        quote_umetsu_t3=("Cluster mass estimates M_3D(<r) from single spherical "
                         "NFW fits to individual surface mass density profiles"),
        repo_lineage_statement=(
            "runs/gravity/g4/cluster-lensing-exploration-v7.json, "
            "data_lineage.lensing: 'strong-lensing, weak-lensing shear, and "
            "magnification constraints converted by the source paper to spherical "
            "NFW Mtot posteriors and then gtot'; "
            "data_lineage.gr_model_independent_target = false"),
        raw_shear_available_in_repo=False,
        can_be_rebuilt_from_raw_shear=False,
        why=("Three separate model layers stand between the CLASH shear "
             "catalogues and the published g_tot, and NONE of them is invertible "
             "from the table: (i) the joint SL+WL+magnification reconstruction "
             "of kappa assumes GR light bending with no gravitational slip, so "
             "kappa is a GR-derived convergence map; (ii) the deprojection to "
             "M_3D(<r) assumes spherical symmetry; (iii) the profile is a "
             "TWO-PARAMETER NFW FIT, i.e. a parametric mass model of exactly the "
             "kind standing constraint 2 excludes.  The published table contains "
             "only the output of (iii).  Recovering a raw-shear numerator would "
             "require the CLASH shear catalogues and the Umetsu+2016 kappa "
             "reconstruction, neither of which is in this repo."),
        verdict="INADMISSIBLE under standing constraint 2 as currently sourced",
        # what IS published, checked table by table across the CLASH lensing papers
        published_lensing_products=[
            dict(product="M200c, c200c per cluster", source="Umetsu+2016 Table 2",
                 kind="two-parameter spherical NFW fit", admissible=False),
            dict(product="M2500c..M200m, M(<1.5Mpc)", source="Umetsu+2016 Table 3",
                 kind="the same NFW fit, evaluated at overdensity radii",
                 admissible=False),
            dict(product="M_2D(<theta), theta = 10-40 arcsec",
                 source="Umetsu+2016 Table 1",
                 kind="Zitrin+2015 parametric strong-lensing models, mass tied to "
                      "light by construction", admissible=False),
            dict(product="S/N of g_+ and n_mu per cluster",
                 source="Umetsu+2014 Table 5",
                 kind="a summary statistic, not a profile", admissible=False),
            dict(product="non-parametric kappa(R) profiles",
                 source="Umetsu+2016 Figure 'kappa'",
                 kind="shown in figures, NOT tabulated anywhere in the e-print; "
                      "and still a GR-derived convergence map",
                 admissible=False)],
        vizier_availability=dict(
            checked=["J/ApJ/821/116 (Umetsu+2016)", "J/ApJ/795/163 (Umetsu+2014)",
                     "J/ApJ/755/56 (Umetsu+2012)"],
            all_absent=True,
            meta_search="METAcat title=*Umetsu* returns only J/ApJ/890/148 "
                        "(XXL, Umetsu+2020)",
            positive_control="J/ApJ/896/70 (Tian+2020) resolves and is the table "
                             "this lane uses"),
        route_if_it_is_ever_wanted=(
            "The CLASH/Subaru Suprime-Cam shear catalogues themselves, scored the "
            "way Run AL.5 scored raw eFEDS/HSC shear -- the law predicting g_+ "
            "rather than being compared against somebody's NFW posterior.  They "
            "are not on VizieR and not in any of these e-prints."))
    for k in ("numerator_is", "raw_shear_available_in_repo",
              "can_be_rebuilt_from_raw_shear", "verdict"):
        print(f"  {k:32s} {OUT['job3_admissibility'][k]}")

    # ------------------------------------------------------------------ (5)
    # the provenance table itself
    rows = []
    for n in sorted(C):
        c = C[n]
        rows.append(dict(name=n, z=c["z"], M200_1e14=c["M200"] / 1e14,
                         c200=c["c200"], M500_1e14=c["M500"] / 1e14,
                         R500_lens_kpc=c["R500_lens"] / KPC,
                         R500_nfw_kpc=c["R500_nfw"] / KPC,
                         R500_xray_kpc=c["R500_xray"] / KPC,
                         kT_keV=c["kT"],
                         n_points=int((T["name"] == n).sum())))
    OUT["per_cluster"] = rows
    OUT["provenance_table"] = [
        dict(quantity="excess numerator  g_obs(r)",
             source="Tian+2020 fig2.dat col log(gtot)",
             derived_from="G M_NFW(<r | M200_i, c200_i) / r^2",
             root_measurement="Umetsu+2016 joint SL+WL+magnification kappa, "
                              "spherical NFW fit",
             assumes_GR="yes -- kappa from GR deflection, no slip",
             assumes_a_halo_model="yes -- NFW, 2 parameters"),
        dict(quantity="excess denominator  g_bar(r)",
             source="Tian+2020 fig2.dat col log(gbar)",
             derived_from="G [M_gas(<r) + M_star(<r)] / r^2",
             root_measurement="Donahue+2014 Chandra X-ray gas + Cooke+2016 BCG "
                              "stellar mass + Chiu+2018 satellite stars",
             assumes_GR="no", assumes_a_halo_model="no"),
        dict(quantity="x-axis radius  r",
             source="Tian+2020 fig2.dat col Rad",
             derived_from="fixed grid 14-30 kpc (BCG) and 100/200/400/600 kpc",
             root_measurement="none -- a chosen radial grid",
             assumes_GR="no", assumes_a_halo_model="no"),
        dict(quantity="x-axis normaliser  R500_i",
             source="Umetsu+2016 Table 3 M500c -> overdensity definition",
             derived_from="M_NFW(<R500 | M200_i, c200_i) = (4/3)pi 500 rho_c R500^3",
             root_measurement="THE SAME Umetsu+2016 NFW fit as the numerator",
             assumes_GR="yes", assumes_a_halo_model="yes -- the same NFW"),
        dict(quantity="independent X-ray normaliser  R500_X,i",
             source="Donahue+2014 CLASH-X Chandra JACO hydrostatic r500",
             derived_from="X-ray hydrostatic mass profile",
             root_measurement="Chandra spectroscopy; no lensing anywhere",
             assumes_GR="no (Newtonian HSE)", assumes_a_halo_model="partly -- JACO "
             "parametric v_circ model, but not tied to the lensing fit")]

    json.dump(OUT, open("provenance_results.json", "w", encoding="utf-8"), indent=1)
    print("\nwrote provenance_results.json")


if __name__ == "__main__":
    main()
