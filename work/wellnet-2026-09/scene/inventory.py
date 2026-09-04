"""inventory.py -- Job 4: the gold-cluster availability matrix.

The charter's Corpus E asks for clusters with the overlapping set

    deep optical/IR imaging, member spectroscopy, internal member-galaxy IFU,
    X-ray, SZ, wide-field weak lensing, strong-lensing image families,
    time delays, surrounding large-scale structure.

This module records, per cluster and per layer, WHAT EXISTS, WHERE IT LIVES,
and -- the classification the charter insists on -- whether it is a RAW
observation or a product DERIVED UNDER AN ASSUMED GRAVITY THEORY.

    "A GR-derived convergence map, an NFW-fitted mass, or a parametric lens
     model tying mass to light by construction is NOT a raw observation."

Provenance of the rows below:
  * layers L1 (imaging/members), L2 (BCG), L3 (ICL), L5 (X-ray), L7 (weak
    lensing), L8 (strong lensing) restate this repository's existing
    `../cluster-data/INVENTORY.md`, compiled 2026-09-04;
  * layers L4 (member IFU), L6 (SZ), L9 (time delays), L10 (environment) are
    NEW, acquired for this lane on 2026-09-04 by three archive sweeps whose
    method notes are recorded in `ACQUISITION_NOTES`.

NOTHING HERE OPENS PIXEL OR SHEAR DATA, AND NO GRAVITY-RELEVANT STATISTIC IS
COMPUTED ON ANY CLUSTER.  Row counts, column lists and catalogue identifiers
are metadata echoed from response headers.  This is an inventory.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------- statuses
RAW_MR = "RAW_MACHINE_READABLE"       # raw observable, tabulated, downloadable
RAW_ARXIV = "RAW_ARXIV_LATEX_ONLY"    # raw, but only inside a paper's source
RAW_PIXELS = "RAW_PIXELS_ONLY"        # raw, but no tabulated product exists
DERIVED = "DERIVED_UNDER_THEORY"      # presupposes a gravity/mass model
PARTIAL = "PARTIAL"                   # exists but materially weaker than asked
ABSENT = "ABSENT"                     # confirmed absent; `searched` says how

#: Statuses that count as "this layer EXISTS for this cluster".
#: RAW_PIXELS_ONLY is included deliberately: a public calibrated Compton-y map
#: or MUSE mosaic is real data that the charter's Corpus E asks for, even
#: though it needs a reduction pipeline before it becomes a scene input.  The
#: `n_layers_raw_tabulated` column keeps that distinction visible.
#: DERIVED_UNDER_THEORY is excluded: a product of the theory under test is not
#: evidence about the theory.
USABLE = {RAW_MR, RAW_ARXIV, RAW_PIXELS, PARTIAL}

CLUSTERS = (
    "Abell 2744", "MACS J0416.1-2403", "MACS J0717.5+3745",
    "MACS J1149.5+2223", "Abell S1063", "Abell 370", "Abell 2029",
)

#: the charter's Corpus E layer list, plus the two components the charter's
#: root-data table names separately (BCG, intracluster light)
LAYERS = {
    "L1_imaging_members": "Deep imaging + member catalogue with resolved "
                          "structural parameters",
    "L2_bcg": "Central galaxy, resolved separately from the members",
    "L3_icl": "Intracluster light",
    "L4_member_ifu": "Internal member-galaxy IFU kinematics",
    "L5_xray": "Hot gas: X-ray density and temperature",
    "L6_sz": "SZ signal",
    "L7_weak_lensing": "Wide-field weak lensing (per-source shapes)",
    "L8_strong_lensing": "Strong-lensing image families",
    "L9_time_delays": "Strong-lensing time delays",
    "L10_environment": "Surrounding large-scale structure",
}

#: Which ontology quantities each layer supplies.  This is what makes the
#: inventory consumable by `bridge.s8_available`: a candidate law that reads
#: `time_delay` can be checked against the clusters that actually have one.
LAYER_QUANTITIES = {
    "L1_imaging_members": ("x", "y", "m_star", "r_e", "sersic_n",
                           "axis_ratio_q", "position_angle", "p_member",
                           "rho_star", "M_enc"),
    "L2_bcg": ("m_star", "r_e", "sersic_n"),
    "L3_icl": ("m_icl",),
    "L4_member_ifu": ("sigma_star",),
    "L5_xray": ("n_e", "T_x", "m_gas_hot", "P_e"),
    "L6_sz": ("y_compton", "P_e"),
    "L7_weak_lensing": ("e1", "e2", "shear_m", "psf_fwhm"),
    "L8_strong_lensing": ("image_position",),
    "L9_time_delays": ("time_delay",),
    "L10_environment": ("rho_env", "ext_axis", "alignment_angle",
                        "path_density", "path_void_fraction", "n_wells",
                        "graph_degree"),
}

#: every layer supplies these regardless
UNIVERSAL_QUANTITIES = ("v_los", "z", "r_proj", "r_3d", "t", "distance",
                        "G", "c_light", "a0", "upsilon_star", "mass",
                        "smoothing_scale", "R500", "g_N", "g_total", "g_vec",
                        "phi_depth_saddle", "phi_depth_r500",
                        "phi_depth_scaleradius", "phi_depth_volume",
                        "tidal_tensor", "tidal_anisotropy", "v_x", "v_y",
                        "v_z", "v_circ", "kappa", "m_bh", "m_hi", "m_h2",
                        "sigma_turb", "t_since_merger", "phi_lensing",
                        "phi_slip", "vacuum_order", "vacuum_axis",
                        "field_memory")


def _c(status, what, where, raw, note="", searched="") -> Dict[str, Any]:
    return {"status": status, "what": what, "where": where,
            "raw_or_derived": raw, "note": note, "searched": searched}


# ======================================================= THE MATRIX
MATRIX: Dict[str, Dict[str, Dict[str, Any]]] = {}


def _row(cluster, **cells):
    MATRIX[cluster] = cells


_row("Abell 2744",
     L1_imaging_members=_c(
         RAW_MR, "225 members with full Sersic fits in 7 HST bands",
         "VizieR J/A%2BA/709/A254/a2744 (Granata+2026), 225 rows x 88 cols",
         "RAW imaging; Sersic parameters are a MODEL FIT to the light, not a "
         "mass model -- no gravity assumption"),
     L2_bcg=_c(PARTIAL, "total fluxes in up to 17 bands; no light profile",
               "bcg/shipley2018_hff_bCG_photometry.tsv (391 rows)",
               "RAW photometry",
               "the HFF-DeepSpace construction MODELS AND SUBTRACTS bCG+ICL "
               "before measuring other sources, so even the total is not clean"),
     L3_icl=_c(RAW_MR, "f_ICL under 5 definitions (Montes & Trujillo 2018) "
                       "plus CICLE wavelet decomposition",
               "published tables", "RAW surface photometry"),
     L4_member_ifu=_c(
         RAW_ARXIV, "76 single-aperture stellar velocity dispersions from "
                    "MUSE pPXF (1.5 arcsec aperture, corrected to R_e/8)",
         "arXiv:2603.26869 e-print, aa.tex Appendix C, table tab:table2744; "
         "7 cols: ID, RA, Dec, ID_MUSE, sigma, dsigma, S/N",
         "RAW spectroscopic",
         "NOT resolved maps -- one number per galaxy.  NOT at CDS: the paper's "
         "data-availability statement covers only the Appendix B structural "
         "tables, so a VizieR-only search records this layer as absent."),
     L5_xray=_c(RAW_MR, "n_e(r) and T(r) profiles", "cluster-data/gas/",
                "RAW X-ray"),
     L6_sz=_c(RAW_PIXELS, "Bolocam 140 GHz calibrated y map, ~1.4 arcmin "
                          "effective resolution, ~10-14 arcmin field; plus "
                          "PSZ2 G008.94-81.22 (1.41') and ACT DR5/DR6",
              "IRSA BOXSZ ABELL_2744.tgz; VizieR J/A%2BA/594/A27/psz2; "
              "J/ApJS/253/3/clusters",
              "RAW for the y map, ACT y0/fy0 and the SZ significances; "
              "DERIVED for Y5R500 and M_SZ",
              "Y5R500 integrates inside 5xR500 where R500 comes from an "
              "assumed GNFW template and the Y-M relation -- theory-contaminated"),
     L7_weak_lensing=_c(
         ABSENT, "no public per-source shear catalogue and no public binned "
                 "shear profile table", "--", "--",
         "the binned profiles of Medezinski 2016, Gruen 2013 and Umetsu "
         "2014/2016 exist only as FIGURES; what those papers tabulate is NFW "
         "masses, which presuppose a dark-matter halo",
         "checked HLSP, BUFFALO (announced arXiv:2602.06904 'upon acceptance', "
         "not on the HLSP), and the four published lensing analyses"),
     L8_strong_lensing=_c(RAW_MR, "identified multiple-image families with "
                                  "spectroscopic source redshifts",
                          "cluster-data/stronglensing/", "RAW image astrometry",
                          "the FAMILY ASSIGNMENT is sometimes model-informed; "
                          "carry that distinction"),
     L9_time_delays=_c(
         ABSENT, "no multiply-imaged variable source with a measured delay",
         "--", "--",
         "heavy JWST multi-epoch transient coverage exists (GLASS ERS, "
         "UNCOVER, MAGNIF, ALT, SAPPHIRES, PLATE) and has produced no measured "
         "cluster-scale delay",
         "the complete census of measured cluster-scale delays is MACS J1149 "
         "(Refsdal), PLCK G165.7+67.0 (SN H0pe), MACS J0138-2155 "
         "(Encore/Requiem) and three cluster-lensed quasars"),
     L10_environment=_c(
         PARTIAL, "Owers+2011 AAOmega spectroscopy to R = 15.0 arcmin = "
                  "4.1 Mpc; degree-scale photo-z only beyond that",
         "VizieR J/ApJ/728/27/table5 (1509 rows); VizieR VII/292 (Legacy "
         "Surveys DR8 photo-z)",
         "RAW redshifts; photo-z DERIVED (SED template fit, no gravity "
         "assumption)",
         "SDSS: 0 rows within 60 arcmin.  DESI EDR and DR1: 0 rows.  Tempel"
         "+2014 filaments: 0 galaxies within 0.5 deg.  4.1 Mpc is about one "
         "virial radius -- not enough for an external tidal axis.",
         "SDSS DR18 SpecObj cone (control Coma 1559 rows), DESI V/161 cone, "
         "Tempel J/MNRAS/438/3465 (control Coma 352)"))

_row("MACS J0416.1-2403",
     L1_imaging_members=_c(RAW_MR, "224 members with full Sersic fits",
                           "VizieR J/A%2BA/709/A254/m0416, 224 rows x 88 cols",
                           "RAW imaging"),
     L2_bcg=_c(PARTIAL, "total fluxes; DeMaio+2018 adds aperture luminosity "
                        "and stellar mass inside r < 10/50/100 kpc",
               "bcg/shipley2018 + DeMaio 2018", "RAW photometry",
               "no light profile for any HFF BCG"),
     L3_icl=_c(RAW_MR, "f_ICL, two independent methods", "published tables",
               "RAW"),
     L4_member_ifu=_c(
         RAW_ARXIV, "52 aperture sigma (Granata+2026) plus 49 independent "
                    "aperture sigma (Bergamini+2019)",
         "arXiv:2603.26869 tab:table0416; arXiv:1905.13236 "
         "table:cluster_sample", "RAW spectroscopic",
         "two independent pPXF measurements of the same cluster is the only "
         "such cross-check in the sample"),
     L5_xray=_c(ABSENT, "no published radial n_e(r) or T(r) anywhere", "--",
                "--", "Bolocam SZ is what keeps this cluster in the sample",
                "Andrade-Santos 2017/2021 (0 hits across 4 tables / 538 rows), "
                "Donahue 2014 CLASH-X (JACO profiles exist only as figures), "
                "CLASH HLSP (optical only), Mantz WtG, CHEX-MATE; eRASS1 has "
                "it but global-only with an empty KT column from a 143 s "
                "exposure giving 144 counts inside R500"),
     L6_sz=_c(RAW_PIXELS, "Bolocam y map; PSZ2 G221.06-44.05 at 0.80 arcmin; "
                          "ACT DR5 0.23', DR6 0.19'",
              "IRSA BOXSZ MACSJ0416.tgz; J/A%2BA/594/A27/psz2",
              "RAW y map and ACT y0; DERIVED Y5R500/M_SZ",
              "PSZ1 does NOT contain it (nearest source 209 arcmin) while PSZ2 "
              "does -- the two Planck catalogues are complementary, not nested"),
     L7_weak_lensing=_c(ABSENT, "no public per-source catalogue", "--", "--",
                        "", "as Abell 2744"),
     L8_strong_lensing=_c(RAW_MR, "image families with spectroscopic redshifts",
                          "cluster-data/stronglensing/", "RAW"),
     L9_time_delays=_c(ABSENT, "none", "--", "--", "", "full delay census"),
     L10_environment=_c(
         RAW_MR, "CLASH-VLT VIMOS spectroscopy, 4386-4391 rows, reaching "
                 "R = 17.2 arcmin = 5.5 Mpc (~2.2 r200); second independent "
                 "catalogue Caminha+2017 MUSE+VIMOS, 4717 rows",
         "VizieR J/ApJS/224/33/table2 (Balestra+2016); J/A%2BA/600/A90",
         "RAW redshifts",
         "the widest DEDICATED spectroscopy of any Frontier Fields target, but "
         "SDSS and DESI both return literally zero rows here"))

_row("MACS J0717.5+3745",
     L1_imaging_members=_c(
         PARTIAL, "q and theta from SExtractor moments only; NO R_e, NO n",
         "members/HFFDS_macs0717clu_v3.9.*, Molino+2017", "RAW imaging",
         "Granata+2026 deliberately excludes it -- a quadruple merger with no "
         "single anchoring BCG; no substitute Sersic catalogue exists",
         "VizieR description and positional-metadata searches, arXiv searches"),
     L2_bcg=_c(PARTIAL, "total fluxes only", "bcg/shipley2018", "RAW"),
     L3_icl=_c(RAW_MR, "f_ICL, two methods", "published tables", "RAW"),
     L4_member_ifu=_c(
         ABSENT, "no IFU coverage of the cluster core", "--", "--",
         "its spectroscopy is Keck DEIMOS + LRIS and Gemini GMOS MULTI-OBJECT "
         "SLIT (1079 redshifts, 537 members) -- slit, not integral field",
         "enumerated the four tables in J/A%2BA/709/A254 (a2744, as1063, "
         "m0416, m1149 only); searched the MUSE cluster-core literature"),
     L5_xray=_c(RAW_MR, "n_e(r), T(r)", "cluster-data/gas/", "RAW"),
     L6_sz=_c(RAW_PIXELS, "Bolocam y map; PSZ2 G180.25+21.03 at 1.16 arcmin",
              "IRSA BOXSZ MACSJ0717.tgz", "RAW y map",
              "OUT OF THE ACT FOOTPRINT (Dec +37.76 against a measured ACT DR6 "
              "maximum of +20.796) and absent from SPT -- Bolocam is the only "
              "resolved SZ product for this cluster"),
     L7_weak_lensing=_c(ABSENT, "no public per-source catalogue", "--", "--",
                        "", "as Abell 2744"),
     L8_strong_lensing=_c(RAW_MR, "image families",
                          "cluster-data/stronglensing/", "RAW"),
     L9_time_delays=_c(ABSENT, "none", "--", "--", "", "full delay census"),
     L10_environment=_c(
         PARTIAL, "DESI DR1 gives 2392 redshifts within 0.5 deg, 84 of them in "
                  "the cluster redshift slice; SDSS has 233 within 60 arcmin "
                  "and ZERO in the slice",
         "VizieR V/161 (DESI DR1 zcat, 28,425,963 rows); SDSS DR18 SpecObj",
         "RAW redshifts",
         "FOOTPRINT PRESENCE IS NOT COVERAGE: the SDSS main sample is far too "
         "shallow at z=0.545, so all 233 SDSS spectra are foreground.  A raw "
         "row count would have scored this cluster as covered."))

_row("MACS J1149.5+2223",
     L1_imaging_members=_c(RAW_MR, "279 members with full Sersic fits",
                           "VizieR J/A%2BA/709/A254/m1149, 279 rows x 88 cols",
                           "RAW imaging"),
     L2_bcg=_c(PARTIAL, "total fluxes; DeMaio+2018 aperture masses",
               "bcg/shipley2018 + DeMaio 2018", "RAW"),
     L3_icl=_c(RAW_MR, "f_ICL, two methods", "published tables", "RAW"),
     L4_member_ifu=_c(RAW_ARXIV, "51 aperture stellar velocity dispersions",
                      "arXiv:2603.26869 tab:table1149", "RAW spectroscopic",
                      "single-aperture, not resolved maps"),
     L5_xray=_c(RAW_MR, "n_e(r), T(r)", "cluster-data/gas/", "RAW"),
     L6_sz=_c(RAW_PIXELS, "Bolocam y map; PSZ2 G228.16+75.20 at 0.53 arcmin",
              "IRSA BOXSZ MACSJ1149.6.tgz", "RAW y map",
              "out of the ACT footprint (Dec +22.40 > +20.796) and absent from "
              "SPT"),
     L7_weak_lensing=_c(ABSENT, "no public per-source catalogue", "--", "--",
                        "", "as Abell 2744"),
     L8_strong_lensing=_c(RAW_MR, "image families including the SN Refsdal host",
                          "cluster-data/stronglensing/", "RAW"),
     L9_time_delays=_c(
         RAW_ARXIV,
         "SN Refsdal: four measured delays. SX-S1 = 376.02 d "
         "(68% interval +5.6/-5.5); S2-S1 = 9.69 d; S3-S1 = 7.92 d; "
         "S4-S1 = 19.44 d.  Combined across four independent light-curve "
         "algorithms (Polyn., GPR, SNTD, PyCS).",
         "Kelly+2023 ApJ 948 93, arXiv:2305.06377 e-print, ms.tex table "
         "tab:measurementsdelaycorr",
         "RAW: a light-curve cross-correlation, theory-free",
         "NOT on VizieR and NOT machine-readable: every cell of the published "
         "table is a \\def macro resolved elsewhere in ms.tex, so extracting "
         "the table alone returns no numerals"),
     L10_environment=_c(
         PARTIAL, "SDSS 1168 redshifts within 60 arcmin, 150 in the cluster "
                  "slice (= 23 Mpc reach); DESI DR1 adds 508 within 0.5 deg, "
                  "22 in slice",
         "SDSS DR18 SpecObj; VizieR V/161", "RAW redshifts",
         "marginal: 150 in-slice tracers over 23 Mpc is sparse for a filament "
         "or tidal-axis reconstruction"))

_row("Abell S1063",
     L1_imaging_members=_c(RAW_MR, "222 members with full Sersic fits",
                           "VizieR J/A%2BA/709/A254/as1063", "RAW imaging"),
     L2_bcg=_c(PARTIAL, "total fluxes; DeMaio+2018 aperture masses",
               "bcg/shipley2018 + DeMaio 2018", "RAW"),
     L3_icl=_c(RAW_MR, "f_ICL, CICLE and surface-brightness cuts",
               "de Oliveira+2022, Montes & Trujillo 2018", "RAW"),
     L4_member_ifu=_c(RAW_ARXIV, "34 aperture sigma (Granata+2026) plus 37 "
                                 "independent (Bergamini+2019)",
                      "arXiv:2603.26869 tab:table2248; arXiv:1905.13236",
                      "RAW spectroscopic", "two independent measurements"),
     L5_xray=_c(RAW_MR, "n_e(r), T(r)", "cluster-data/gas/", "RAW"),
     L6_sz=_c(RAW_PIXELS, "Bolocam y map; PSZ2 G349.46-59.95; ACT DR6 at 0.29 "
                          "arcmin, SNR 83.4; SPT SZ J2248-4431",
              "IRSA BOXSZ ABELL_S1063.tgz", "RAW y map and ACT/SPT y0",
              "the best-covered SZ target: Bolocam + ACT + SPT + Planck"),
     L7_weak_lensing=_c(ABSENT, "no public per-source catalogue", "--", "--",
                        "", "as Abell 2744"),
     L8_strong_lensing=_c(RAW_MR, "image families", "cluster-data/stronglensing/",
                          "RAW"),
     L9_time_delays=_c(ABSENT, "none", "--", "--", "", "full delay census"),
     L10_environment=_c(
         PARTIAL, "CLASH-VLT 25x25 arcmin (Mercurio+2021), ~3850 redshifts, "
                  "R ~ 17.7 arcmin = 5.2 Mpc",
         "CLASH-VLT project site -- NOT served by VizieR",
         "RAW redshifts",
         "SDSS 0 rows, DESI 0 rows, Tempel filaments 0 galaxies.  "
         "Pan-STARRS also absent (Dec -44.5).",
         "J/A%2BA/656/A147 returns a clean 'Table or Catalog not found' on the "
         "CfA mirror; a METAcat title search for *CLASH-VLT* returns exactly "
         "one hit, J/ApJS/224/33, which is MACS J0416"))

_row("Abell 370",
     L1_imaging_members=_c(
         PARTIAL, "870 members with q and theta from SExtractor moments only; "
                  "NO R_e, NO n",
         "weaklensing/buffalo_a370/...galcat-redseq.cat", "RAW imaging",
         "the ONLY target with a raw shear catalogue is also one of the two "
         "without resolved Sersic parameters"),
     L2_bcg=_c(PARTIAL, "total fluxes only", "bcg/shipley2018", "RAW"),
     L3_icl=_c(RAW_MR, "f_ICL, CICLE and surface-brightness cuts",
               "de Oliveira+2022", "RAW"),
     L4_member_ifu=_c(
         RAW_PIXELS, "MUSE mosaic covering > 14 arcmin^2 exists, but no "
                     "published stellar-kinematics catalogue",
         "ESO archive (Lagattuta+2017; Pilot-WINGS 2022)", "RAW pixels",
         "the raw IFU data exist; the derived kinematics do not",
         "VizieR J/MNRAS/469/3946 verified real (exact #Name echo, "
         "CatalogsExamined 0): 3 tables, 167 rows, columns ID/RA/Dec/z/"
         "F435W/F606W/F814W/Type -- grep for a dispersion column returns 0 "
         "hits.  The Pilot-WINGS master catalogue is positions + redshifts + "
         "7-band photometry only."),
     L5_xray=_c(RAW_MR, "n_e(r), T(r)", "cluster-data/gas/", "RAW"),
     L6_sz=_c(RAW_PIXELS, "Bolocam y map; PSZ2 G172.98-53.55; ACT DR6 0.13'",
              "IRSA BOXSZ ABELL_0370.tgz", "RAW y map and ACT y0"),
     L7_weak_lensing=_c(
         RAW_MR, "18,556 per-source shape measurements to 6.2 Mpc",
         "weaklensing/ (BUFFALO A370)",
         "RAW: measured ellipticities, weights and a PSF model",
         "THE ONLY public raw cluster shear catalogue among the seven"),
     L8_strong_lensing=_c(RAW_MR, "image families", "cluster-data/stronglensing/",
                          "RAW"),
     L9_time_delays=_c(ABSENT, "none", "--", "--",
                       "a dedicated published supernova search behind its "
                       "multiply-imaged galaxies produced no delay",
                       "full delay census"),
     L10_environment=_c(
         PARTIAL, "SDSS 3249 redshifts within 60 arcmin (99 in the cluster "
                  "slice, ~18.6 Mpc reach) and 13,903 within 120 arcmin; "
                  "DESI DR1 adds 1545 within 0.5 deg, 117 in slice",
         "SDSS DR18 SpecObj; VizieR V/161; VII/292 photo-z; II/371 DES DR2; "
         "II/349 Pan-STARRS DR1",
         "RAW redshifts and photometry; photo-z DERIVED (SED fit)",
         "the second-best environment case after A2029, and the only target "
         "that combines a raw shear catalogue with usable wide spectroscopy"))

_row("Abell 2029",
     L1_imaging_members=_c(
         PARTIAL, "1054 members; Sersic n, q, theta, R_e for 388 of them "
                  "(36.8%)",
         "members/A2029_members_Sohn2019_x_Simard2011_structural.csv",
         "RAW imaging",
         "Simard+2011 fits only the SDSS SPECTROSCOPIC sample (complete to "
         "r=17.77) while Sohn+2019 reaches r=21.3 with MMT/Hectospec, so the "
         "63% unmatched are real members Simard never fitted -- a selection "
         "effect, not a cross-match failure (matched pairs agree to a median "
         "0.026 arcsec).  These are GROUND-BASED SDSS fits at ~1.4 arcsec "
         "seeing and are NOT on the same footing as the HST fits; do not pool."),
     L2_bcg=_c(RAW_MR, "three independent products for IC 1101: Kluge+2020 "
                       "single-Sersic (n=5.55+-0.26, r_e=261 arcsec, reaching "
                       "mu=30 g' mag/arcsec^2), Donzelli+2011, and Lauer+2014 "
                       "curve of growth plus sigma* = 386 km/s",
               "bcg/", "RAW surface photometry",
               "the only target with a resolved BCG light profile"),
     L3_icl=_c(ABSENT, "no published ICL measurement exists", "--", "--", "",
               "literature search"),
     L4_member_ifu=_c(
         ABSENT, "no member IFU kinematics", "--", "--",
         "not in SAMI's 8 clusters; not in Granata+2026 or Bergamini+2019; no "
         "MUSE member-kinematics catalogue.  The only conceivable route is the "
         "MaNGA BCG ancillary (128 X-ray clusters, z=0.02-0.15) possibly "
         "covering IC 1101 -- searched and NOT confirmed.",
         "SAMI DR3 cluster list, Granata/Bergamini tables, MaNGA ancillaries"),
     L5_xray=_c(RAW_MR, "X-COP: 0.7-1.2 keV count mosaics, exposure and "
                        "background maps, density and spectral results",
               "X-COP Data Release, isdc.unige.ch/~deckert/XCOP/",
               "RAW counts; the hydrostatic mass products are DERIVED"),
     L6_sz=_c(
         RAW_MR,
         "X-COP measured Planck/MILCA Compton-y RADIAL PROFILE with its full "
         "bin-bin covariance matrix, plus a pressure profile with covariance; "
         "also PSZ2 G006.49+50.56, ACT DR5 at 0.02 arcmin",
         "A2029_Y-PROF-COVMAT_P-PROF-COVMAT.20170830.fits",
         "RAW for Y-PROF + covariance.  The pressure profile is DERIVED but "
         "only GEOMETRICALLY (Abel deprojection, spherical symmetry, a "
         "temperature to convert y to P) -- it does NOT presuppose dark matter.",
         "the strongest SZ asset in the whole inventory: a y PROFILE with "
         "covariance rather than an integrated Y5R500.  A2029 is also the one "
         "primary Bolocam omits."),
     L7_weak_lensing=_c(ABSENT, "no public per-source catalogue", "--", "--",
                        "", "as Abell 2744"),
     L8_strong_lensing=_c(ABSENT, "no strong-lensing image families", "--",
                          "--", "a low-redshift relaxed cluster; not a "
                          "strong-lensing target", "literature search"),
     L9_time_delays=_c(ABSENT, "none", "--", "--", "", "full delay census"),
     L10_environment=_c(
         RAW_MR,
         "SDSS 12,604 redshifts within 180 arcmin of which 2289 are in the "
         "cluster slice (= 15.8 Mpc); DESI DR1 adds 2348 within 0.5 deg (454 "
         "in slice); Sohn+2019b dedicated survey to R = 99.6 arcmin = 8.7 Mpc; "
         "Tempel+2014 published filament field through the position (261 "
         "galaxies within 0.5 deg)",
         "SDSS DR18; VizieR V/161; J/ApJ/872/192/table1 (1215 rows x 20 cols); "
         "J/MNRAS/438/3465 (Tempel+2014)",
         "RAW redshifts.  The Sohn+2019b MEMBERSHIP column is DERIVED "
         "(caustic/phase-space assignment presumes a dynamical mass model) and "
         "the Tempel filament field is DERIVED (a Bisous marked point process "
         "on the redshift-space galaxy field).  Use the redshifts, not the "
         "labels.",
         "four independent, mutually checkable environment layers -- the only "
         "target where an external tidal axis can actually be reconstructed"))


# ================================================================ analysis
def availability(quantity: str) -> Dict[str, bool]:
    """Which clusters supply this ontology quantity?  Feeds bridge.s8."""
    if quantity in UNIVERSAL_QUANTITIES:
        return {c: True for c in CLUSTERS}
    out = {}
    for c in CLUSTERS:
        ok = False
        for layer, qs in LAYER_QUANTITIES.items():
            if quantity in qs and MATRIX[c][layer]["status"] in USABLE:
                ok = True
                break
        out[c] = ok
    return out


def availability_index(quantities: Sequence[str]) -> Dict[str, Dict[str, bool]]:
    return {q: availability(q) for q in quantities}


def layer_counts() -> Dict[str, Dict[str, int]]:
    out = {}
    for layer in LAYERS:
        c = {}
        for cl in CLUSTERS:
            s = MATRIX[cl][layer]["status"]
            c[s] = c.get(s, 0) + 1
        out[layer] = c
    return out


def cluster_scores() -> List[Dict[str, Any]]:
    """How many Corpus-E layers does each cluster actually have?"""
    rows = []
    for cl in CLUSTERS:
        got = [l for l in LAYERS if MATRIX[cl][l]["status"] in USABLE]
        raw = [l for l in LAYERS
               if MATRIX[cl][l]["status"] in (RAW_MR, RAW_ARXIV)]
        missing = [l for l in LAYERS if MATRIX[cl][l]["status"] == ABSENT]
        rows.append({"cluster": cl, "n_layers_usable": len(got),
                     "n_layers_raw_tabulated": len(raw),
                     "n_layers_absent": len(missing),
                     "usable": got, "absent": missing})
    return sorted(rows, key=lambda r: -r["n_layers_usable"])


def gold_cluster_verdict() -> Dict[str, Any]:
    """Does ANY cluster meet the charter's Corpus E specification?"""
    need = list(LAYERS)
    full = [c for c in CLUSTERS
            if all(MATRIX[c][l]["status"] in USABLE for l in need)]
    # which layer is the binding constraint?
    per_layer_absent = {l: [c for c in CLUSTERS
                            if MATRIX[c][l]["status"] == ABSENT]
                        for l in need}
    binding = sorted(per_layer_absent.items(), key=lambda kv: -len(kv[1]))
    return {"n_layers_required": len(need),
            "clusters_meeting_all": full,
            "corpus_E_satisfied": bool(full),
            "absent_by_layer": {k: v for k, v in binding},
            "binding_constraints": [k for k, v in binding if len(v) >= 5],
            "best": cluster_scores()[0]}


def dm_contaminated_products() -> List[Dict[str, str]]:
    """Every catalogued product that presupposes a gravity theory."""
    out = []
    for c in CLUSTERS:
        for l, cell in MATRIX[c].items():
            r = cell["raw_or_derived"]
            if "DERIVED" in r.upper():
                out.append({"cluster": c, "layer": l,
                            "classification": r, "note": cell["note"]})
    return out


ACQUISITION_NOTES = {
    "method": "VizieR ASU (TAP returns 403). Every pull asserted the "
              "row count AND the column list, passed -out.all=1 explicitly, "
              "percent-encoded '+' as %2B, and checked both fuzzy-fallback "
              "detectors (#Name echo, CatalogsExamined). No cone-search null "
              "was trusted; all coordinate matching was done numerically "
              "against fully downloaded tables with a positive control.",
    "new_traps_found": [
        "The VizieR fuzzy-fallback trap is MIRROR-DEPENDENT. The same bad ID "
        "J/A%2BA/621/A41 serves 5.9 MB of an unrelated real catalogue at HTTP "
        "200 from CDS but returns a clean 'Error=Table or Catalog not found' "
        "from vizier.cfa.harvard.edu. CfA is not merely faster, it is more "
        "truthful on bad IDs -- use it for existence tests.",
        "A SILENT ALIASED CATALOGUE that the CatalogsExamined detector cannot "
        "see: -source=J/A%2BA/590/A30 returns #Name: J/A+A/590/A31, a "
        "DIFFERENT PAPER, with CatalogsExamined=0 and no error. The #Name "
        "echo is the only detector that catches it.",
        "#Name: echoes the PARENT for a subtable request "
        "(J/A%2BA/594/A27/psz2 -> #Name: J/A+A/594/A27), so the echo test must "
        "be a PREFIX test; exact string equality false-positives.",
        "NOIRLab Data Lab TAP returns errors as HTTP 200 VOTABLE. A wrong "
        "column name (target_ra; the real one is mean_fiber_ra) produced "
        "<INFO name='QUERY_STATUS' value='ERROR'> inside a normal 200 and a "
        "line-based parse rendered it as an empty count for every cluster "
        "INCLUDING the positive control. Grep for QUERY_STATUS ERROR first.",
        "A subtable that looks like the main catalogue can hold a handful of "
        "rows: J/ApJS/247/25/table10 has 18 rows while the real SPT-ECS "
        "catalogue is /cand with 470. Enumerate subtables via METAtab before "
        "querying, and note that METAtab's `records` column sits after a "
        "free-text comment column so it must be requested explicitly.",
        "A cone search on a table with no sky columns returns 0 rows for EVERY "
        "position including the positive control: Tempel+2014 table2 carries "
        "only Cartesian x,y,z. Without a control this reads as 'no filaments "
        "anywhere in the sky'.",
        "Sexagesimal coordinate columns coerce silently: three catalogues "
        "serve RAJ2000 as 'h:m:s' strings, and a naive numeric cast gave "
        "RA=0.0000, Dec=-30.0000 and a fabricated 187 arcmin survey extent -- "
        "a plausible wrong number, not a crash.",
        "FOOTPRINT PRESENCE IS NOT COVERAGE. MACS J0717 has 233 SDSS spectra "
        "within 60 arcmin and ZERO at the cluster redshift; the SDSS main "
        "sample is far too shallow at z=0.545. A row count alone scores it as "
        "covered.",
        "A published data-availability statement can be narrower than the "
        "paper. Granata+2026 deposits only its Appendix B structural tables at "
        "CDS; the 213 velocity dispersions are Appendix C and exist only in the "
        "arXiv LaTeX source. A VizieR-only inventory records this layer as "
        "absent.",
        "Time-delay tables can contain no numerals at all: every cell of "
        "Kelly+2023's delay table is a \\def macro resolved elsewhere in the "
        "manuscript source.",
    ],
    "sealed": "KiDS and the wide binaries were not loaded, looked at, or "
              "queried at any point in this lane.",
}
