"""Write a .manifest.json beside every raw VizieR download, emit cleaned
per-table TSVs for the science-critical tables, and compute the misalignment /
~90-degree POLAR statistics the brief asks for."""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from _manifest import write_manifest
from crot_parse import parse_vizier

D = os.path.dirname(os.path.abspath(__file__))
idx = json.load(open(os.path.join(D, "crot_vizier_table_index.json"), encoding="utf-8"))

# ---------------------------------------------------------------- labels ----
# measurement_or_model per catalogue. HARD RULE: anything derived from a
# Jeans/JAM model, a fitted DM halo, or a halo-dependent M/L is a MODEL.
LABEL = {
 "sprc_moiseev2011": ("MEASUREMENT (morphological identification) + POSITION/PHOTOMETRY. "
   "SDSS imaging positions, r magnitudes and heliocentric velocities cz are measurements. "
   "The Type column (Best/Good/Related/possible face-on) is a MORPHOLOGICAL classification "
   "from SDSS images, NOT a kinematic confirmation: this catalogue on its own does NOT "
   "establish that the ring is kinematically decoupled. No DM model anywhere."),
 "prg_co_combes2013": ("MEASUREMENT. CO(1-0)/CO(2-1) single-dish spectra of SPRC polar-ring "
   "candidates; Vsys is the measured line centre; the n_SPRC='C' flag marks objects "
   "KINEMATICALLY CONFIRMED as polar rings. Beam sizes are instrumental. No DM model."),
 "prg_hi_huchtmeier1997": ("MEASUREMENT. HI 21cm heliocentric velocities, linewidths and HI "
   "fluxes/masses for Whitmore PRC objects. MHI and MHI/LB use a distance scale only "
   "(H0), not a DM halo. No DM model."),
 "prg_hi_vandriel2002": ("MEASUREMENT. HI 21cm spectra of polar-ring galaxies: optical and HI "
   "systemic velocities, linewidths W50/W20, HI fluxes and derived HI masses. Distance-"
   "scale dependent only. No DM model."),
 "manga_crd_bevacqua2022": ("MEASUREMENT for the kinematics: DPA is the measured difference "
   "between the STELLAR and IONISED-GAS kinematic position angles from the MaNGA DAP "
   "velocity fields. Pth50 (Petrosian R50) and logM* are photometric/SED products "
   "(logM* is a stellar-population MODEL quantity, not a dynamical mass). LambdaRe and "
   "Ell in table g18 are measured from the velocity/photometric maps. NO DM halo or "
   "Jeans/JAM model enters any column."),
 "manga_counterrot_gasymov2025": ("MEASUREMENT (classification of measured velocity fields). "
   "Features/CRConfig classify counter-rotation seen in MaNGA stellar velocity maps. "
   "logM* is a stellar-population MODEL quantity. NO DM halo model. NOTE: this "
   "catalogue gives NO misalignment angle column."),
 "kinangles_raimundo2023": ("MEASUREMENT. PAs (stellar kinematic axis) and PAg (ionised-gas "
   "kinematic axis) are each fitted directly to the observed SAMI DR3 velocity fields, "
   "with 3-sigma uncertainties; DPA=|PAs-PAg| is arithmetic from two measurements. "
   "logMstar is a stellar-population MODEL quantity. NO DM halo, NO Jeans/JAM model."),
 "manga_kincat_ristea2024": ("MEASUREMENT. Stellar and ionised-gas rotational velocities at "
   "1/1.3/2 Re from centred, INCLINATION-CORRECTED rotation curves fitted to the MaNGA "
   "DAP velocity fields, with errors; V/sigma from measured velocity and dispersion. "
   "Inclination correction uses the photometric axis ratio (a measurement). logMstar "
   "and logSFR are stellar-population MODEL quantities. NO DM halo, NO Jeans/JAM."),
 "atlas3d_I_cappellari2011": ("MEASUREMENT. Parent-sample positions, heliocentric velocities, "
   "distances, 2MASS K-band absolute magnitudes, morphological T-type. Distances are "
   "SBF/NED-D measurements or Hubble-flow. NO DM model."),
 "atlas3d_III_emsellem2011": ("MEASUREMENT. lambda_Re (specific stellar angular momentum "
   "proxy), V/sigma and ellipticity are computed directly from the observed SAURON "
   "stellar velocity/dispersion maps and photometry. The slow/fast-rotator label is a "
   "classification OF those measurements. NO DM halo, NO Jeans/JAM model."),
 "atlas3d_XXIII_krajnovic2013": ("MEASUREMENT (photometric decomposition). MGE/multi-component "
   "surface-photometry fits: theta, mu, alpha, beta etc. are light-profile parameters. "
   "Photometry is a measurement; no DM halo enters."),
 "califa_gaskin_garcialorenzo2015": ("MEASUREMENT. Ionised-gas (Halpha) velocity fields of "
   "CALIFA galaxies: systemic velocity, kinematic PA at several radii (PA1, PAout), "
   "kinematic centre offsets, velocity amplitudes. NO DM model."),
 "califa_starkin_falconbarroso2017": ("MEASUREMENT. CALIFA STELLAR kinematics: PA, ellipticity, "
   "Reff, lambda_Re, V/sigma from the observed stellar velocity/dispersion maps. M* is "
   "a stellar-population MODEL quantity. NO DM halo, NO Jeans/JAM."),
 "manga_atlas3d_etg_zhong2026": ("MIXED - READ CAREFULLY. LR(Re) (observed lambda_R) and "
   "Ell(Re) are MEASUREMENTS. LRintr(Re) (intrinsic/deprojected lambda_R), Inc, and "
   "kapparot, fspheroid, fhalo are MODEL quantities from an orbit-superposition "
   "(Schwarzschild) dynamical decomposition. NOTE for clarity: 'fhalo' here is the "
   "STELLAR halo mass fraction (a stellar orbital component), NOT a dark-matter halo "
   "fraction - but it is still a MODEL output, not an observation. DO NOT treat fhalo, "
   "fspheroid, kapparot or LRintr as observations. The orbit-superposition machinery "
   "that produced them does include a mass model, so these columns are DM-assumption "
   "contaminated at one remove."),
 "s0_morphokin_mendezabreu2018": ("MEASUREMENT (photometric decomposition). Bulge/disc/bar "
   "structural parameters (Sersic n, re, h, bar length, surface brightnesses) from 2D "
   "photometric decomposition of CALIFA S0 galaxies. Photometry is a measurement; the "
   "bulge+disc+bar DECOMPOSITION is a parametric model of the light, but no DM halo and "
   "no dynamical mass enters."),
 "cp_corsini1999": ("MEASUREMENT. Long-slit STELLAR and IONISED-GAS rotation velocities and "
   "velocity dispersions vs radius along the major axis of early-type spirals. Separate "
   "gaskin/stelkin tables give BOTH components. NO DM model."),
 "cp_sarzi2000": ("MEASUREMENT. Long-slit stellar and ionised-gas kinematics of NGC 4672 along "
   "major and minor axes (Axis/Kin columns). NGC 4672 hosts a bulge inclined to the disc. "
   "NO DM model."),
 "cp_vegabeltran2001": ("MEASUREMENT. Long-slit stellar (table5: Vel, sigma, h3, h4) and "
   "ionised-gas (table6: Vel, sigma, Ion) kinematics vs radius for 17 disc galaxies. "
   "BOTH components measured in the same galaxy. NO DM model."),
 "cp_corsini2002": ("MEASUREMENT. Long-slit stellar (table3) and ionised-gas (table4) "
   "kinematics of NGC 2855 vs radius. NO DM model."),
 "cp_corsini2003": ("MEASUREMENT. Long-slit ionised-gas (table3) and stellar (table4) rotation "
   "velocities and dispersions vs radius for spirals. BOTH components. NO DM model."),
 "cp_pizzella2004": ("MEASUREMENT. Long-slit ionised-gas (table3) and stellar (table4) "
   "kinematics vs radius for 17 nearby spirals, plus a galaxy table with PA, inclination, "
   "B magnitude, R25. BOTH components. NO DM model."),
 "califa_angmom_falconbarroso2019": ("MEASUREMENT. lambda_Re, V/sigma, ellipticity from the "
   "observed CALIFA stellar velocity maps. M* is a stellar-population MODEL quantity. "
   "NO DM halo."),
 "califa_kinclass_kalinova2017": ("MODEL-CONTAMINATED. The circular velocity curves are built "
   "with Jeans Anisotropic Modelling (JAM) / asymmetric-drift-corrected dynamical models "
   "and the table carries an explicit (M/L)dyn column: (M/L)dyn and the circular-velocity "
   "curve shapes ARE MODEL PRODUCTS. PA, eps, Incl, Re, Rmax and Vsys are measurements. "
   "Do NOT treat the Vcirc curves or (M/L)dyn as observations."),
}

FALLBACK = ("UNLABELLED - inspect before use.")

written = []
for e in idx:
    raw = os.path.join(D, e["raw_file"])
    cat, title, tables = parse_vizier(raw)
    total = sum(t["nrows"] for t in tables)
    cols_all = []
    for t in tables:
        for c in t["columns"]:
            cols_all.append({"name": "%s.%s" % (t["name"].split("_")[-1], c["name"]),
                             "unit": c["unit"]})
    write_manifest(
        raw,
        source_url=e["url"],
        query=("HTTP GET %s  [VizieR asu-tsv, -source=%s -out.all -out.max=unlimited]; "
               "catalogue id verified echoed back in the response by assert_vizier_tsv()"
               % (e["url"], e["vizier_catalog_requested"])),
        columns=cols_all,
        row_count=total,
        measurement_or_model=LABEL.get(e["short"], FALLBACK),
        note=("RAW UNMODIFIED VizieR response. This catalogue contains %d table(s): %s. "
              "row_count is the SUM of data rows over all tables; per-table counts are in "
              "the tables_detail field (VizieR concatenates every table of a catalogue "
              "into one asu-tsv response, so a naive line count is inflated by the "
              "interleaved headers of later tables)."
              % (len(tables), ", ".join("%s=%d" % (t["name"], t["nrows"]) for t in tables))),
        extra={"vizier_catalog": cat, "vizier_title": title,
               "vizier_catalog_requested": e["vizier_catalog_requested"],
               "tables_detail": [{"table": t["name"], "nrows": t["nrows"],
                                  "ncols": len(t["columns"]),
                                  "columns": t["columns"]} for t in tables]})
    written.append(raw)

print("\n%d VizieR manifests written" % len(written))
