import viz_get as V, json
GN = ("Granata et al. 2026 A&A 709, A254, table '%s': STRUCTURAL PARAMETERS of the %s CLUSTER MEMBERS. "
 "Membership: inclusion in this table IS the membership statement -- these are the spectroscopically confirmed / "
 "colour-selected cluster members used to build the paper's samples; no separate probability column. "
 "MEASURED (2D single-Sersic surface-brightness fits to HST images, per band F435W F606W F814W F105W F125W F140W F160W): "
 "total AB magnitude MAG_*, Sersic index n_*, axis ratio AR_* (=q=b/a), position angle PA_* (deg), "
 "mean surface brightness within Re MU_*, all with 1-sigma errors; RA/Dec (J2000). "
 "SEMI-MODEL-DERIVED: Re_* is the CIRCULARISED effective radius quoted in kpc -- the fitted angular Re is measured, but its "
 "conversion to kpc uses the cluster redshift and an assumed (flat LCDM) cosmology. "
 "NOT in this table: redshift, stellar mass, velocity dispersion. NO table exists here for MACS J0717.5+3745.")
out={}
jobs=[
 ("J/A+A/709/A254/a2744","A2744_Granata2026_AA709_A254_members_structural", GN % ("a2744","Abell 2744")),
 ("J/A+A/709/A254/m0416","MACS0416_Granata2026_AA709_A254_members_structural", GN % ("m0416","MACS J0416.1-2403")),
 ("J/A+A/709/A254/m1149","MACS1149_Granata2026_AA709_A254_members_structural", GN % ("m1149","MACS J1149.6+2223")),
 ("J/A+A/709/A254/as1063","AS1063_Granata2026_AA709_A254_members_structural", GN % ("as1063","Abell S1063 / RXC J2248.7-4431")),
 ("J/MNRAS/489/99/a370ph","A370_Bradac2019_MNRAS489_99_a370ph_photometry",
  "Bradac et al. 2019 MNRAS 489, 99 (ASTRODEEP-style HFF release) table a370ph: MEASURED photometric catalogue of the ABELL 370 cluster field. RA/Dec (J2000), image x/y, SExtractor stellarity S/G, and iso/auto/3-arcsec-aperture fluxes (uJy) and AB magnitudes with errors in HST F435W F606W F814W F105W F125W F140W F160W plus ground-based/IRAC bands. Join to a370pr on ID/row for photo-z and stellar mass. No membership flag, no Sersic parameters."),
 ("J/MNRAS/489/99/a370pr","A370_Bradac2019_MNRAS489_99_a370pr_zphys",
  "Bradac et al. 2019 MNRAS 489, 99 table a370pr: photometric-redshift and physical-parameter catalogue for ABELL 370. MEASURED: RA/Dec; zsp = spectroscopic redshift where available. MODEL-DERIVED: zbest (median photo-z), zphEAZY, zphOAR, and Mstar + SFR from SED fitting with and without nebular emission. LENS-MODEL DERIVED: Magnif (median magnification over public HFF v4 models) -- NOT an observation. No membership flag, no Sersic parameters."),
 ("J/MNRAS/489/99/j2248ph","AS1063_Bradac2019_MNRAS489_99_j2248ph_photometry",
  "Bradac et al. 2019 MNRAS 489, 99 table j2248ph: MEASURED photometric catalogue of the RXC J2248.7-4431 = ABELL S1063 cluster field. Same content as a370ph. Join to j2248pr for photo-z/Mstar."),
 ("J/MNRAS/489/99/j2248pr","AS1063_Bradac2019_MNRAS489_99_j2248pr_zphys",
  "Bradac et al. 2019 MNRAS 489, 99 table j2248pr: photo-z and physical parameters for RXC J2248.7-4431 = ABELL S1063. Same content/caveats as a370pr: zsp measured, zbest/Mstar/SFR SED-model-derived, Magnif lens-model-derived."),
 ("J/MNRAS/514/497/tablea1","A370_Lagattuta2022_PilotWINGS_MNRAS514_497_tablea1_specz",
  "Lagattuta et al. 2022 MNRAS 514, 497, Table A1: the ABELL 370 'Pilot-WINGS' MASTER CATALOGUE. MEASURED: RA/Dec aligned to Gaia DR2, redshift z (mostly MUSE spectroscopic), redshift quality flag q_z (1=low..3=high), and HST AB magnitudes F435W..F160W with errors. IDfrom indicates the detection route; MULID flags multiply-imaged systems (those are LENSED BACKGROUND sources, not members). Cluster membership must be applied by the user as a redshift cut around z_cl=0.375; no membership column. No stellar mass, no Sersic parameters."),
 ("J/ApJ/871/129/table1","A2029_Sohn2019_ApJ871_129_table1_redshift_catalog",
  "Sohn et al. 2019 ApJ 871, 129, Table 1: the redshift galaxy catalogue of ABELL 2029 (4376 galaxies). MEASURED: RA/Dec (J2000), SDSS composite-model r magnitude, redshift z with error and provenance r_z. MEMBERSHIP FLAG: column 'Mm' = Y/N from the CAUSTIC technique -- 1054 members (Y), 3322 non-members (N). This is the best available RA+Dec+z+membership-flag table for A2029. No colour beyond r, no stellar mass, no Sersic parameters in this table."),
 ("J/ApJ/872/192/table1","A2029_Sohn2019_ApJ872_192_table1_member_stellarmass",
  "Sohn et al. 2019 ApJ 872, 192, Table 1: physical properties of the 1215 SPECTROSCOPIC MEMBERS of ABELL 2029. MEASURED: SDSS name/objID, RA/Dec, spectroscopic redshift z + error. MODEL-DERIVED: logM* (log stellar mass) + error, from SED/spectral fitting. Membership: inclusion in this table IS the membership statement (Sohn+2019 caustic members). No Sersic parameters."),
 ("J/ApJ/773/86/table1","A2029_Tyler2013_ApJ773_86_table1_memberlist",
  "Tyler, Rieke & Bai 2013 ApJ 773, 86, Table 1: ABELL 2029 member list (585 galaxies). Independent earlier member compilation; see column list in manifest."),
 ("J/MNRAS/477/648/tablea1","AS1063_Tortorelli2018_MNRAS477_648_tablea1_photometry",
  "Tortorelli et al. 2018 MNRAS 477, 648, Table A1: photometric catalogue for ABELL S1063 (z=0.348) cluster members used for surface-photometry/Fundamental-Plane work. Small (95 rows)."),
 ("J/MNRAS/477/648/tablea2","MACS1149_Tortorelli2018_MNRAS477_648_tablea2_photometry",
  "Tortorelli et al. 2018 MNRAS 477, 648, Table A2: photometric catalogue for MACS J1149.5+2223 (z=0.542) cluster members. Small (68 rows)."),
]
for src,name,note in jobs:
    out[src]=V.pull(src,name,note)
print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='file'} for k,v in out.items()},indent=1))
