# -*- coding: utf-8 -*-
import json, os

MEM = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\cluster-data\members"

P = {
 "A2029_Sohn2019_ApJ872_192_table1_member_stellarmass.raw.tsv.manifest.json":
 ("Sohn et al. 2019 ApJ 872, 192, Table 1: physical properties of the 1215 SPECTROSCOPIC MEMBERS of ABELL 2029 (z_cl = 0.0784). "
  "MEMBERSHIP: inclusion in this table IS the membership statement -- these are the caustic-technique members of Sohn et al. 2019 "
  "(see also the Y/N 'Mm' flag in the companion file A2029_Sohn2019_ApJ871_129_table1_redshift_catalog.raw.tsv). "
  "MEASURED: SDSS designation and objID, RAJ2000/DEJ2000, spectroscopic redshift z with error; "
  "sigma = central stellar VELOCITY DISPERSION (km/s) with error, from SDSS/Hectospec spectra -- a genuine kinematic observable; "
  "Dn4000 = the 4000 Angstrom break index with error, a directly measured spectral index; "
  "Class = spectral classification, 'E+A?' = post-starburst flag; T13/S17/PapI cross-identification flags to Tyler+2013, "
  "Sohn+2017 and Paper I. "
  "MODEL-DERIVED: logM* (log10 stellar mass, Msun) with error, from SED/spectral fitting -- label as model-derived. "
  "NOT PRESENT: multi-band magnitudes and colours (use the r magnitude in the Sohn+2019 ApJ 871, 129 redshift catalogue, or SDSS "
  "directly), effective radius Re, Sersic index n, axis ratio q, position angle theta. No structural parameters of any kind."),
 "A2029_Tyler2013_ApJ773_86_table1_memberlist.raw.tsv.manifest.json":
 ("Tyler, Rieke & Bai 2013 ApJ 773, 86, Table 1: ABELL 2029 member list, 585 galaxies. An earlier, independent member compilation "
  "than Sohn et al. 2017/2019, focused on star-formation diagnostics. "
  "MEMBERSHIP: inclusion in this table IS the membership statement. "
  "MEASURED: RAJ2000/DEJ2000, spectroscopic redshift z. "
  "MODEL-DERIVED / luminosity-derived: logL24 (24 um), logLHa (H-alpha) and logLFUV (far-UV) luminosities, SFRmin/SFRmax "
  "(star-formation-rate bounds) and logM* (log10 stellar mass). These depend on an assumed cosmology and on SFR/mass calibrations. "
  "NOT PRESENT: apparent magnitudes and colours, effective radius Re, Sersic index n, axis ratio q, position angle theta, "
  "velocity dispersion."),
 "A2029_Sohn2017_ApJS229_20_table2_specmembers.raw.tsv.manifest.json":
 ("Sohn et al. 2017 ApJS 229, 20, Table 2: 982 SPECTROSCOPIC MEMBERS of ABELL 2029 (MMT/Hectospec plus SDSS). "
  "MEMBERSHIP: inclusion in this table IS the membership determination (caustic technique). "
  "MEASURED: RAJ2000/DEJ2000; cz and its error in km/s -- note the redshift is given as cz, so z = cz/299792.458; "
  "sigma = central stellar VELOCITY DISPERSION within a rest-frame 3 kpc aperture, with error (f_sigma flags rows where the "
  "uncertainty is unavailable) -- a genuine kinematic observable; Dn4000 = 4000 Angstrom break index; r_cz = redshift provenance; "
  "T13 = cross-match flag to Tyler+2013 within 2 arcsec. "
  "NOT PRESENT: stellar mass, magnitudes/colours, effective radius Re, Sersic index n, axis ratio q, position angle theta."),
}

for fn, note in P.items():
    p = os.path.join(MEM, fn)
    m = json.load(open(p, encoding="utf-8"))
    m["note"] = note
    json.dump(m, open(p, "w", encoding="utf-8"), indent=2)
    print("patched", fn)
