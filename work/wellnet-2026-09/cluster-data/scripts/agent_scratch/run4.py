import viz_get as V, json
MUSE=("Richard et al. 2021 A&A 646, A83, table '%s': MUSE integral-field SPECTROSCOPIC redshift catalogue for %s (HFF-DeepSpace/MUSE GTO). "
 "MEASURED: RA/Dec (J2000), spectroscopic redshift z with confidence class zconf (higher = more secure), the individual line redshifts "
 "(zforbid/zbalmer/zabs/zlya) with errors, SExtractor shape quantities A_WORLD/B_WORLD (major/minor axis -> axis ratio q=B/A), THETA_J2000 "
 "(position angle), FLUX_RADIUS (half-light radius, arcsec), KRON_RADIUS, CLASS_STAR, FWHM, and HST ISO + AUTO AB magnitudes in "
 "F435W F606W F814W F105W F125W F140W F160W with errors (-> colours). LENS-MODEL DERIVED: mu (magnification) -- NOT an observation. "
 "'Mul' flags multiply-imaged LENSED BACKGROUND sources (not members). NO membership column: apply a redshift cut around the cluster "
 "redshift. No Sersic index, no stellar mass.")
out={}
jobs=[
 ("J/A+A/646/A83/a2744","A2744_Richard2021_MUSE_AA646_A83_a2744_speczcat", MUSE%("a2744","ABELL 2744 (z_cl=0.308)")),
 ("J/A+A/646/A83/mc416ne","MACS0416_Richard2021_MUSE_AA646_A83_mc416ne_speczcat", MUSE%("mc416ne","MACS J0416.1-2403, North-East pointing (z_cl=0.396)")),
 ("J/A+A/646/A83/mc416s","MACS0416_Richard2021_MUSE_AA646_A83_mc416s_speczcat", MUSE%("mc416s","MACS J0416.1-2403, South pointing (z_cl=0.396)")),
 ("J/A+A/646/A83/a370","A370_Richard2021_MUSE_AA646_A83_a370_speczcat", MUSE%("a370","ABELL 370 (z_cl=0.375)")),
 ("J/ApJS/224/33/table2","MACS0416_Balestra2016_CLASHVLT_ApJS224_33_table2_specz",
  "Balestra et al. 2016 ApJS 224, 33, Table 2: the CLASH-VLT VIMOS SPECTROSCOPIC redshift catalogue of MACS J0416.1-2403 (4386 objects over ~25x25 arcmin, far wider than the HST/HFF footprint). MEASURED: RA/Dec (J2000), spectroscopic redshift z, quality flag q_z (2,3,4,9 = insecure..secure), reference code r_z, Subaru/Suprime-Cam Kron R-band AB magnitude. NO membership column: Balestra+2016 define members by |z - z_cl|; apply a redshift cut around z_cl = 0.396. Single band only, so no colour. No stellar mass, no Sersic parameters."),
 ("J/ApJ/812/114/table3","MACS0717_Treu2015_GLASS_ApJ812_114_table3_specz",
  "Treu et al. 2015 ApJ 812, 114, Table 3: the GLASS (HST/WFC3 Grism Lens-Amplified Survey from Space) redshift catalogue for MACS J0717.5+3745 (247 entries). MEASURED: RA/Dec (J2000), GLASS grism/spectroscopic redshift z, redshift quality q_z (0-4), multiple-solution flag f_z. This is a SMALL catalogue and is the weakest link of the seven clusters. NO membership column, no magnitudes in the VizieR table, no stellar mass, no Sersic parameters."),
]
for src,name,note in jobs: out[src]=V.pull(src,name,note)
print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='file'} for k,v in out.items()},indent=1))
