import viz_get as V, json
out={}
jobs = [
 ("J/ApJS/235/14/clugal","HFF6_Shipley2018_ApJS235_14_clugal_photometry",
  "Shipley et al. 2018 ApJS 235, 14 (HFF-DeepSpace) 'clugal' = combined CLUSTER-field photometric catalogue for ALL SIX Hubble Frontier Fields clusters (Abell 2744, MACS J0416.1-2403, MACS J0717.5+3745, MACS J1149.5+2223, Abell S1063/RXC J2248.7-4431, Abell 370). Column 'Cl' selects the cluster. MEASURED: RAJ2000, DEJ2000, image x/y, total fluxes in up to 22 bands (HST UVIS/ACS/WFC3 F225W..F160W, Keck/MOSFIRE Ks, Spitzer/IRAC ch1-ch4) with errors -> magnitudes and colours; SExtractor morphology: theta (position angle of major axis, deg), Aimg/Bimg (semi-major/semi-minor axes in pix -> axis ratio q=Bimg/Aimg), FRad (circular aperture radius enclosing half the total flux, i.e. half-light radius, pix), Rad (Kron radius, pix), CLASS_STAR, star/galaxy flag. zspec = spectroscopic redshift where available (compiled from literature, column r_zspec gives the reference). NOT in the VizieR version: photometric redshift (EAZY) and stellar mass (FAST) -- those live in the .zout/.fout files of the MAST HLSP release. NO Sersic index n and NO Sersic effective radius (SExtractor FLUX_RADIUS is a non-parametric half-light radius, not a Sersic R_e). NO membership flag: membership must be derived by the user from zspec/photo-z."),
 ("J/A+A/590/A31/a2744cl","A2744_ASTRODEEP_Merlin2016_AA590_A31_a2744cl_photometry",
  "Merlin et al. 2016 A&A 590, A31 (ASTRODEEP) table a2744cl: multiwavelength photometric catalogue of the Abell 2744 CLUSTER field, HFF. MEASURED: RAJ2000/DEJ2000, image x/y, AB magnitudes + errors and fluxes + errors in B435 V606 I814 Y105 J125 JH140 H160 (HST) plus Ks (HAWK-I) and IRAC CH1/CH2, SExtractor segmentation params. Photo-z / stellar mass are NOT in this table (they are in Castellano+2016 A&A 590 A30, which is not in VizieR). No membership flag, no Sersic parameters."),
 ("J/A+A/590/A31/m0416cl","MACS0416_ASTRODEEP_Merlin2016_AA590_A31_m0416cl_photometry",
  "Merlin et al. 2016 A&A 590, A31 (ASTRODEEP) table m0416cl: multiwavelength photometric catalogue of the MACS J0416.1-2403 CLUSTER field, HFF. Same content as a2744cl. Photo-z/Mstar not in this table; no membership flag; no Sersic parameters."),
 ("J/A+A/607/A30/m0717clz","MACS0717_ASTRODEEP_DiCriscienzo2017_AA607_A30_m0717clz_zphys",
  "Di Criscienzo et al. 2017 A&A 607, A30 (ASTRODEEP) table m0717clz: photometric-redshift and physical-parameter catalogue for the MACS J0717.5+3745 CLUSTER field, HFF. MODEL-DERIVED: zbest (median photo-z from 6 codes, or spec-z where f_zbest=1), Mstar and SFR from SED fitting (Salpeter IMF), with and without nebular emission; magnif = lens-model magnification (LENS-MODEL DERIVED, not an observation). MEASURED only indirectly: f_zbest=1 flags a spectroscopic redshift. Cross-match on ID to table m0717cla (magnitudes) for RA/Dec and photometry. No membership flag, no Sersic parameters."),
 ("J/A+A/607/A30/m1149clz","MACS1149_ASTRODEEP_DiCriscienzo2017_AA607_A30_m1149clz_zphys",
  "Di Criscienzo et al. 2017 A&A 607, A30 (ASTRODEEP) table m1149clz: photo-z and physical parameters for the MACS J1149.5+2223 CLUSTER field, HFF. Same content as m0717clz. Mstar/SFR are SED-fit MODEL-DERIVED; magnif is LENS-MODEL DERIVED."),
 ("J/A+A/607/A30/m0717cla","MACS0717_ASTRODEEP_DiCriscienzo2017_AA607_A30_m0717cla_mags",
  "Di Criscienzo et al. 2017 A&A 607, A30 (ASTRODEEP) table m0717cla: MEASURED AB magnitudes + errors and sky/image positions for the MACS J0717.5+3745 CLUSTER field. Join to m0717clz on ID for photo-z and stellar mass."),
 ("J/A+A/607/A30/m1149cla","MACS1149_ASTRODEEP_DiCriscienzo2017_AA607_A30_m1149cla_mags",
  "Di Criscienzo et al. 2017 A&A 607, A30 (ASTRODEEP) table m1149cla: MEASURED AB magnitudes + errors and sky/image positions for the MACS J1149.5+2223 CLUSTER field. Join to m1149clz on ID."),
]
for src,name,note in jobs:
    out[src]=V.pull(src,name,note)
print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='file'} for k,v in out.items()},indent=1))
