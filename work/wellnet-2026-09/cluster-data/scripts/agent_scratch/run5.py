import viz_get as V, json
MOL=("Molino et al. 2017 MNRAS 470, 95, table '%s': the CLASH 24-band BPZ photometric-redshift catalogue for %s. "
 "MEASURED: RA/Dec (J2000), image x/y, SExtractor shape parameters -- theta (position angle, CCW from x), a and b "
 "(profile RMS along the major/minor axes, pix -> axis ratio q=b/a), rf (fraction-of-light / half-light radius, pix), "
 "rk (Kron radius), rp (Petrosian radius), isophotal area, FWHM, S/N, point-source flag PS; AB magnitudes with errors in "
 "up to 24 HST/WFC3+ACS bands F225W..F160W in two flavours (…PZ 'restricted', optimised for photo-z; …M 'moderated', "
 "optimised for stellar mass) -> full colour information; zsp = SPECTROSCOPIC redshift where known (-99 if unknown) with "
 "reference r_zsp and quality q_zsp (0=secure). MODEL-DERIVED: zb1 (BPZ photometric redshift, first peak) with 95 per cent "
 "limits, spectral type Tb1, Odds1, Chi2, BMAG (absolute B magnitude) and logM* (log10 stellar mass, Msun). "
 "LENS-MODEL DERIVED -- DO NOT treat as observations: F814W (lensing-corrected magnitude), BMAGLC and logM*LC "
 "(lensing-corrected absolute magnitude and stellar mass). Also carries zCluster and PDistBCG (projected physical distance "
 "to the BCG, Mpc -- depends on an assumed cosmology). NO membership flag: apply a cut on zsp/zb1 about zCluster. "
 "NO Sersic index. Footprint is the CLASH/HST field, which differs from the HFF footprint.")
out={}
jobs=[
 ("J/MNRAS/470/95/macs0717","MACS0717_Molino2017_CLASH_MNRAS470_95_photoz_mass", MOL%("macs0717","MACS J0717.5+3745 (z_cl=0.548)")),
 ("J/MNRAS/470/95/macs0416","MACS0416_Molino2017_CLASH_MNRAS470_95_photoz_mass", MOL%("macs0416","MACS J0416.1-2403 (z_cl=0.396)")),
 ("J/MNRAS/470/95/macs1149","MACS1149_Molino2017_CLASH_MNRAS470_95_photoz_mass", MOL%("macs1149","MACS J1149.5+2223 (z_cl=0.544)")),
 ("J/MNRAS/470/95/rxj2248","AS1063_Molino2017_CLASH_MNRAS470_95_photoz_mass", MOL%("rxj2248","RX J2248-4431 = Abell S1063 (z_cl=0.348)")),
]
for src,name,note in jobs: out[src]=V.pull(src,name,note)
print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='file'} for k,v in out.items()},indent=1))
