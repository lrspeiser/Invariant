import viz_get as V, json
r=V.pull("J/ApJS/211/21/zspec","MACS0717_MACS1149_MACS0416_Ebeling2014_ApJS211_21_zspec",
 "Ebeling, Ma & Barrett 2014 ApJS 211, 21, combined 'zspec' table: SPECTROSCOPIC redshift catalogue for the fields of three "
 "MACS clusters, obtained with Keck/DEIMOS and other instruments. Row counts per cluster (column 'MACS'): "
 "J0717.5+3745 = 1266, J1149.5+2223 = 590, J0416.1-2403 = 65 (total 1921). "
 "MEASURED: RA/Dec (J2000), spectroscopic redshift z with uncertainty e_z, emission-line spectral type EmLT, instrument Inst, "
 "and an X-ray point-source counterpart flag CXO. "
 "This is by far the largest spectroscopic sample for MACS J0717.5+3745 and is the primary membership source for that cluster, "
 "which is NOT covered by the Granata+2026 structural catalogue and has only 247 GLASS grism redshifts in Treu+2015. "
 "NO membership column: apply a redshift cut about the cluster redshift (z_cl = 0.545 for M0717, 0.543 for M1149, 0.396 for M0416); "
 "note that M0717 is a well-known multi-component merger with a large line-of-sight velocity spread, so a simple cut is crude. "
 "No magnitudes, no colours, no stellar mass, no structural parameters in this table.")
print(json.dumps({k:v for k,v in r.items() if k!='file'},indent=1))
