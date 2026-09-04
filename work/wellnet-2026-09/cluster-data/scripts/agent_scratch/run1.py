import viz_get as V
res={}
res['sohn']=V.pull("J/ApJS/229/20/table2","A2029_Sohn2017_ApJS229_20_table2_specmembers",
  "Sohn et al. 2017 ApJS 229, 20 Table 2: 982 SPECTROSCOPIC MEMBERS of Abell 2029 (MMT/Hectospec + SDSS). MEASURED: RA, Dec, cz (redshift x c), cz error, central stellar velocity dispersion sigma within rest-frame 3 kpc aperture, Dn4000 index. Membership: inclusion in this table IS the membership determination (caustic technique, Sohn+2017). No stellar mass, no Sersic/R_e/q/PA in this table.")
res['viz_bad']=V.pull("J/A+A/590/A30/xxxx","TRAPTEST_nonexistent","trap test")
import json; print(json.dumps(res,indent=1)[:1500])
