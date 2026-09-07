import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,csv,hashlib
import numpy as np
from scipy.special import erf
from mond_atlas_actual_spectra import ROOT,PACKAGE,load_instrument

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 out=PACKAGE/'independent-review';out.mkdir(exist_ok=False)
 r=PACKAGE/'run002';s=json.loads((r/'summary.json').read_text());packet=ROOT/s['private_packet'];assert sha(packet)==s['private_sha256']
 z=np.load(packet);inst=load_instrument();v=inst['velocity_km_s'];pairs={'planar':('p0p0625_z24','p0p03125_z24'),'vertical':('p0p0625_z24','p0p0625_z48'),'joint':('p0p0625_z24','p0p03125_z48'),'fine_planar_vertical':('p0p03125_z24','p0p03125_z48'),'fine_vertical_planar':('p0p0625_z48','p0p03125_z48')}
 rows=list(csv.DictReader((r/'refinement.csv').open()));error=0.;primary=[]
 for row in rows:
  lo,hi=pairs[row['comparison']];i=int(row['aperture']);a=z[lo+'__'+row['case']][i];b=z[hi+'__'+row['case']][i]
  l1=abs(a-b).sum()/abs(b).sum();dc=abs(a@v/a.sum()-b@v/b.sum());error=max(error,abs(l1-float(row['profile_L1'])),abs(dc-float(row['centroid_km_s'])));assert (l1<.01 and dc<.5)==(row['passed']=='True')
  if row['case'].startswith('h0p1_newton_p10_') and '_s1_' in row['case']:primary.append(row)
 cache=ROOT/'work/private/mond-atlas-actual-spectra-001/cache001/p0p03125_z48.npz';c=np.load(cache)
 hist=json.loads((ROOT/'work/gravity-first-principles/mond-atlas-native-spectral-001/NGC2976.json').read_text())['provenance'];cal=np.array(hist['continuum_fit_parent_indices_zero_based']);oi=np.arange(11,53);X=np.vander(np.arange(63,dtype=float),N=hist['polynomial_order']+1,increasing=True);A=np.eye(63)[oi];A[:,cal]-=X[oi]@np.linalg.pinv(X[cal]);assert np.max(abs(A-inst['continuum']))<1e-12
 checks=[]
 for height,gravity,pressure,spin in [('h0p1','newton',10,1),('h0p4','newton_plus_log',15,-1)]:
  rads=c['radius_kpc'];v2=rads*(c[f'force_{height}_{gravity}'].sum(axis=0)+pressure**2*c['pressure_gradient']/c[f'den_{height}']);ok=c['force_supported']&(v2>=0)&np.isfinite(v2);vel=spin*float(c['sin_inclination'])*np.cos(c['phi_rad'][ok])*np.sqrt(v2[ok]);W=c['grouped_flux'][:,ok]
  for branch in ['boxcar_independent','boxcar_hanning_full','boxcar_hanning_decimated']:
   if branch=='boxcar_independent':H=np.eye(63);grid=np.arange(63);width=1
   else:
    step=2 if branch.endswith('decimated') else 1;width=1/step;H=np.zeros((63,step*62+3));grid=(np.arange(H.shape[1])-1)/step
    for i in range(63):H[i,step*i:step*i+3]=[.25,.5,.25]
   dv=abs(inst['velocity_increment_km_s'])*width;cent=inst['first_parent_velocity_km_s']+grid*inst['velocity_increment_km_s'];bins=.5*(erf((cent[:,None]+dv/2-vel)/(np.sqrt(2)*10))-erf((cent[:,None]-dv/2-vel)/(np.sqrt(2)*10)))/dv
   pred=1000*(W@bins.T)@H.T@A.T;tag=f'{height}_{gravity}_p{pressure}_{branch}_s{spin}_sys0_sig10';err=float(abs(pred-z['p0p03125_z48__'+tag]).max());assert err<1e-10;checks.append(dict(case=tag,max_absolute_mjy_beam=err))
 receipt=dict(status='INDEPENDENT_ARITHMETIC_REPLAY_PASS',refinement_rows=len(rows),max_metric_error=error,reference_cases=checks,primary_rows=len(primary),primary_failed=sum(x['passed']!='True' for x in primary),primary_max_L1=max(float(x['profile_L1']) for x in primary),primary_max_centroid_km_s=max(float(x['centroid_km_s']) for x in primary),source_observed_spectra_read=0,scope='Independent erf integration, independently constructed H and OLS continuum, explicit source v2 law; cached sampled beam/source quadratures inherited, not independently regenerated.',bindings={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),packet,cache,r/'summary.json',r/'refinement.csv']})
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':run()
