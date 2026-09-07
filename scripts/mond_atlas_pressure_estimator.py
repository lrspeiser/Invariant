import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,csv,gzip
import numpy as np
from scipy.ndimage import map_coordinates,gaussian_filter1d
from mond_atlas_hi_column import smooth
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-pressure-estimator-001';S=R/'work/gravity-first-principles/mond-atlas-source-sensitivity-001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 out=P/'run001';out.mkdir(exist_ok=False);assets=json.loads((S/'run001/summary.json').read_text())['assets'];assets=[a for a in assets if a['component']=='atomic_helium' and a.get('quadrature_factor',1)==1]
 with gzip.open(S/'hi001/pressure-profiles.csv.gz','rt') as f:old=list(csv.DictReader(f))
 paths=[Path(__file__),P/'PREFLIGHT.md',S/'run001/summary.json',S/'hi001/pressure-profiles.csv.gz',R/'scripts/mond_atlas_hi_column.py']+[R/a['path'] for a in assets]
 bindings={p.relative_to(R).as_posix():sha(p) for p in paths};(out/'bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')
 rows=[];stats=[]
 for a in assets:
  assert sha(R/a['path'])==a['sha256']
  with np.load(R/a['path']) as z:axis=z['latent_axis'];surface=z['intrinsic_effective_surface']/1.36
  dr=.00625;r=np.arange(1921)*dr;theta=np.arange(2048)*2*np.pi/2048;raw=np.zeros(len(r))
  for start in range(0,len(r),128):
   rr=r[start:start+128,None];x=rr*np.cos(theta);y=rr*np.sin(theta);raw[start:start+len(rr)]=map_coordinates(surface,[(x-axis[0])/(axis[1]-axis[0]),(y-axis[0])/(axis[1]-axis[0])],order=1,mode='constant',cval=0,prefilter=False).mean(axis=1)
  sm=gaussian_filter1d(np.r_[raw[:0:-1],raw],.25/dr,mode='constant',cval=0,truncate=8)[len(raw)-1:];finite=np.gradient(sm,dr,edge_order=2)
  selected=sorted([x for x in old if x['branch']==a['branch'] and float(x['radial_spacing_kpc'])==dr],key=lambda x:float(x['r_kpc']));targets=np.array([float(x['r_kpc']) for x in selected]);indices=np.rint(targets/dr).astype(int)
  replay=float(np.max(abs(finite[indices]-[float(x['smoothed_gradient']) for x in selected])));assert replay<1e-10
  analytic=np.empty(len(targets))
  for start in range(0,len(targets),128):_,analytic[start:start+128]=smooth(r,raw,targets[start:start+128])
  positive=raw[indices]>0;fd=100*finite[indices][positive]/raw[indices][positive];ad=100*analytic[positive]/raw[indices][positive];rad=targets[positive]
  for scope,mask in [('full',np.ones(len(rad),bool)),('aperture',(rad>=.75)&(rad<=2.5))]:
   delta=fd[mask]-ad[mask];stats.append(dict(branch=a['branch'],scope=scope,points=int(mask.sum()),exported_derivative_replay_error=replay,pressure_acceleration_max_abs_difference=float(np.max(abs(delta))),pressure_acceleration_rms_difference=float(np.sqrt(np.mean(delta**2))),relative_L2_difference=float(np.linalg.norm(delta)/np.linalg.norm(ad[mask])),units='(km/s)^2/kpc'))
  rows.extend(dict(branch=a['branch'],r_kpc=float(rr),sampled_pressure_acceleration=float(f),integrated_pressure_acceleration=float(d)) for rr,f,d in zip(rad,fd,ad))
 with (out/'comparison.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 result=dict(status='MATCHED_RAW_PROFILE_PRESSURE_ESTIMATOR_COMPARISON',rows=stats,source_region_reads=0,observed_scores=0,estimator_replaced=False)
 (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 assert all(sha(R/k)==h for k,h in bindings.items())
if __name__=='__main__':run()
