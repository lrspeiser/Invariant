"""Coverage-qualified gas ablation after frozen cube nuisance fits.

The pilot-001 gas diagnostic used blanked zeros and is not admissible evidence
for a density/void mechanism. This replacement preserves those original scores.
"""
import json,shutil,warnings
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter,map_coordinates
from astropy.io import fits
from astropy.wcs import WCS
import torch
from gravity_cube_model import CubeModel,tensor
from run_gravity_cube_pilot import block
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False))
def sample(asset,ra,dec):
 with fits.open(ROOT/asset['file']) as hd:
  a=np.squeeze(hd[0].data).astype(float);w=WCS(hd[0].header).celestial
 x,y=w.all_world2pix(ra,dec,0)
 return map_coordinates(a,[y,x],order=1,mode='constant',cval=np.nan)

def main():
 dest=ROOT/'work/gravity-first-principles/cube-gas-coverage-001';dest.mkdir(exist_ok=False)
 shutil.copy2(__file__,dest/'runner.py')
 reg={'predecessor':'conditional-cube-pilot-001','correction':'Pilot gas term treated blanked MOM0 as intensity zero. Those gas scores are inadmissible; retain for audit.',
  'source_mask':'MOM0 finite and positive; block-average intensity divided by covered fraction; Gaussian support >=.98 at 48 and 96 arcsec FWHM. Detected-emission domain only, not empty-space inference.',
  'context':'Bounded log(broad/local) on covered points; subtract mean in fixed 50 arcsec deprojected annuli, clip [-1,1]. Zero candidate correction outside coverage, not zero physical mass.',
  'spatial_score':'Original geometric train/test masks intersect context coverage, minimum 30 pixels each. Same full-cube baseline parameters remain frozen.',
  'transfer':'13 beta values [-.3,.3]; choose from OTHER galaxies training loss, normalized per-galaxy by beta=0; score target test spectra. No target test tuning.',
  'eligibility':'All five original cube fits converged, covariance gate passed, and sufficient covered train/test pixels.',
  'matter_audit':'Stellar light/mass-tracer and CO intensity/error map footprints at the same sky positions. Finite coverage is not a detection or calibrated total density. CO nondetection is not zero H2.',
  'independent_confirmation':False}
 save(dest/'registration.json',reg)
 result=read('work/gravity-first-principles/conditional-cube-pilot-001/result.json')
 audit=read('work/gravity-first-principles/conditional-cube-pilot-001/data-audit.json')
 geom={a['name']:a['geometry'] for a in audit}
 moments={a['name']:a for a in read('work/gravity-first-principles/things-observable-acquisition-003/receipt.json')['files'] if a['resolution']=='NA' and a['moment']==0}
 assets=read('work/gravity-first-principles/stellar-co-acquisition-001/receipt.json')['files']
 rows=[];skipped=[];matter=[];betas=np.linspace(-.3,.3,13)
 for original in result['objects']:
  name=original['name'];p=dict(np.load(ROOT/'work/private/conditional-cube-pilot-001'/(name+'.npz')))
  g=geom[name];ra=p['east']/np.cos(np.deg2rad(g['dec']))/3600+g['ra'];dec=p['north']/3600+g['dec']
  selected=p['train_mask']|p['test_mask'];specific={a['role']:a for a in assets if a['name']==name}
  maps={role:sample(a,ra,dec) for role,a in specific.items() if role in ('STELLAR_MASS_MAP','STELLAR_IRAC1_FLUX','STELLAR_IRAC1_WEIGHT','STELLAR_ICA_MASK','CO21_MOM0','CO21_EMOM0')}
  star=maps.get('STELLAR_MASS_MAP',maps.get('STELLAR_IRAC1_FLUX'))
  starcover=np.isfinite(star)
  if 'STELLAR_IRAC1_WEIGHT' in maps:starcover &= np.isfinite(maps['STELLAR_IRAC1_WEIGHT'])&(maps['STELLAR_IRAC1_WEIGHT']>0)
  co=maps['CO21_MOM0'];err=maps['CO21_EMOM0'];cocover=np.isfinite(co)&np.isfinite(err)&(err>0)
  matter.append(dict(name=name,selected_pixels=int(selected.sum()),
   stellar_product='ICA stellar tracer' if 'STELLAR_MASS_MAP' in maps else 'SINGS 3.6um light',
   stellar_covered_fraction=float(np.mean(starcover[selected])),
   co_with_positive_error_fraction=float(np.mean(cocover[selected])),
   jointly_covered_fraction=float(np.mean((starcover&cocover)[selected])),
   co_gt_3sigma_pixels=int(np.sum(selected&cocover&(co>3*err))),
   co_covered_nondetection_pixels=int(np.sum(selected&cocover&(co<=3*err))),
   foreground_mask_calibration_complete=False,total_density_available=False))
  mom=np.squeeze(fits.getdata(ROOT/moments[name]['file'])).astype(float)
  mask=np.isfinite(mom)&(mom>0);coverage=block(mask.astype(float))
  total=block(np.where(mask,mom*.001,0))
  localweight=gaussian_filter(coverage,48/2.35482/12)
  broadweight=gaussian_filter(coverage,96/2.35482/12)
  local=gaussian_filter(total,48/2.35482/12)/np.maximum(localweight,1e-10)
  broad=gaussian_filter(total,96/2.35482/12)/np.maximum(broadweight,1e-10)
  valid=(localweight>=.98)&(broadweight>=.98)&(local>0)&(broad>0)
  context=np.zeros_like(local);context[valid]=np.clip(np.log(broad[valid]/local[valid]),-1,1)
  for lo in range(0,601,50):
   region=valid&(p['radius']>=lo)&(p['radius']<lo+50)
   if np.any(region):context[region]-=np.mean(context[region])
  context=np.clip(context,-1,1);p['gas_context']=context
  p['train_mask'] &= valid;p['test_mask'] &= valid
  nt=int(p['train_mask'].sum());nv=int(p['test_mask'].sum())
  matter[-1].update(hi_context_train_pixels=nt,hi_context_test_pixels=nv)
  if min(nt,nv)<30 or not all(f['optimizer_success'] for f in original['fits']):
   skipped.append(dict(name=name,train_pixels=nt,test_pixels=nv,reason='Coverage or original convergence gate'));continue
  model=CubeModel(p);full=original['fits'][-1];train=[];test=[]
  with torch.no_grad():
   for beta in betas:
    train.append(float(model.loss(tensor(full['params']),'full',gas_beta=beta,penalize=False)))
    test.append(float(model.loss(tensor(full['params']),'full','test',gas_beta=beta,penalize=False)))
  rows.append(dict(name=name,train_pixels=nt,test_pixels=nv,train_loss=train,test_loss=test))
  del model;torch.cuda.empty_cache()
 predictions=[]
 if len(rows)>=3:
  for row in rows:
   others=[r for r in rows if r['name']!=row['name']]
   score=np.mean([np.array(r['train_loss'])/r['train_loss'][6] for r in others],axis=0)
   choice=int(np.argmin(score));base=row['test_loss'][6]
   predictions.append(dict(name=row['name'],beta=float(betas[choice]),baseline_test_loss=base,
    gas_test_loss=row['test_loss'][choice],fractional_improvement=1-row['test_loss'][choice]/base))
 output=dict(status='COMPLETE_DEVELOPMENT_COVERAGE_ABLATION',rows=rows,skipped=skipped,matter_coverage=matter,
  beta_grid=betas.tolist(),predictions=predictions,registration=reg,
  mean_fractional_improvement=float(np.mean([r['fractional_improvement'] for r in predictions])) if predictions else None)
 save(dest/'result.json',output)
 print(json.dumps({'eligible':len(rows),'predictions':predictions,'skipped':skipped},indent=2))

if __name__=='__main__':
 warnings.filterwarnings('ignore',category=Warning,module='astropy')
 main()
