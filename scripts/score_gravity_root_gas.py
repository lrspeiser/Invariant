"""Repeat the fixed covered-HI descriptor against the repaired cube baseline."""
import json,shutil,warnings
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from astropy.io import fits
import torch
from gravity_cube_model import tensor
from gravity_cube_root_geometry import RootGeometryCube as ConstrainedCube
from run_gravity_cube_pilot import block
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False))
def main():
    dest=ROOT/'work/gravity-first-principles/root-cube-gas-001';dest.mkdir(exist_ok=False)
    shutil.copy2(__file__,dest/'runner.py')
    reg=dict(baseline='constrained-cube-root-001',descriptor_predecessor='cube-gas-coverage-002',
        descriptor='Same bounded broad/local HI contrast with radial mean removed. Positive detected MOM0 support >=.98 at 48/96 arcsec; no physical void inference.',
        grid=np.linspace(-.3,.3,13).tolist(),gate='All five selected fits converge; at least 30 covered calibration/test pixels each; no flagged projection fold and numerical root checks pass.',
        validation='One beta chosen on other galaxies calibration loss normalized at beta=0, equal galaxy weighting. Full-model parameters frozen; evaluate target test spectra.',
        interpretation='Previously exposed development observations and still-imperfect conditional kinematic baseline. This does not test total-density coherence.')
    save(dest/'registration.json',reg)
    result=read('work/gravity-first-principles/constrained-cube-root-001/result.json')
    assets={a['name']:a for a in read('work/gravity-first-principles/things-observable-acquisition-003/receipt.json')['files'] if a['resolution']=='NA' and a['moment']==0}
    rows=[];skipped=[]
    for obj in result['objects']:
        name=obj['name'];p=dict(np.load(ROOT/'work/private/constrained-cube-balanced-001'/(name+'.npz')))
        mom=np.squeeze(fits.getdata(ROOT/assets[name]['file'])).astype(float);valid=np.isfinite(mom)&(mom>0)
        coverage=block(valid.astype(float));values=block(np.where(valid,mom*.001,0))
        lw=gaussian_filter(coverage,48/2.35482/12);bw=gaussian_filter(coverage,96/2.35482/12)
        local=gaussian_filter(values,48/2.35482/12)/np.maximum(lw,1e-10)
        broad=gaussian_filter(values,96/2.35482/12)/np.maximum(bw,1e-10)
        good=(lw>=.98)&(bw>=.98)&(local>0)&(broad>0)
        context=np.zeros_like(local);context[good]=np.clip(np.log(broad[good]/local[good]),-1,1)
        for lo in range(0,601,50):
            region=good&(p['radius']>=lo)&(p['radius']<lo+50)
            if np.any(region):context[region]-=context[region].mean()
        p['gas_context']=np.clip(context,-1,1);p['train_mask'] &= good;p['test_mask'] &= good
        nt=int(p['train_mask'].sum());nv=int(p['test_mask'].sum())
        numerical=obj['geometry_diagnostic']
        bad_geometry=(numerical['possible_fold_fraction']>0 or numerical['max_root_residual_arcsec']>=.02 or numerical['root_36_vs_52_whitened_rms']>=.005)
        if min(nt,nv)<30 or bad_geometry or not all(f['optimizer_success'] for f in obj['fits']):
            skipped.append(dict(name=name,train_pixels=nt,test_pixels=nv,reason='Coverage, selected-fit convergence or projection geometry gate'));continue
        m=ConstrainedCube(p);full=obj['fits'][-1];tr=[];te=[]
        with torch.no_grad():
            for beta in reg['grid']:
                tr.append(float(m.loss(tensor(full['params']),'full',gas_beta=beta,penalize=False)))
                te.append(float(m.loss(tensor(full['params']),'full','test',gas_beta=beta,penalize=False)))
        rows.append(dict(name=name,train_pixels=nt,test_pixels=nv,train_loss=tr,test_loss=te));del m
    predictions=[]
    if len(rows)>=3:
        for row in rows:
            others=[o for o in rows if o['name']!=row['name']]
            index=int(np.argmin(np.mean([np.array(o['train_loss'])/o['train_loss'][6] for o in others],axis=0)))
            baseline=row['test_loss'][6]
            predictions.append(dict(name=row['name'],beta=reg['grid'][index],baseline_test_loss=baseline,
                gas_test_loss=row['test_loss'][index],fractional_improvement=1-row['test_loss'][index]/baseline))
    output=dict(status='COMPLETE_DEVELOPMENT_DESCRIPTOR_CHECK',rows=rows,skipped=skipped,predictions=predictions,
        mean_fractional_improvement=float(np.mean([r['fractional_improvement'] for r in predictions])) if predictions else None,registration=reg)
    save(dest/'result.json',output);print(json.dumps({k:v for k,v in output.items() if k not in ('rows','registration')},indent=2))
if __name__=='__main__':
    warnings.filterwarnings('ignore',module='astropy');main()
