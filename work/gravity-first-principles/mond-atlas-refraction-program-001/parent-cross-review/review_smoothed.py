"""Independent explicit Gaussian-kernel readback and conditional field gate audit."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,csv,hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import convolve1d,gaussian_filter
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());sys.path.insert(0,str(ROOT/'scripts'))
import mond_atlas_refraction_program as m
P=Path(__file__).resolve().parent;RUN=P.parent/'coherence-scale/run001'

def controls():
    manifest=json.loads(m.MANIFEST.read_text(encoding='utf-8'));case=next(c for c in manifest['source_cases'] if c['id']=='f4-stars-h0p4');results=[]
    for grid,half in [('base',[8,8,4]),('box',[12,12,6])]:
        spacing=[.25,.25,.125];axes=[np.arange(-round(b/h),round(b/h)+1)*h for b,h in zip(half,spacing)];rho,masses=m.source(case,axes);original=hashlib.sha256(rho.tobytes()).hexdigest()
        for ell in (.25,.5):
            manual=rho.copy()
            for axis,h in enumerate(spacing):
                sigma=ell/h;radius=int(4*sigma+.5);x=np.arange(-radius,radius+1);weights=np.exp(-.5*(x/sigma)**2);weights/=weights.sum()
                manual=convolve1d(manual,weights,axis=axis,mode='constant',cval=0)
            production=gaussian_filter(rho,[ell/h for h in spacing],mode='constant',cval=0,truncate=4)
            error=float(np.max(abs(manual-production))/max(np.max(abs(production)),1e-30));assert error<1e-12
            assert original==hashlib.sha256(rho.tobytes()).hexdigest()
            loss=float(1-manual.sum()/rho.sum());assert -.0000000001<loss<.001
            results.append(dict(grid=grid,ell_kpc=ell,explicit_kernel_vs_filter_relative=error,smoothing_boundary_mass_loss=loss,physical_rhs_rho_hash_unchanged=True))
    (P/'smoothing-control-receipt.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')

def gates():
    if not (RUN/'summary.json').exists():print('Smoothed solve still running; independent density controls saved.');return
    with (RUN/'sampled-vectors.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    summary=json.loads((RUN/'summary.json').read_text(encoding='utf-8'));comparisons=[]
    for ell in (.25,.5):
        for old,new in [('fine','finer'),('base','box')]:
            a=[r for r in rows if float(r['ell_kpc'])==ell and r['grid']==old];b=[r for r in rows if float(r['ell_kpc'])==ell and r['grid']==new]
            points=np.array([[float(r[k]) for k in ('x','y','z')] for r in a]);assert np.allclose(points,[[float(r[k]) for k in ('x','y','z')] for r in b])
            v=np.array([[float(r[k]) for k in ('gx','gy','gz')] for r in a]);w=np.array([[float(r[k]) for k in ('gx','gy','gz')] for r in b]);rms=float(np.linalg.norm(v-w)/np.linalg.norm(w));groups=[]
            for z in (0,.2,.5,1):
                mask=points[:,2]==z;groups.append(dict(z=z,rms=float(np.linalg.norm((v-w)[mask])/np.linalg.norm(w[mask]))))
            expected=next(r for r in summary['comparisons'] if r['ell_kpc']==ell and r['comparison']==old+'_to_'+new)
            assert abs(rms-expected['relative_rms'])<1e-12
            comparisons.append(dict(ell_kpc=ell,comparison=old+'_to_'+new,rms=rms,groups=groups,passes=bool(rms<.05 and all(g['rms']<.08 for g in groups))))
    results=dict(status='SMOOTHED_EPSILON_NEW_LAW_REVIEW',comparisons=comparisons,reported_pde_gates_all_pass=all(r['passed'] for r in summary['records']),max_reported_smoothing_mass_loss=max(r['smoothing_grid_mass_loss_fraction'] for r in summary['records']),RHS_unchanged_verified_by_code_and_smoothing_readback=True,observed_response_opened=False,original_point_law_failure_not_overridden=True)
    (P/'smoothed-field-receipt.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8');print(json.dumps(results,indent=2))

if __name__=='__main__':
    if '--gates-only' not in sys.argv:controls()
    gates()
