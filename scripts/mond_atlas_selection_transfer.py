"""Frozen kinematic-mask transfer using existing conditional native pipeline."""
from pathlib import Path
import json, itertools, shutil
import numpy as np
from astropy.io import fits
from threadpoolctl import threadpool_limits
from mond_atlas_native_selection import *
from mond_atlas_native_spectral import continuum_operator
from run_mond_atlas_native_selection import sha, write_json, write_csv, controls

ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT/'work/gravity-first-principles/mond-atlas-selection-transfer-001'
OLD=ROOT/'work/gravity-first-principles/mond-atlas-native-selection-001/run-001'

def intrinsic(kind, grid, width, center, speed=4., radial=2.):
    yy,xx=np.indices((81,81),dtype=float)
    x=(xx-40)*1.5; y=(yy-40)*1.5/.5
    r=np.hypot(x,y); safe=np.maximum(r,1e-30)
    theta=np.deg2rad(30)*r/(r+15) if kind=='warp' else np.zeros_like(r)
    velocity=speed*np.tanh(r/15)*(x*np.cos(theta)+y*np.sin(theta))/safe
    if kind=='streaming': velocity+=radial*np.tanh(r/15)*y/safe
    surface=np.exp(-4*np.log(2)*r*r/30**2)
    profile=integrated_gaussian(np.asarray(grid)[:,None,None],center+velocity,2.,width)
    return profile*surface,velocity

def new_controls():
    grid=np.arange(-40,41,dtype=float)
    cubes=[intrinsic(k,grid,1,0,0,0)[0] for k in ['rotation','warp','streaming']]
    identity=max(float(np.max(abs(v-cubes[0]))) for v in cubes)
    values=[intrinsic(k,grid,1,0) for k in ['rotation','warp','streaming']]
    antisym=max(float(np.max(abs(v+v[::-1,::-1]))) for _,v in values)
    totals=[float(c.sum()) for c,_ in values]
    flux=max(abs(v/totals[0]-1) for v in totals)
    rng=np.random.default_rng(9062601); a=rng.normal(size=(11,5,7))
    expected=np.zeros_like(a,dtype=bool)
    for y,x in itertools.product(range(5),range(7)):
        for c in range(9):
            if all(a[c:c+3,y,x]>.3): expected[c:c+3,y,x]=True
    mask=np.array_equal(expected,select_runs(a,np.ones(11)*.15))
    result=dict(zero_velocity_identity=identity,velocity_antisymmetry=antisym,
                integrated_flux_relative_spread=flux,independent_mask_equal=mask,
                pre_continuum_positive=all(c.min()>=0 for c,_ in values))
    result['passed']=identity<1e-12 and antisym<1e-12 and flux<1e-5 and mask and result['pre_continuum_positive']
    return result

def run():
    out=PKG/'run001'; out.mkdir(exist_ok=False)
    config=json.loads((ROOT/'configs/mond_atlas_native_selection_v1.json').read_text())
    paths=[Path(__file__),PKG/'PREFLIGHT.md',ROOT/'configs/mond_atlas_native_selection_v1.json',
           ROOT/'scripts/mond_atlas_native_selection.py',ROOT/'scripts/mond_atlas_native_spectral.py',
           ROOT/'scripts/run_mond_atlas_native_selection.py',ROOT/config['native_history'],
           ROOT/config['cube_path'],OLD/'support.json',OLD/'prospective-bindings.json',
           ROOT/'work/private/mond-atlas-native-selection-001/run-001/selection-and-support.npz']
    bound={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    assert bound[config['cube_path']]==config['cube_sha256']
    assert shutil.disk_usage(ROOT).free>8e9
    write_json(out/'pre-access-bindings.json',dict(bindings=bound,new_arrays_opened=False,
                source_background_previous_exposure=True,new_gravity_scores=0))
    hist=json.loads((ROOT/config['native_history']).read_text()); prov=hist['provenance']
    indices=prov['parent_channel_indices_zero_based']
    op=continuum_operator(prov['parent_channel_count'],prov['continuum_fit_parent_indices_zero_based'],indices,prov['polynomial_order'])
    header=fits.getheader(ROOT/config['cube_path']); beam=beam_from_history(header)
    ncov=beam_covariance(beam['major_arcsec'],beam['minor_arcsec'],beam['pa_deg'],header['CDELT2']*3600,header['CDELT1']*3600)
    ecov=np.eye(2)*(30/FWHM_SIGMA/abs(header['CDELT1']*3600))**2-ncov
    ctl=controls(config,op,prov,ncov,ecov); ctl['new_template_controls']=new_controls()
    write_json(out/'controls.json',ctl); assert ctl['new_template_controls']['passed']
    nk=gaussian_kernel(ncov); ek=gaussian_kernel(ecov)
    support=json.loads((OLD/'support.json').read_text())
    cache=np.load(paths[-1]); sigma=cache['sigma_jy_per_native_beam']; scale=float(np.median(sigma))
    positions=support['selected_positions_yx_zero_based']; patches=[]
    with fits.open(ROOT/config['cube_path'],memmap=True) as f:
        for y,x in positions:
            radius=40+len(ek)//2
            extended=f[0].data.squeeze()[:,y-radius:y+radius+1,x-radius:x+radius+1].astype(float)
            d=convolve_spatial(extended,ek)[:,radius-40:radius+41,radius-40:radius+41]-cache['median'][:,None,None]
            n=extended[:,radius-40:radius+41,radius-40:radius+41]-cache['native_median'][:,None,None]
            patches.append((n,d))
    fluxfactor=1.5**2/(np.pi*beam['major_arcsec']*beam['minor_arcsec']/(4*np.log(2)))*abs(header['CDELT3'])/1000
    rng=np.random.default_rng(9062607); rows=[]; noise=[]; template_rows=[]
    for branch in config['spectral_branches']:
        h,grid,width=spectral_matrix(op.shape[1],branch); covariance=op@h@h.T@op.T
        templates=[]
        for center in [10,20,30]:
            norm=None
            for kind in ['rotation','warp','streaming']:
                c,_=intrinsic(kind,grid,width,indices[center])
                pre=(h@c.reshape(len(grid),-1)).reshape(h.shape[0],81,81)
                restored=convolve_spatial(pre,nk)
                positive=restored[indices]
                ds_pre=convolve_spatial(positive,ek)
                if norm is None: norm=float(ds_pre.max())
                ns=(op@restored.reshape(len(restored),-1)).reshape(42,81,81)/norm
                ds=convolve_spatial(ns,ek); ps=positive/norm
                templates.append((center,kind,ns,ds,ps))
                template_rows.append(dict(branch=branch,center=center,kind=kind,reference_sum=float(ps.sum()),
                     peak_detector_relative_to_rotation=float(ds_pre.max()/norm),post_continuum_sum_ratio=float(ns.sum()/ps.sum())))
        def evaluate(n,d,group,draw,local_sigma):
            baseline=select_runs(d,local_sigma)
            for center,kind,ns,ds,ps in templates:
                for amplitude in [5,10]:
                    result=recovery(n,d,ns,ds,ps,local_sigma,amplitude*scale,fluxfactor,baseline)
                    rows.append(dict(group=group,branch=branch,draw=draw,center=center,kind=kind,amplitude=amplitude,**result))
        zero=np.zeros((42,81,81)); evaluate(zero,zero,'noiseless',0,sigma)
        for i,(n,d) in enumerate(patches): evaluate(n,d,'empirical',i,sigma)
        synsigma=scale*np.sqrt(np.diag(covariance)/np.diag(covariance).mean())
        for i in range(16):
            n,d=conditional_noise(rng,(81,81),covariance,nk,ek,scale)
            noise.append(dict(branch=branch,draw=i,rms_over_target=float(np.sqrt(np.mean(d*d))/scale),selected_fraction=float(select_runs(d,synsigma).mean())))
            evaluate(n,d,'gaussian',i,synsigma)
        print(branch,'complete',flush=True)
    write_csv(out/'trials.csv',rows); write_csv(out/'noise.csv',noise); write_csv(out/'templates.csv',template_rows)
    aggregates=[]; paired=[]
    metrics=['true_flux_fraction_retained','peak_selected','paired_selected_flux_difference_over_reference','selected_noisy_flux_over_reference']
    for group,branch,center,amplitude in itertools.product(['empirical','gaussian','noiseless'],config['spectral_branches'],[10,20,30],[5,10]):
        subset=[r for r in rows if (r['group'],r['branch'],r['center'],r['amplitude'])==(group,branch,center,amplitude)]
        bykind={k:sorted([r for r in subset if r['kind']==k],key=lambda r:r['draw']) for k in ['rotation','warp','streaming']}
        for kind,rr in bykind.items():
            a=dict(group=group,branch=branch,center=center,amplitude=amplitude,kind=kind,n=len(rr))
            for metric in metrics:
                v=np.array([r[metric] for r in rr],float)
                a[metric+'_mean']=float(v.mean()); a[metric+'_sd']=float(v.std(ddof=1)) if len(v)>1 else 0.
                a[metric+'_min']=float(v.min()); a[metric+'_max']=float(v.max())
                a[metric+'_conditional_mc_se']=float(v.std(ddof=1)/np.sqrt(len(v))) if group=='gaussian' else ''
            a['adequate_recovery']=a[metrics[0]+'_mean']>=.9 and abs(a[metrics[2]+'_mean']-1)<=.1
            aggregates.append(a)
        for kind in ['warp','streaming']:
            v=np.array([r[metrics[0]]-s[metrics[0]] for r,s in zip(bykind[kind],bykind['rotation'])])
            paired.append(dict(group=group,branch=branch,center=center,amplitude=amplitude,kind=kind,
                 mean_retention_difference=float(v.mean()),sd=float(v.std(ddof=1)) if len(v)>1 else 0.,
                 minimum=float(v.min()),maximum=float(v.max()),transfer_gate_pass=abs(float(v.mean()))<=.05))
    write_csv(out/'case-summary.csv',aggregates); write_csv(out/'paired-morphology.csv',paired)
    assert all(sha(ROOT/p)==v for p,v in bound.items())
    write_json(out/'summary.json',dict(status='CONDITIONAL_SELECTION_TRANSFER_EXECUTED',admission='SOURCE_BLOCKED',
        counts={g:sum(r['group']==g for r in rows) for g in ['empirical','gaussian','noiseless']},
        recovery_cases_pass={g:sum(r['adequate_recovery'] for r in aggregates if r['group']==g) for g in ['empirical','gaussian','noiseless']},
        transfer_pairs_pass={g:sum(r['transfer_gate_pass'] for r in paired if r['group']==g) for g in ['empirical','gaussian','noiseless']},
        cases_per_group=54,pairs_per_group=36,new_private_bytes=0,new_gravity_scores=0,
        admitted_observed_likelihoods=0,all_inputs_reverified=True))

if __name__=='__main__':
    with threadpool_limits(limits=1): run()
