"""Full real detector fields and separate local covariance recovery controls."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,csv,time
from pathlib import Path
import numpy as np
from scipy.fft import idctn
from astropy.io import fits
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
from mond_atlas_native_selection import beam_from_history,beam_covariance,gaussian_kernel,convolve_spatial,spectral_matrix,integrated_gaussian,select_runs,FWHM_SIGMA
from mond_atlas_native_spectral import continuum_operator
from mond_atlas_noise_scale_channel import geometry,transform
P=ROOT/'work/gravity-first-principles/mond-atlas-selection-second-galaxy-001'
NP=ROOT/'work/gravity-first-principles/mond-atlas-noise-transfer-001'
PRIVATE=ROOT/'work/private/mond-atlas-selection-second-galaxy-001'
KINDS=['rotation','warp','streaming'];BRANCHES=['boxcar_independent','boxcar_hanning_full','boxcar_hanning_decimated']

def source(kind,grid,width,center,shift=0,size=121,speed=4,stream=2):
    yy,xx=np.indices((size,size),dtype=float);x=(xx-(size-2)/2-shift)*1.5;y=(yy-(size-2)/2)*3;r=np.hypot(x,y);safe=np.maximum(r,1e-30)
    theta=np.deg2rad(30)*r/(r+15) if kind=='warp' else 0
    velocity=speed*np.tanh(r/15)*(x*np.cos(theta)+y*np.sin(theta))/safe
    if kind=='streaming':velocity+=stream*np.tanh(r/15)*y/safe
    brightness=np.exp(-4*np.log(2)*r*r/900)
    return integrated_gaussian(grid[:,None,None],center+velocity,2,width)*brightness

def controls(op,kernel):
    n=op.shape[1];poly=np.column_stack((np.ones(n),np.arange(n)))
    assert np.max(abs(op@poly))<1e-10 and abs(kernel.sum()-1)<1e-14
    cubes=[source(k,np.arange(108.),1,50,speed=0,stream=0,size=25) for k in KINDS];assert all(np.array_equal(v,cubes[0]) for v in cubes)
    totals=[source(k,np.arange(108.),1,50,size=121).sum() for k in KINDS];assert max(totals)/min(totals)-1<1e-10
    a=source('rotation',np.arange(108.),1,50,size=121);large=source('rotation',np.arange(108.),1,50,size=161);assert abs(a.sum()/large.sum()-1)<1e-10
    rng=np.random.default_rng(9063000);v=rng.normal(size=(8,3,4));expected=np.zeros_like(v,dtype=bool)
    for y in range(3):
        for x in range(4):
            for c in range(6):
                if np.all(v[c:c+3,y,x]>.4):expected[c:c+3,y,x]=True
    assert np.array_equal(expected,select_runs(v,np.full(8,.2)))
    C=np.array([[2,.3],[.3,1.]]);t=np.array([1.,2.]);inv=np.linalg.inv(C);info=t@inv@t;noise=rng.multivariate_normal([0,0],C,size=20000);z=noise@inv@t/np.sqrt(info)
    assert abs(np.mean(z*z)-1)<.035
    return dict(status='PASS',continuum_polynomial_error=float(np.max(abs(op@poly))),flux_spread=float(max(totals)/min(totals)-1),padding_flux_error=float(abs(a.sum()/large.sum()-1)),mask_loop=True,scalar_GLS_gaussian_z2=float(np.mean(z*z)))

def metrics(noise,signal,positive,sigma,amplitude,reference):
    base=select_runs(noise,sigma);mask=select_runs(noise+amplitude*signal,sigma);truth=amplitude*positive;total=amplitude*reference
    return dict(captured_positive_fraction=float(positive.sum()/reference),true_full_retained=float(truth[mask].sum()/total),true_core_retained=float(truth[mask].sum()/truth.sum()),selected_noisy_full=float((noise+amplitude*signal)[mask].sum()/total),paired_selected_full=float(((noise+amplitude*signal)[mask].sum()-noise[base].sum())/total),post_continuum_core_fraction=float(signal.sum()/reference),peak_selected=bool(mask[np.unravel_index(np.argmax(signal),signal.shape)]),selected_voxel_fraction=float(mask.mean()))

def run():
    started=time.monotonic();out=P/'run001';out.mkdir(exist_ok=False);PRIVATE.mkdir(parents=True,exist_ok=True)
    try:
        hist=ROOT/'work/gravity-first-principles/mond-atlas-native-spectral-001/NGC3198.json';history=read_json(hist);prov=history['provenance'];indices=prov['parent_channel_indices_zero_based'];assert prov['direct_channel_mapping'] and len(indices)==72
        op=continuum_operator(prov['parent_channel_count'],prov['continuum_fit_parent_indices_zero_based'],indices,prov['polynomial_order'])
        cube=ROOT/history['source_path'];header=fits.getheader(cube);beam=beam_from_history(header);ncov=beam_covariance(beam['major_arcsec'],beam['minor_arcsec'],beam['pa_deg'],header['CDELT2']*3600,header['CDELT1']*3600);nk=gaussian_kernel(ncov);ek=gaussian_kernel(np.eye(2)*(30/FWHM_SIGMA/abs(header['CDELT1']*3600))**2-ncov)
        write_json(out/'controls.json',controls(op,nk))
        modelpath=NP/'run001/model-before-east.json';model=read_json(modelpath);mean=np.array(model['mean']);models={b:{k:(np.array(v) if k in ['power','C'] else v) for k,v in m.items()} for b,m in model['models'].items()};bands,U,r2=geometry(24,24)
        marginal=sum((U[:,mask]**2@models[b]['power'])[:,None]*np.diag(models[b]['C'])[None,:] for b,mask in bands.items());sigma_core=np.sqrt(marginal.reshape(24,24,72)).transpose(2,0,1);core_scale=float(np.median(sigma_core))
        geo=read_json(P/'geometry001/summary.json');assert geo['status']=='FULL_PADDING_ELIGIBLE';wr=[r for r in geo['rows'] if r['eligible'] and r['region']=='training'];er=[r for r in geo['rows'] if r['eligible'] and r['region']=='validation'];radius=geo['required_radius'];assert radius==40+len(ek)//2
        deps=[Path(__file__),P/'PREFLIGHT.md',P/'PADDING_ADDENDUM.md',P/'TRIAL_ADDENDUM.md',P/'geometry001/summary.json',hist,modelpath,ROOT/'scripts/mond_atlas_native_selection.py',ROOT/'scripts/mond_atlas_native_spectral.py',ROOT/'scripts/mond_atlas_noise_scale_channel.py',ROOT/'scripts/mond_atlas_common.py',ROOT/'docs/OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md',NP/'run001/summary.json']
        packet=ROOT/'work/private/mond-atlas-noise-transfer-001/cores.npz';assert digest(packet)==read_json(NP/'run001/summary.json')['private_sha256'];assert digest(cube)==history['source_sha256']
        write_json(out/'bindings.json',dict(files={p.relative_to(ROOT).as_posix():digest(p) for p in deps},cube_sha256=history['source_sha256'],packet_sha256=digest(packet),paper='https://arxiv.org/abs/0810.2125',observed_source_velocities_read=False))
        def patches(rows):
            natives=[];detectors=[]
            with fits.open(cube,memmap=True) as hdus:
                for r in rows:
                    y,x=r['center_y'],r['center_x'];patch=hdus[0].data.squeeze()[:,y-radius:y+radius+1,x-radius:x+radius+1].astype(float)*1000
                    assert np.isfinite(patch).all();k=radius-40;natives.append(patch[:,k:k+81,k:k+81]);detectors.append(convolve_spatial(patch,ek)[:,k:k+81,k:k+81])
            return np.array(natives),np.array(detectors)
        wn,wd=patches(wr);native_mean=wn.mean(axis=(0,2,3));detector_mean=wd.mean(axis=(0,2,3));sigma_full=np.sqrt(np.mean((wd-detector_mean[None,:,None,None])**2,axis=(0,2,3)));full_scale=float(np.median(sigma_full));del wn,wd
        frozen=out/'detector-before-east.json';write_json(frozen,dict(native_mean=native_mean.tolist(),detector_mean=detector_mean.tolist(),sigma=sigma_full.tolist(),full_scale=full_scale,core_scale=core_scale,western_centers=wr,eastern_centers=er,east_opened=False));frozenhash=digest(frozen)
        en,ed=patches(er);en-=native_mean[None,:,None,None];ed-=detector_mean[None,:,None,None]
        with np.load(packet) as z:east=z['validation']-mean
        rng=np.random.default_rng(9063001);co=np.empty((32,576,72))
        for b,mask in bands.items():co[:,mask]=(rng.normal(size=(32,int(mask.sum()),72))@np.linalg.cholesky(models[b]['C']).T)*np.sqrt(models[b]['power'])[None,:,None]
        simulated=idctn(co.reshape(32,24,24,72),norm='ortho',axes=(1,2));empcoef=transform(east,np.zeros(72));groups=[('empirical',east,empcoef),('gaussian',simulated,co),('noiseless',np.zeros((1,24,24,72)),np.zeros((1,576,72)))]
        np.savez_compressed(PRIVATE/'review-inputs.npz',full_native=en,full_detector=ed,core_gaussian=simulated)
        trials=[];gls=[];templates=[];template_arrays=[]
        for branch in BRANCHES:
            H,grid,width=spectral_matrix(op.shape[1],branch)
            for center in [10,36,61]:
                norm=None
                for kind in KINDS:
                    for shift in [0,-4,4]:
                        intrinsic=source(kind,grid,width,indices[center],shift=shift);pre=(H@intrinsic.reshape(len(grid),-1)).reshape(108,121,121);restored=convolve_spatial(pre,nk);positive=restored[indices];post=(op@restored.reshape(108,-1)).reshape(72,121,121)
                        if norm is None:norm=float(positive.max())
                        positive/=norm;post/=norm;reference=float(positive.sum());fullpost=post[:,20:101,20:101];fullpositive=positive[:,20:101,20:101];detect=convolve_spatial(post,ek)[:,20:101,20:101]
                        corepost=post[:,48:72,48:72];corepositive=positive[:,48:72,48:72];coeff=transform(corepost.transpose(1,2,0)[None],np.zeros(72))[0];weight=np.empty_like(coeff)
                        for b,mask in bands.items():weight[mask]=(coeff[mask]@np.linalg.inv(models[b]['C']))/models[b]['power'][:,None]
                        info=float(np.sum(weight*coeff));assert info>0;se=1/np.sqrt(info);case=dict(branch=branch,center=center,kind=kind,shift=shift)
                        templates.append(dict(**case,reference=reference,full_capture=float(fullpositive.sum()/reference),core_capture=float(corepositive.sum()/reference),GLS_standard_error=se))
                        template_arrays.append(dict(case=case,source=corepost,positive=corepositive,reference=reference,weight=weight,information=info))
                        for amplitude in [5,10]:
                            for group,noise,coeffnoise in groups:
                                errors=np.einsum('nmc,mc->n',coeffnoise,weight)/info
                                for j,n in enumerate(noise):
                                    trials.append(dict(detector='native_core',group=group,draw=j,**case,amplitude=amplitude,**metrics(n.transpose(2,0,1),corepost,corepositive,sigma_core,amplitude*core_scale,reference)))
                                    gls.append(dict(group=group,draw=j,**case,amplitude=amplitude,estimated_amplitude=float(amplitude*core_scale+errors[j]),injected_amplitude=amplitude*core_scale,standard_error=se,standardized_error=float(errors[j]/se)))
                            for group,noises,detectors in [('empirical',en,ed),('noiseless',np.zeros_like(en[:1]),np.zeros_like(ed[:1]))]:
                                for j,(n,d) in enumerate(zip(noises,detectors)):
                                    amp=amplitude*full_scale;mask=select_runs(d+amp*detect,sigma_full);base=select_runs(d,sigma_full);truth=amp*fullpositive;total=amp*reference
                                    values=dict(captured_positive_fraction=float(fullpositive.sum()/reference),true_full_retained=float(truth[mask].sum()/total),true_core_retained=float(truth[mask].sum()/truth.sum()),selected_noisy_full=float((n+amp*fullpost)[mask].sum()/total),paired_selected_full=float(((n+amp*fullpost)[mask].sum()-n[base].sum())/total),post_continuum_core_fraction=float(fullpost.sum()/reference),peak_selected=bool(mask[np.unravel_index(np.argmax(detect),detect.shape)]),selected_voxel_fraction=float(mask.mean()))
                                    trials.append(dict(detector='full_30arcsec',group=group,draw=j,**case,amplitude=amplitude,**values))
            print(branch,'finished',flush=True)
        write_csv(out/'trials.csv',trials);write_csv(out/'GLS-trials.csv',gls);write_csv(out/'templates.csv',templates)
        np.savez_compressed(PRIVATE/'template-review.npz',source=np.array([t['source'] for t in template_arrays]),positive=np.array([t['positive'] for t in template_arrays]),reference=np.array([t['reference'] for t in template_arrays]),weight=np.array([t['weight'] for t in template_arrays]),information=np.array([t['information'] for t in template_arrays]))
        ag=[]
        for detector in ['native_core','full_30arcsec']:
            for group in ['empirical','gaussian','noiseless']:
                for t in templates:
                    for amplitude in [5,10]:
                        rr=[r for r in trials if r['detector']==detector and r['group']==group and r['amplitude']==amplitude and all(r[k]==t[k] for k in ['branch','center','kind','shift'])]
                        if not rr:continue
                        vals={k:float(np.mean([r[k] for r in rr])) for k in ['true_full_retained','true_core_retained','paired_selected_full','peak_selected']};ag.append(dict(detector=detector,group=group,branch=t['branch'],center=t['center'],kind=t['kind'],shift=t['shift'],amplitude=amplitude,n=len(rr),**vals,recovery_flag=bool(vals['true_core_retained']>=.9 and abs(vals['paired_selected_full']-1)<=.1)))
        write_csv(out/'case-summary.csv',ag)
        gs=[]
        for group in ['empirical','gaussian','noiseless']:
            for t in templates:
                rr=[r for r in gls if r['group']==group and r['amplitude']==5 and all(r[k]==t[k] for k in ['branch','center','kind','shift'])];z=np.array([r['standardized_error'] for r in rr]);gs.append(dict(group=group,branch=t['branch'],center=t['center'],kind=t['kind'],shift=t['shift'],n=len(z),z_mean=float(z.mean()),z2_mean=float(np.mean(z*z)),z_sd=float(z.std(ddof=1)) if len(z)>1 else 0))
        write_csv(out/'GLS-summary.csv',gs);assert digest(frozen)==frozenhash
        result=dict(status='CONDITIONAL_SECOND_GALAXY_INJECTION_COMPLETE',trials=len(trials),GLS_trials=len(gls),cases=162,full_western_fields=9,full_eastern_fields=7,core_eastern_fields=18,gaussian_core_draws=32,recovery_flags={f'{d}_{g}':sum(r['recovery_flag'] for r in ag if r['detector']==d and r['group']==g) for d,g in [('native_core','empirical'),('native_core','gaussian'),('full_30arcsec','empirical'),('full_30arcsec','noiseless')]},GLS_z2_ranges={g:[min(r['z2_mean'] for r in gs if r['group']==g),max(r['z2_mean'] for r in gs if r['group']==g)] for g in ['empirical','gaussian']},full_capture_range=[min(t['full_capture'] for t in templates),max(t['full_capture'] for t in templates)],core_capture_range=[min(t['core_capture'] for t in templates),max(t['core_capture'] for t in templates)],private_files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in PRIVATE.glob('*.npz')},elapsed_seconds=time.monotonic()-started,observed_velocity_arrays_read=0,admitted_observed_likelihoods=0)
        write_json(out/'summary.json',result);print(json.dumps(result,indent=2));assert time.monotonic()-started<300;assert sum(p.stat().st_size for p in PRIVATE.glob('*.npz'))<100*1024**2
    except Exception as e:write_json(out/'failure.json',dict(error=repr(e)));raise
if __name__=='__main__':run()
