"""Post-result, explicitly development-only robustness checks."""
import run_gravity_observable_patterns as base
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
import cupy as cp
from astropy.io import fits
from cupyx.scipy.signal import fftconvolve
from scipy.signal import fftconvolve as cpu_convolve

ROOT=base.ROOT
DEST=Path(sys.argv[1]);DEST.mkdir(parents=True,exist_ok=False)
(DEST/'runner.py').write_bytes(Path(__file__).read_bytes())
PARENT=ROOT/'work/gravity-first-principles/observable-pattern-campaign-002'
MAP=ROOT/'work/gravity-first-principles/observable-map-pilot-001'
parent=json.loads((PARENT/'result.json').read_text())
registration=dict(scope='Post-result development diagnostics, not independent confirmation or a discovery significance test.',
    parents={str(p.relative_to(ROOT)):base.sha(p) for p in [PARENT/'result.json',MAP/'map_audit.json']},
    bootstrap='2000 paired whole-galaxy resamples of fixed outer predictions; does not repeat model selection.',
    permutation='Eight within-galaxy row permutations of added feature columns together; baseline local columns fixed; rerun inner selection and outer fit for trees and GPU kernel models, both splits, nominal mass only.',
    map_geometry='Match Gaussian restoring-beam covariance before applying projected elliptical filters; local sigma-major=30 arcsec, surrounding=60 arcsec; q=.284,.334,.384 and PA=28.6,33.6,38.6 degrees. Sensitivity brackets, not posterior draws.',
    map_boundary='Aperture sensitivity does not identify line-of-sight depth, disk thickness or gravity. No noise covariance or independent measurement.')
base.save(DEST/'registration.json',registration)

def covariance(major,minor,pa):
    # x pixel increases west for these maps; y increases north.
    angle=np.deg2rad(pa)
    v=np.array([-np.sin(angle),np.cos(angle)])
    u=np.array([np.cos(angle),np.sin(angle)])
    return major**2*np.outer(v,v)+minor**2*np.outer(u,u)

def kernel(cov):
    eig=np.linalg.eigvalsh(cov);assert eig.min()>0
    radius=int(np.ceil(5*np.sqrt(eig.max())))
    yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
    inverse=np.linalg.inv(cov)
    k=np.exp(-.5*(inverse[0,0]*xx**2+2*inverse[0,1]*xx*yy+inverse[1,1]*yy**2))
    return k/k.sum()

def geometry():
    audit=json.loads((MAP/'map_audit.json').read_text())
    maps={}
    for label in ('NATURAL','ROBUST'):
        p=ROOT/f'work/private/open-gravity-rg-12gal-source-only-v1/NGC3198__THINGS__HI_MOM0_{label}.fits'
        a=np.squeeze(fits.getdata(p)).astype(float)
        mask=np.isfinite(a)&(a!=0)
        beam=audit['beams'][label];pixel=beam['pixel_arcsec']
        bcov=covariance(beam['major_arcsec']/2.354820045,beam['minor_arcsec']/2.354820045,beam['pa_deg'])
        maps[label]=(cp.asarray(np.where(mask,a/beam['area_arcsec2']/1000,0.)),cp.asarray(mask.astype(float)),bcov,pixel)
    tests={}
    k=kernel(covariance(3,2,31))
    a=np.random.default_rng(19).normal(size=(79,83))
    tests['fft_cpu_gpu_max_abs']=float(np.max(abs(cp.asnumpy(fftconvolve(cp.asarray(a),cp.asarray(k),mode='same'))-cpu_convolve(a,k,mode='same'))))
    # Analytic Gaussian convolution covariance addition on a sufficiently large grid.
    original=covariance(4,2,-70);target=covariance(10,6,34)
    image=cpu_convolve(kernel(original),kernel(target-original),mode='full')
    yy,xx=np.indices(image.shape);xx=xx-(image.shape[1]-1)/2;yy=yy-(image.shape[0]-1)/2
    measured=np.array([[np.sum(image*xx*xx),np.sum(image*xx*yy)],[np.sum(image*xx*yy),np.sum(image*yy*yy)]])
    tests['manufactured_covariance_max_relative']=float(np.max(abs(measured-target))/np.max(abs(target)))
    assert tests['fft_cpu_gpu_max_abs']<1e-10 and tests['manufactured_covariance_max_relative']<1e-4
    contrasts={}; supports={}; comparisons=[]
    grid=np.zeros((1024,1024),bool);grid[::20,::20]=True
    for q in (.284,.334,.384):
        for pa in (28.6,33.6,38.6):
            scenario=f'q{q}_pa{pa}'
            bylabel={};coverage={}
            for label,(value,mask,bcov,pixel) in maps.items():
                smoothed=[];covered=[]
                for size in (30.,60.):
                    target=covariance(size,size*q,pa)
                    k=cp.asarray(kernel((target-bcov)/pixel**2))
                    weight=fftconvolve(mask,k,mode='same')
                    mean=fftconvolve(value,k,mode='same')/cp.maximum(weight,1e-12)
                    smoothed.append(cp.asnumpy(mean));covered.append(cp.asnumpy(weight)>.95)
                bylabel[label]=smoothed[1]/np.maximum(smoothed[0],1e-20)-1
                coverage[label]=covered[0]&covered[1]
            common=coverage['NATURAL']&coverage['ROBUST']
            valid=common&grid
            comparisons.append(dict(scenario=scenario,locations=int(valid.sum()),
                rho=float(base.spearmanr(bylabel['NATURAL'][valid],bylabel['ROBUST'][valid]).statistic),
                sign_agreement=float(np.mean(np.sign(bylabel['NATURAL'][valid])==np.sign(bylabel['ROBUST'][valid])))))
            contrasts[scenario]=bylabel['ROBUST'];supports[scenario]=common
    shared=np.logical_and.reduce(list(supports.values()))&grid
    stack=np.array([a[shared] for a in contrasts.values()])
    reference=contrasts['q0.334_pa33.6'][shared]
    # Near zero signs are fragile; report both all points and explicit |contrast|>.05 subset.
    same=(np.sign(stack)==np.sign(reference)).all(axis=0)
    material=abs(reference)>.05
    result=dict(controls=tests,processing_agreement=comparisons,common_locations=int(shared.sum()),
        all_geometry_sign_agreement_fraction=float(same.mean()),
        material_locations=int(material.sum()),material_sign_agreement_fraction=float(same[material].mean()),
        median_geometry_contrast_range=float(np.median(np.ptp(stack,axis=0))),
        reference_contrast_quantiles=np.quantile(reference,[.05,.5,.95]).tolist())
    base.save(DEST/'geometry.json',result)
    print('Geometry',json.dumps(result),flush=True)

def statistics():
    models=parent['models']; rng=np.random.default_rng(5934)
    indices=rng.integers(0,139,size=(2000,139)); intervals=[]
    for model in models:
        if model['scenario']!='nominal':continue
        local=next(m for m in models if m['scenario']==model['scenario'] and m['salt']==model['salt'] and m['algorithm']==model['algorithm'] and m['group']=='local_coverage')
        before=np.array(local['after']);after=np.array(model['after'])
        draws=1-after[indices].mean(axis=1)/before[indices].mean(axis=1)
        intervals.append(dict(algorithm=model['algorithm'],salt=model['salt'],group=model['group'],
            incremental_gain=float(1-after.mean()/before.mean()),
            fixed_prediction_bootstrap_quantiles=np.quantile(draws,[.025,.5,.975]).tolist()))
    base.save(DEST/'bootstrap.json',intervals)
    reg=parent['registration'];names=reg['names']
    raw={g['name']:g for g in json.loads((ROOT/'configs/sparc_rotation_curves_full_v1.json').read_text())['galaxies'] if g['name'] in names}
    photo={g['galaxy']:g for g in json.loads((ROOT/'configs/sparc_surface_brightness_exploration_v1.json').read_text())['galaxies']}
    original=base.build(raw,photo,names,[.5,.7]);results=[]
    for algorithm in ('trees','gpu_rbf_features'):
        for group in ('photometry_multiscale','gas_force_proxy'):
            for salt in reg['salts']:
                outer=base.foldmap(names,range(139),salt,5)
                real=next(m for m in models if m['scenario']=='nominal' and m['salt']==salt[-1] and m['algorithm']==algorithm and m['group']==group)
                null=[]
                for iteration in range(8):
                    objects=copy.deepcopy(original)
                    permrng=np.random.default_rng(5935+iteration)
                    for o in objects:
                        x=o['X'][group];x[:,10:]=x[permrng.permutation(len(x)),10:]
                    prediction=[None]*139
                    for fold in range(5):
                        train=[i for i in range(139) if outer[i]!=fold];test=[i for i in range(139) if outer[i]==fold]
                        inner=base.foldmap(names,train,salt+f'-inner-{fold}',3);losses=[]
                        for parameter in reg['algorithms'][algorithm]:
                            loss=[]
                            for infold in range(3):
                                tr=[i for i in train if inner[i]!=infold];va=[i for i in train if inner[i]==infold]
                                pp=base.train_predict(objects,tr,va,group,algorithm,parameter)
                                loss.extend(np.mean((objects[i]['target']-p)**2) for i,p in zip(va,pp))
                            losses.append(np.mean(loss))
                        best=reg['algorithms'][algorithm][int(np.argmin(losses))]
                        pp=base.train_predict(objects,train,test,group,algorithm,best)
                        for i,p in zip(test,pp):prediction[i]=p
                    null.append(base.score(objects,prediction)['fractional_mse_gain'])
                result=dict(algorithm=algorithm,group=group,salt=salt[-1],real_gain=real['fractional_mse_gain'],
                    shuffled_gains=null,shuffles_at_least_as_good=int(np.sum(np.array(null)>=real['fractional_mse_gain'])))
                results.append(result);base.save(DEST/'permutations_partial.json',results)
                print('Permutation',json.dumps(result),flush=True)
    base.save(DEST/'permutations.json',results)

start=time.perf_counter()
geometry()
statistics()
base.save(DEST/'completion.json',dict(status='COMPLETED',seconds=time.perf_counter()-start))
