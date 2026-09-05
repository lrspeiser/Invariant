"""Observable-structure development campaign. No claim of 3D reconstruction."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
import numpy as np
import cupy as cp
import cupyx
from astropy.io import fits
from scipy.ndimage import gaussian_filter as cpu_filter
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from cupyx.scipy.ndimage import gaussian_filter as gpu_filter

ROOT = Path(__file__).resolve().parents[1]

def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def map_audit(dest):
    source = ROOT/'work/private/open-gravity-rg-12gal-source-only-v1'
    inventory = []
    for path in sorted(source.iterdir()):
        a = np.squeeze(fits.getdata(path)).astype(float)
        h = fits.getheader(path)
        history = '\n'.join(str(v) for v in h.get('HISTORY', []))
        inventory.append(dict(file=path.name, sha256=sha(path), shape=list(a.shape),
            bunit=h.get('BUNIT'), finite_fraction=float(np.isfinite(a).mean()),
            zero_fraction=float((a == 0).mean()), negative_fraction=float((a < 0).mean()),
            blanking_history=[line for line in history.splitlines() if any(s in line for s in ('NBLANK', 'PIXVAL', 'CLEAN BMAJ'))]))
    # Fixed projected angular scales. No distance, tilt, circular velocities or mass conversion used.
    scales = [15., 30., 60.]
    arrays, coverages, originals = {}, {}, {}
    beam_info = {}
    for label in ('NATURAL', 'ROBUST'):
        path = source/f'NGC3198__THINGS__HI_MOM0_{label}.fits'
        a = np.squeeze(fits.getdata(path)).astype(float)
        h = fits.getheader(path)
        history = '\n'.join(str(v) for v in h['HISTORY'])
        matches = re.findall(r'CLEAN BMAJ=\s*([\d.E+-]+) BMIN=\s*([\d.E+-]+) BPA=\s*([\d.E+-]+)', history)
        assert matches, 'Need documented beam; never assume a missing header value.'
        bmaj, bmin, bpa = map(float, matches[-1])
        pixel = abs(h['CDELT1'])*3600
        assert abs(abs(h['CDELT2'])*3600-pixel)<1e-5
        # Jy/beam m/s -> Jy/arcsec^2 km/s. No gas mass conversion.
        beam_area = np.pi/(4*np.log(2))*(bmaj*3600)*(bmin*3600)
        a = a/1000/beam_area
        valid = np.isfinite(a) & (a != 0)
        # The history states blanked pixels were replaced by zero. Exclude these
        # conservatively; they are not measured empty regions or upper limits.
        value = cp.asarray(np.where(valid, a, 0.))
        mask = cp.asarray(valid.astype(float))
        beam_info[label] = dict(major_arcsec=bmaj*3600, minor_arcsec=bmin*3600,
            pa_deg=bpa, pixel_arcsec=pixel, area_arcsec2=beam_area)
        arrays[label], coverages[label] = [], []
        for scale in scales:
            # All effective scales are far above the native beam. Native beams
            # are not exactly matched; processing agreement is a sensitivity check.
            weight = gpu_filter(mask, scale/pixel, mode='constant')
            smoothed = gpu_filter(value, scale/pixel, mode='constant')/cp.maximum(weight, 1e-15)
            arrays[label].append(cp.asnumpy(smoothed))
            coverages[label].append(cp.asnumpy(weight))
        originals[label] = a
    tests = {}
    rng = np.random.default_rng(5931)
    small = rng.normal(size=(67, 71))
    tests['gpu_cpu_filter_max_abs_error'] = float(np.max(np.abs(
        cp.asnumpy(gpu_filter(cp.asarray(small), 2.3, mode='constant'))-
        cpu_filter(small, 2.3, mode='constant'))))
    mask = np.ones((67, 71)); mask[20:30, 25:35] = 0
    c = gpu_filter(cp.asarray(7*mask), 3., mode='constant')/gpu_filter(cp.asarray(mask), 3., mode='constant')
    tests['masked_constant_max_abs_error'] = float(cp.max(cp.abs(c-7)).get())
    assert max(tests.values()) < 1e-10
    results = []
    grid = np.zeros_like(originals['ROBUST'], dtype=bool)
    grid[::20, ::20] = True  # sparse display sampling, not independent statistical observations
    for j in (1, 2):
        common = np.logical_and.reduce([coverages[l][k]>.95 for l in arrays for k in (0, j)])
        x = arrays['NATURAL'][j]/np.maximum(arrays['NATURAL'][0], 1e-20)-1
        y = arrays['ROBUST'][j]/np.maximum(arrays['ROBUST'][0], 1e-20)-1
        good = common & grid & np.isfinite(x) & np.isfinite(y)
        assert good.sum()>10
        results.append(dict(scale_sigma_arcsec=scales[j], reference_sigma_arcsec=15,
            high_support_pixels=int(common.sum()), sampled_locations=int(good.sum()),
            natural_robust_spearman=float(spearmanr(x[good], y[good]).statistic),
            sign_agreement=float(np.mean(np.sign(x[good])==np.sign(y[good]))),
            median_absolute_contrast_difference=float(np.median(abs(x[good]-y[good])))))
    # Save a visual and numerical source-only descriptors. No rotation target here.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    valid = originals['ROBUST'] != 0
    im = axs[0].imshow(np.where(valid, np.log10(np.maximum(originals['ROBUST'],1e-12)), np.nan), origin='lower', cmap='viridis')
    axs[0].set_title('Measured HI emission\nblanked pixels hidden')
    fig.colorbar(im, ax=axs[0], label='log10 Jy / arcsec² km/s', shrink=.75)
    for ax,j in zip(axs[1:],(1,2)):
        common = np.logical_and.reduce([coverages[l][k]>.95 for l in arrays for k in (0,j)])
        contrast = arrays['ROBUST'][j]/np.maximum(arrays['ROBUST'][0],1e-20)-1
        im=ax.imshow(np.where(common,contrast,np.nan),origin='lower',cmap='coolwarm',vmin=-.6,vmax=.6)
        ax.set_title(f'Surroundings at {scales[j]:g} arcsec / local at 15\nGaussian widths; projected structure')
        fig.colorbar(im,ax=ax,label='relative brightness contrast',shrink=.75)
    for ax in axs:
        ax.set_xlim(200,850); ax.set_ylim(150,850); ax.set_xlabel('image pixel')
    fig.suptitle('NGC3198: observable multiscale structure, not a gravity or 3D map')
    fig.savefig(dest/'NGC3198-multiscale.png',dpi=150); plt.close(fig)
    np.savez_compressed(dest/'map_descriptors.npz', **{f'{l}_{s}':a for l in arrays for s,a in zip(scales,arrays[l])})
    out=dict(inventory=inventory, beams=beam_info, scales_sigma_arcsec=scales, controls=tests,
        processing_agreement=results, caveats=[
        'Zero-filled blanking prevents identifying physical voids from these delivered maps.',
        'Finite nonzero support is a conservative analysis mask, not a calibrated detection selection function.',
        'Natural and robust products share observations and masking; agreement is not independent replication.',
        'Native beams differ; broad Gaussian smoothing is not exact beam matching.',
        'No propagated HI noise covariance, deprojection, mass conversion or gravity-response fit.',
        'Sampled locations are correlated; no pixel-based significance is claimed.'])
    save(dest/'map_audit.json',out)
    return out

def foldmap(names, indices, salt, k):
    order=sorted(indices,key=lambda i:hashlib.sha256((salt+'|'+names[i]).encode()).hexdigest())
    return {i:j%k for j,i in enumerate(order)}

def build(raw, photo, names, ml):
    out=[]
    for name in names:
        r,v,e,gas,disk,bulge=np.array(raw[name]['rows'],float).T
        sb=np.array(photo[name]['rows'],float).sum(axis=1)
        vb=gas*abs(gas)+ml[0]*disk**2+ml[1]*bulge**2
        assert np.all(vb>0) and np.all(np.diff(r)>0)
        y=vb/r/3702.81458; vrar=np.sqrt(vb/(-np.expm1(-np.sqrt(y))))
        denom=gas**2+ml[0]*disk**2+ml[1]*bulge**2
        fg=gas**2/denom; fb=ml[1]*bulge**2/denom
        logr=np.log(r); ly=np.log10(y); logspan=logr[-1]-logr[0]
        pos=(logr-logr[0])/logspan
        local=np.column_stack([ly,ly**2,np.log10(r/r[np.argmax(disk**2)]),fg,fb,
            np.log10(1+sb),np.gradient(np.log(vb),logr),pos,pos**2,np.full(len(r),logspan)])
        # Published radial surface brightness, not a 2D aperture or measured 3D density.
        # Finite-domain annular area weights; no source invented outside observed radii.
        edges=np.r_[r[0],(r[:-1]+r[1:])/2,r[-1]]
        area=np.diff(edges**2)
        dx=r[None,:]-r[:,None]
        extra=[]
        for reach in (.5,2.,8.):
            for side in (-1,1):
                w=np.exp(-abs(dx)/reach)*(dx*side>=0)*area[None,:]
                mean=w@sb/w.sum(axis=1)
                extra.append(np.log10(1+mean)-np.log10(1+sb))
        sbextra=np.column_stack(extra)
        dxlog=logr[None,:]-logr[:,None]; cell=np.gradient(logr)
        gasextra=[]
        for side in (-1,1):
            w=np.exp(-abs(dxlog))*(dxlog*side>=0)*cell[None,:]
            gasextra.append(w@fg/w.sum(axis=1)-fg)
        X={'local_coverage':local,'photometry_multiscale':np.column_stack([local,sbextra]),
           'gas_force_proxy':np.column_stack([local,*gasextra])}
        out.append(dict(name=name,r=r,v=v,error=e,vrar=vrar,target=v/vrar-1,X=X))
    return out

def combine(objects, indices, group):
    X=np.concatenate([objects[i]['X'][group] for i in indices])
    y=np.concatenate([objects[i]['target'] for i in indices])
    w=np.concatenate([np.full(len(objects[i]['r']),1/len(objects[i]['r'])) for i in indices])
    w*=len(w)/w.sum()
    return X,y,w

RNG=np.random.default_rng(5932)
OMEGA=RNG.normal(size=(32,256)); PHASE=RNG.uniform(0,2*np.pi,size=256)

def train_predict(objects, train, test, group, algorithm, parameter):
    X,y,w=combine(objects,train,group)
    T=np.concatenate([objects[i]['X'][group] for i in test])
    if algorithm=='trees':
        leaves,minimum=parameter
        model=HistGradientBoostingRegressor(max_iter=120,learning_rate=.05,
            max_leaf_nodes=leaves,min_samples_leaf=minimum,l2_regularization=1.,
            early_stopping=False,random_state=5933)
        model.fit(X,y,sample_weight=w); pred=model.predict(T)
    else:
        mean=np.average(X,axis=0,weights=w)
        scale=np.sqrt(np.average((X-mean)**2,axis=0,weights=w)); scale[scale<1e-10]=1
        X=(X-mean)/scale; T=(T-mean)/scale
        if algorithm=='ridge':
            X=np.column_stack([np.ones(len(X)),X]); T=np.column_stack([np.ones(len(T)),T])
            reg=np.eye(X.shape[1])*parameter*len(train); reg[0,0]=0
            pred=T@np.linalg.solve(X.T@(w[:,None]*X)+reg,X.T@(w*y))
        else:
            length,penalty=parameter
            gx=cp.asarray(X); gt=cp.asarray(T)
            om=cp.asarray(OMEGA[:X.shape[1]])/(length*np.sqrt(X.shape[1]))
            phase=cp.asarray(PHASE)
            zx=cp.cos(gx@om+phase)*np.sqrt(2/256)
            zt=cp.cos(gt@om+phase)*np.sqrt(2/256)
            zx=cp.column_stack([cp.ones(len(X)),zx]); zt=cp.column_stack([cp.ones(len(T)),zt])
            gw=cp.asarray(w); gy=cp.asarray(y)
            reg=cp.eye(257)*penalty*len(train); reg[0,0]=0
            with cupyx.errstate(linalg='raise'):
                coef=cp.linalg.solve(zx.T@(gw[:,None]*zx)+reg,zx.T@(gw*gy))
            pred=cp.asnumpy(zt@coef)
    assert np.isfinite(pred).all()
    cuts=np.cumsum([len(objects[i]['r']) for i in test])[:-1]
    return list(np.split(pred,cuts))

def score(objects,pred):
    before=np.array([np.mean(o['target']**2) for o in objects])
    after=np.array([np.mean((o['target']-p)**2) for o,p in zip(objects,pred)])
    velocities=[o['vrar']*(1+p) for o,p in zip(objects,pred)]
    chi0=sum(np.sum(((o['v']-o['vrar'])/o['error'])**2) for o in objects)
    chi=sum(np.sum(((o['v']-v)/o['error'])**2) for o,v in zip(objects,velocities))
    km0=np.mean([np.mean((o['v']-o['vrar'])**2) for o in objects])
    km=np.mean([np.mean((o['v']-v)**2) for o,v in zip(objects,velocities)])
    return dict(fractional_mse_gain=float(1-after.mean()/before.mean()),
        kms_mse_gain=float(1-km/km0),chi_square_gain=float(1-chi/chi0),
        galaxies_improved=int(sum(after<before)),nonpositive_predictions=int(sum(np.sum(v<=0) for v in velocities)),
        baseline_chi_square=float(chi0),chi_square=float(chi),
        before=before.tolist(),after=after.tolist())

def campaign(dest):
    rawpath=ROOT/'configs/sparc_rotation_curves_full_v1.json'
    photopath=ROOT/'configs/sparc_surface_brightness_exploration_v1.json'
    photo={g['galaxy']:g for g in json.loads(photopath.read_text())['galaxies']}
    names=sorted(photo); assert len(names)==139
    raw={g['name']:g for g in json.loads(rawpath.read_text())['galaxies'] if g['name'] in photo}
    parameters={'ridge':[.01,.1,1.], 'trees':[(3,40),(7,80)],
                'gpu_rbf_features':[(1.,.01),(1.,.1),(3.,.01),(3.,.1)]}
    registration=dict(scope='Already exposed development galaxies only. No independent confirmation.',
        names=names,input_hashes={str(p.relative_to(ROOT)):sha(p) for p in (rawpath,photopath)},
        scenarios={'nominal':[.5,.7],'lighter_stars':[.35,.5],'heavier_stars':[.65,.9]},
        salts=['observable-round1-A','observable-round1-B'],outer_folds=5,inner_folds=3,
        algorithms=parameters,groups=['local_coverage','photometry_multiscale','gas_force_proxy'],
        selection_metric='Equal-galaxy mean squared fractional velocity residual; no outer early stopping.',
        features='Local RAR/source features and observational radial coverage; add annular-area-weighted interior/exterior photometric contrasts at 0.5,2,8 kpc, or derived gas-force contrasts. Gas force is not observed gas density.',
        gpu_model='256 fixed random Fourier features approximating an RBF kernel; regularized regression, not a calibrated posterior or exact GP.',
        limitations=['SB values include publication processing and zero values of uncertain censoring status.',
        'Physical-radius filters inherit distance assumptions; incomplete source outside tabulated span.',
        'No full noise, inclination, distance, thickness or molecular conversion marginalization.',
        'Stellar mass scenarios are sensitivity brackets, not measured uncertainty intervals.',
        'Map pilot is separate; one map galaxy cannot furnish whole-galaxy validation.'])
    save(dest/'registration.json',registration)
    nominal=build(raw,photo,names,[.5,.7])
    # Check features cannot read observed velocity/error values.
    import copy
    changed=copy.deepcopy(raw)
    for g in changed.values():
        for row in g['rows']:
            row[1]=str(float(row[1])+123); row[2]=str(float(row[2])+17)
    poisoned=build(changed,photo,names,[.5,.7])
    assert all(np.array_equal(a['X'][key],b['X'][key]) for a,b in zip(nominal,poisoned) for key in a['X'])
    assert abs(score(nominal,[np.zeros(len(o['r'])) for o in nominal])['baseline_chi_square']-130714.6893155)<.01
    # Independent CPU/GPU implementation comparison for kernel feature regression.
    tr=list(range(10)); te=[10,11]; X,y,w=combine(nominal,tr,'photometry_multiscale')
    T=np.concatenate([nominal[i]['X']['photometry_multiscale'] for i in te])
    mean=np.average(X,axis=0,weights=w); sd=np.sqrt(np.average((X-mean)**2,axis=0,weights=w)); sd[sd<1e-10]=1
    om=OMEGA[:X.shape[1]]/np.sqrt(X.shape[1])
    zx=np.column_stack([np.ones(len(X)),np.cos(((X-mean)/sd)@om+PHASE)*np.sqrt(2/256)])
    zt=np.column_stack([np.ones(len(T)),np.cos(((T-mean)/sd)@om+PHASE)*np.sqrt(2/256)])
    reg=np.eye(257)*.1*len(tr);reg[0,0]=0
    cpu=zt@np.linalg.solve(zx.T@(w[:,None]*zx)+reg,zx.T@(w*y))
    gpu=np.concatenate(train_predict(nominal,tr,te,'photometry_multiscale','gpu_rbf_features',(1.,.1)))
    error=float(np.max(abs(cpu-gpu))); assert error<1e-9
    save(dest/'controls.json',dict(target_poison_features_unchanged=True,gpu_cpu_prediction_max_abs=error,
         nominal_baseline_reproduced=True))
    results=[]
    for scenario,ml in registration['scenarios'].items():
        objects=build(raw,photo,names,ml)
        for salt in registration['salts']:
            outer=foldmap(names,range(139),salt,5)
            for algorithm,choices in parameters.items():
                for group in registration['groups']:
                    predictions=[None]*139; folds=[]
                    for fold in range(5):
                        train=[i for i in range(139) if outer[i]!=fold]; test=[i for i in range(139) if outer[i]==fold]
                        inner=foldmap(names,train,salt+f'-inner-{fold}',3)
                        losses=[]
                        for parameter in choices:
                            loss=[]
                            for infold in range(3):
                                tr=[i for i in train if inner[i]!=infold];va=[i for i in train if inner[i]==infold]
                                pp=train_predict(objects,tr,va,group,algorithm,parameter)
                                loss.extend(np.mean((objects[i]['target']-p)**2) for i,p in zip(va,pp))
                            losses.append(float(np.mean(loss)))
                        best=choices[int(np.argmin(losses))]
                        pp=train_predict(objects,train,test,group,algorithm,best)
                        for i,p in zip(test,pp):predictions[i]=p
                        folds.append(dict(fold=fold,selected=best,inner_losses=losses,test_names=[names[i] for i in test]))
                    result=dict(scenario=scenario,salt=salt[-1],algorithm=algorithm,group=group,
                        folds=folds,**score(objects,predictions))
                    results.append(result)
                    save(dest/'results_partial.json',results)
                    if scenario=='nominal':
                        save(dest/f'predictions_{salt[-1]}_{algorithm}_{group}.json',[
                            dict(name=o['name'],r=o['r'].tolist(),observed=o['v'].tolist(),error=o['error'].tolist(),
                                rar=o['vrar'].tolist(),prediction=(o['vrar']*(1+p)).tolist()) for o,p in zip(objects,predictions)])
                    print(f'{scenario} {salt[-1]} {algorithm} {group}: fractional {result["fractional_mse_gain"]:.4f}, km {result["kms_mse_gain"]:.4f}, chi {result["chi_square_gain"]:.4f}',flush=True)
    save(dest/'result.json',dict(registration=registration,models=results,new_gravity_laws_admitted=0))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['maps','predict'],required=True)
    ap.add_argument('--output',required=True);args=ap.parse_args()
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    (dest/'runner.py').write_bytes(Path(__file__).read_bytes())
    start=time.perf_counter()
    save(dest/'runtime.json',dict(cupy=cp.__version__,device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cuda_driver=cp.cuda.runtime.driverGetVersion(),
        free_total_bytes=list(cp.cuda.runtime.memGetInfo())))
    if args.stage=='maps': map_audit(dest)
    else: campaign(dest)
    save(dest/'completion.json',dict(status='COMPLETED',seconds=time.perf_counter()-start))

if __name__=='__main__':main()
