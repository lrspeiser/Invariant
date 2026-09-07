"""Frozen source-continuum x channel-threshold factorial; no gravity scoring."""
from pathlib import Path
import csv,itertools,json,shutil
import numpy as np
from astropy.io import fits
from scipy.signal import convolve2d
from threadpoolctl import threadpool_limits
from mond_atlas_native_selection import *
from mond_atlas_native_spectral import continuum_operator
from mond_atlas_selection_transfer import intrinsic,new_controls
from run_mond_atlas_native_selection import sha,write_json,write_csv,controls

ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT/'work/gravity-first-principles/mond-atlas-selection-ablation-001'
OLD=ROOT/'work/gravity-first-principles/mond-atlas-native-selection-001/run-001'
TRANSFER=ROOT/'work/gravity-first-principles/mond-atlas-selection-transfer-001/run001'

def ablation_controls(op,indices,kernel):
    rng=np.random.default_rng(9062701)
    n=op.shape[1]; s=np.eye(n)[indices]
    x=rng.normal(size=(n,7,9))
    selection_error=float(np.max(abs((s@x.reshape(n,-1)).reshape(len(indices),7,9)-x[indices])))
    spectral={}
    for branch in ['boxcar_independent','boxcar_hanning_full','boxcar_hanning_decimated']:
        h,_,_=spectral_matrix(n,branch); raw=rng.normal(size=h.shape[1])
        ref=raw if branch=='boxcar_independent' else np.convolve(raw,[.25,.5,.25],mode='valid')
        if branch=='boxcar_hanning_decimated': ref=ref[::2]
        spectral[branch]=float(np.max(abs(h@raw-ref)))
    plane=rng.uniform(size=(7,9))
    full=convolve2d(plane,kernel,mode='full')
    conservation=abs(float(full.sum()/plane.sum())-1)
    first=convolve_spatial((op@x.reshape(n,-1)).reshape(len(indices),7,9),kernel)
    second=(op@convolve_spatial(x,kernel).reshape(n,-1)).reshape(first.shape)
    commute=float(np.max(abs(first-second)))
    passed=selection_error<1e-12 and max(spectral.values())<1e-12 and conservation<1e-12 and commute<1e-10
    return dict(selection_error=selection_error,direct_spectral_errors=spectral,
                full_spatial_flux_relative_error=conservation,commutation_max_abs=commute,passed=passed)

def run():
    out=PKG/'run001'; out.mkdir(exist_ok=False)
    cfgpath=ROOT/'configs/mond_atlas_native_selection_v1.json'
    config=json.loads(cfgpath.read_text())
    cachepath=ROOT/'work/private/mond-atlas-native-selection-001/run-001/selection-and-support.npz'
    paths=[Path(__file__),PKG/'PREFLIGHT.md',cfgpath,cachepath,OLD/'support.json',TRANSFER/'trials.csv',
           TRANSFER/'pre-access-bindings.json',ROOT/config['native_history'],ROOT/config['cube_path']]
    paths += [ROOT/'scripts'/name for name in ['mond_atlas_native_selection.py','mond_atlas_native_spectral.py',
                  'mond_atlas_selection_transfer.py','run_mond_atlas_native_selection.py','mond_atlas_common.py']]
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    assert bindings[config['cube_path']]==config['cube_sha256']
    assert shutil.disk_usage(ROOT).free>8e9
    write_json(out/'pre-access-bindings.json',dict(bindings=bindings,new_arrays_opened=False,
              previous_development_exposure=True,posthoc_pipeline_ablation=True))
    hist=json.loads((ROOT/config['native_history']).read_text()); prov=hist['provenance']
    indices=prov['parent_channel_indices_zero_based']
    op=continuum_operator(prov['parent_channel_count'],prov['continuum_fit_parent_indices_zero_based'],indices,prov['polynomial_order'])
    select=np.eye(op.shape[1])[indices]
    header=fits.getheader(ROOT/config['cube_path']); beam=beam_from_history(header)
    ncov=beam_covariance(beam['major_arcsec'],beam['minor_arcsec'],beam['pa_deg'],header['CDELT2']*3600,header['CDELT1']*3600)
    ecov=np.eye(2)*(30/FWHM_SIGMA/abs(header['CDELT1']*3600))**2-ncov
    nk=gaussian_kernel(ncov); ek=gaussian_kernel(ecov)
    ctl=controls(config,op,prov,ncov,ecov); ctl['templates']=new_controls(); ctl['ablation']=ablation_controls(op,indices,nk)
    write_json(out/'controls.json',ctl); assert ctl['templates']['passed'] and ctl['ablation']['passed']
    # Array access begins only after frozen implementation controls passed.
    cache=np.load(cachepath); sigma=cache['sigma_jy_per_native_beam']; scale=float(np.median(sigma))
    positions=json.loads((OLD/'support.json').read_text())['selected_positions_yx_zero_based']
    patches=[]
    with fits.open(ROOT/config['cube_path'],memmap=True) as f:
        for y,x in positions:
            radius=40+len(ek)//2
            extended=f[0].data.squeeze()[:,y-radius:y+radius+1,x-radius:x+radius+1].astype(float)
            cut=np.s_[:,radius-40:radius+41,radius-40:radius+41]
            patches.append((extended[cut]-cache['native_median'][:,None,None],
                  convolve_spatial(extended,ek)[cut]-cache['median'][:,None,None]))
    fluxfactor=1.5**2/(np.pi*beam['major_arcsec']*beam['minor_arcsec']/(4*np.log(2)))*abs(header['CDELT3'])/1000
    rows=[]; ledger=[]; profiles=[]; matrix={'A':op.tolist(),'S':select.tolist(),'branches':{}}
    zero=np.zeros((42,81,81))
    for branch in config['spectral_branches']:
        h,grid,width=spectral_matrix(op.shape[1],branch)
        matrix['branches'][branch]=dict(H=h.tolist(),precell_centers=grid.tolist(),precell_width=width,
                      actual_AHHtAt=(op@h@h.T@op.T).tolist(),source_disabled_SHHtSt=(select@h@h.T@select.T).tolist(),
                      note='S covariance is diagnostic only; observed background is unchanged in all trials.')
        for center in [10,20,30]:
            norm=None
            for kind in ['rotation','warp','streaming']:
                raw,_=intrinsic(kind,grid,width,indices[center])
                parent=(h@raw.reshape(len(grid),-1)).reshape(op.shape[1],81,81)
                # Match the existing replay's ordering, and audit alternate stage order.
                native_parent=convolve_spatial(parent,nk); positive=native_parent[indices]
                ds_pre=convolve_spatial(positive,ek)
                if norm is None: norm=float(ds_pre.max())
                for continuum,operator in [('actual',op),('source_disabled',select)]:
                    post=(operator@parent.reshape(len(parent),-1)).reshape(42,81,81)
                    native=(operator@native_parent.reshape(len(parent),-1)).reshape(42,81,81)
                    detector=convolve_spatial(native,ek)
                    native_reference=convolve_spatial(post,nk)
                    commutation=float(np.max(abs(native-native_reference)))
                    assert commutation<1e-10
                    sums=dict(intrinsic_pre_H_weighted=float(raw.sum()*width),parent_post_H=float(parent.sum()),
                        stored_pre_continuum=float(parent[indices].sum()),stored_post_continuum=float(post.sum()),
                        native_post_continuum=float(native.sum()),detector_post_continuum=float(detector.sum()))
                    ledger.append(dict(branch=branch,center=center,kind=kind,continuum=continuum,**sums,
                        continuum_signed_fraction=float(post.sum()/parent[indices].sum()),
                        native_crop_ratio=float(native.sum()/post.sum()),detector_crop_ratio=float(detector.sum()/native.sum()),
                        native_ordering_max_abs=commutation,normalization_symmetric_detector_peak=norm))
                    for c in range(42):
                        profiles.append(dict(branch=branch,center=center,kind=kind,continuum=continuum,stored_channel=c,
                            pre_continuum=float(parent[indices[c]].sum()),post_continuum=float(post[c].sum()),
                            native=float(native[c].sum()),detector=float(detector[c].sum()),
                            detector_peak=float(detector[c].max()/norm),western_mad=float(sigma[c]),global_mad=scale))
                    ns,ds,ps=native/norm,detector/norm,positive/norm
                    for threshold,local_sigma in [('per_channel',sigma),('fixed_global',np.full_like(sigma,scale))]:
                        for group,backgrounds in [('noiseless',[(zero,zero)]),('empirical',patches)]:
                            for draw,(n,d) in enumerate(backgrounds):
                                baseline=select_runs(d,local_sigma)
                                for amplitude in [5,10]:
                                    result=recovery(n,d,ns,ds,ps,local_sigma,amplitude*scale,fluxfactor,baseline)
                                    rows.append(dict(group=group,branch=branch,center=center,kind=kind,continuum=continuum,
                                        threshold=threshold,draw=draw,amplitude=amplitude,**result))
        print(branch,'finished',flush=True)
    write_json(out/'linear-operators.json',matrix)
    write_csv(out/'stage-ledger.csv',ledger); write_csv(out/'channel-ledger.csv',profiles); write_csv(out/'trials.csv',rows)
    previous=list(csv.DictReader((TRANSFER/'trials.csv').open()))
    metric='true_flux_fraction_retained'; compared=0; maximum=0.
    keynames=['group','branch','center','kind','draw','amplitude']
    lookup={tuple(str(r[k]) for k in keynames):r for r in rows if r['continuum']=='actual' and r['threshold']=='per_channel'}
    for r in previous:
        if r['group'] not in ['empirical','noiseless']: continue
        current=lookup[tuple(r[k] for k in keynames)]
        for m in [metric,'paired_selected_flux_difference_over_reference','selected_noisy_flux_over_reference']:
            maximum=max(maximum,abs(float(current[m])-float(r[m])))
        compared+=1
    write_json(out/'prior-replay.json',dict(trials_compared=compared,max_absolute_fraction_error=maximum,passed=maximum<1e-10))
    assert maximum<1e-10
    aggregate=[]; effects=[]; gaps=[]
    for group,branch,kind,amplitude in itertools.product(['empirical','noiseless'],config['spectral_branches'],['rotation','warp','streaming'],[5,10]):
        data=[r for r in rows if (r['group'],r['branch'],r['kind'],r['amplitude'])==(group,branch,kind,amplitude)]
        common=dict(group=group,branch=branch,kind=kind,amplitude=amplitude)
        cells={}
        for center,continuum,threshold in itertools.product([10,20,30],['actual','source_disabled'],['per_channel','fixed_global']):
            rr=sorted([r for r in data if (r['center'],r['continuum'],r['threshold'])==(center,continuum,threshold)],key=lambda r:r['draw'])
            v=np.array([r[metric] for r in rr]); cells[(center,continuum,threshold)]=v
            aggregate.append(dict(**common,center=center,continuum=continuum,threshold=threshold,
                 mean_retention=float(v.mean()),sd=float(v.std(ddof=1)) if len(v)>1 else 0.,minimum=float(v.min()),maximum=float(v.max()),
                 mean_paired_flux=float(np.mean([r['paired_selected_flux_difference_over_reference'] for r in rr]))))
        for center in [10,20,30]:
            a=cells[(center,'actual','per_channel')]; b=cells[(center,'source_disabled','per_channel')]
            c=cells[(center,'actual','fixed_global')]; d=cells[(center,'source_disabled','fixed_global')]
            for label,v in [('remove_source_continuum',b-a),('fixed_threshold',c-a),('interaction',d-c-b+a)]:
                effects.append(dict(**common,center=center,effect=label,mean=float(v.mean()),sd=float(v.std(ddof=1)) if len(v)>1 else 0.,
                      minimum=float(v.min()),maximum=float(v.max()),material=abs(float(v.mean()))>.05))
        for continuum,threshold in itertools.product(['actual','source_disabled'],['per_channel','fixed_global']):
            v=cells[(20,continuum,threshold)]-.5*(cells[(10,continuum,threshold)]+cells[(30,continuum,threshold)])
            gaps.append(dict(**common,continuum=continuum,threshold=threshold,center20_minus_outermean=float(v.mean()),
                    sd=float(v.std(ddof=1)) if len(v)>1 else 0.,minimum=float(v.min()),maximum=float(v.max())))
    write_csv(out/'case-summary.csv',aggregate); write_csv(out/'paired-effects.csv',effects); write_csv(out/'placement-gaps.csv',gaps)
    assert all(sha(ROOT/k)==v for k,v in bindings.items())
    write_json(out/'summary.json',dict(status='CONDITIONAL_PIPELINE_ABLATION_EXECUTED',admission='SOURCE_BLOCKED',
        empirical_trials=sum(r['group']=='empirical' for r in rows),noiseless_trials=sum(r['group']=='noiseless' for r in rows),
        prior_trials_replayed=compared,prior_replay_max_error=maximum,controls_passed=True,
        background_unchanged_for_source_continuum_ablation=True,new_gravity_scores=0,new_private_bytes=0,all_inputs_reverified=True))

if __name__=='__main__':
    with threadpool_limits(limits=1): run()
