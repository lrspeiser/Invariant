"""Recover actual-source mock signals using frozen training/evaluation apertures."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import csv, hashlib, json, time
import numpy as np
from mond_atlas_actual_spectra import ActualSpectrumModel
from mond_atlas_aperture_injection_gpu import GPUCallback
from mond_atlas_aperture_scorer import training_fit, freeze_predictions, evaluate

R=Path(__file__).resolve().parents[1]
P=R/'work/gravity-first-principles/mond-atlas-aperture-injection-001'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v): p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def run():
    start=time.monotonic()
    review=R/'work/gravity-first-principles/mond-atlas-actual-spectra-001/independent-review/receipt.json'
    gate=json.loads(review.read_text())
    assert gate['primary_failed']==0 and gate['primary_rows']==1350
    assert json.loads((P/'gpu001/receipt.json').read_text())['all_pass']
    for name in ('noise001','gpu001'):
        for path,sha in json.loads((P/name/'bindings.json').read_text()).items(): assert digest(R/path)==sha
    receipt=json.loads((P/'noise001/receipt.json').read_text());packet=R/receipt['path']
    assert digest(packet)==receipt['sha256']
    cache=R/'work/private/mond-atlas-actual-spectra-001/cache001/p0p03125_z48.npz'
    aps=R/'work/gravity-first-principles/mond-atlas-observation-admission-001/run004/frozen-geometric-apertures.csv'
    with aps.open(newline='') as f: rows=list(csv.DictReader(f))
    train=[i for i,r in enumerate(rows) if r['subset']=='training'];held=[i for i,r in enumerate(rows) if r['subset']=='evaluation']
    assert len(train)==10 and len(held)==5 and not set(train)&set(held)
    out=P/'fit002';out.mkdir(exist_ok=False)
    private=R/'work/private/mond-atlas-aperture-injection-001/fit002';private.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),review,cache,aps,packet,P/'PREFLIGHT.md',P/'gpu001/receipt.json',R/'scripts/mond_atlas_aperture_scorer.py',R/'scripts/mond_atlas_aperture_injection_gpu.py',R/'scripts/mond_atlas_actual_spectra.py']
    save(out/'bindings.json',{p.relative_to(R).as_posix():digest(p) for p in paths})
    with np.load(packet) as z: noise={k:z[k].copy() for k in z.files}
    C=noise['covariance'];mean=noise['western_mean'];truth=np.array([0.,10.,1.]);records=[]
    for branch in ('boxcar_independent','boxcar_hanning_full','boxcar_hanning_decimated'):
        model=ActualSpectrumModel(cache,branch=branch);callback=GPUCallback(model);signal=callback.predict(truth)
        trials=[('noiseless',0,np.broadcast_to(mean,(15,42)))]+[(family,t,bg) for family in ('gaussian_background','real_background') for t,bg in enumerate(noise[family])]
        for family,t,background in trials:
            name=f'{branch}_{family}_{t}'
            fit=training_fit(callback,train,signal[train]+background[train],mean,C)
            row=dict(branch=branch,family=family,trial=t,fit=fit,global_steady_valid=model.global_steady_valid)
            if fit['evaluation_allowed']:
                frozen=freeze_predictions(callback,fit,held)
                envelope=callback.unknown_envelope(fit['parameters'])[held]
                dest=private/(name+'.npz')
                np.savez_compressed(dest,predictions=frozen.values,unknown_envelope=envelope,indices=held,parameters=fit['parameters'])
                row['frozen_prediction']=dict(path=dest.relative_to(R).as_posix(),sha256=digest(dest))
                score=evaluate(frozen,signal[held]+background[held],mean,C)
                row['evaluation']={k:v for k,v in score.items() if k!='residuals'}
                row['parameter_error']=(np.array(fit['parameters'])-truth).tolist()
                row['held_signal_relative_L2']=float(np.linalg.norm(frozen.values-signal[held])/np.linalg.norm(signal[held]))
                row['max_unknown_envelope_mjy_beam']=float(envelope.max())
                if family=='noiseless':
                    row['recovery_pass']=bool(np.max(np.abs(row['parameter_error']))<1e-5 and row['held_signal_relative_L2']<1e-6)
            else: row['recovery_pass']=False
            records.append(row);save(out/(name+'.json'),row)
            print(name,fit['status'],flush=True)
    summaries=[]
    for family in ('noiseless','gaussian_background','real_background'):
        selected=[r for r in records if r['family']==family];ok=[r for r in selected if 'evaluation' in r]
        summaries.append(dict(family=family,trials=len(selected),identifiable=len(ok),
            median_absolute_parameter_error=np.median([np.abs(r['parameter_error']) for r in ok],axis=0).tolist() if ok else None,
            median_held_q_per_channel=float(np.median([r['evaluation']['mean_q_per_channel'] for r in ok])) if ok else None,
            max_held_q_per_channel=max([r['evaluation']['mean_q_per_channel'] for r in ok],default=None)))
    result=dict(status='ACTUAL_SOURCE_TEMPLATE_INJECTION_FITS_EXECUTED',source_region_reads=0,observed_gravity_scores=0,
        trials=len(records),noiseless_all_pass=all(r.get('recovery_pass',False) for r in records if r['family']=='noiseless'),
        summaries=summaries,seconds=time.monotonic()-start,training_indices=train,evaluation_indices=held,
        interpretation='Conditional valid-emitter signals; real backgrounds shared and previously exposed; working covariance; no discovery significance or observed mechanism preference')
    save(out/'summary.json',result);print(json.dumps(result));assert result['noiseless_all_pass']

if __name__=='__main__': run()
