"""Refit frozen injection trials across all preexisting western LOO covariances."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,time
from pathlib import Path
import numpy as np
from mond_atlas_actual_spectra import ActualSpectrumModel
from mond_atlas_aperture_injection_gpu import GPUCallback
from mond_atlas_aperture_scorer import training_fit,freeze_predictions,evaluate
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-covariance-transfer-001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def run():
 start=time.monotonic();out=P/'run001';out.mkdir(exist_ok=False)
 original=R/'work/gravity-first-principles/mond-atlas-aperture-injection-001/fit002';baseline=read(original/'summary.json')
 train=baseline['training_indices'];held=baseline['evaluation_indices'];assert len(train)==10 and len(held)==5 and not set(train)&set(held)
 for path,h in read(original/'bindings.json').items():assert sha(R/path)==h
 nreceipt=read(original.parent/'noise001/receipt.json');packet=R/nreceipt['path'];assert sha(packet)==nreceipt['sha256']
 covpath=R/'work/gravity-first-principles/mond-atlas-aperture-covariance-001/run001/western-leave-one-core-out.json';variants=read(covpath);assert len(variants)==29
 cache=R/'work/private/mond-atlas-actual-spectra-001/cache001/p0p03125_z48.npz'
 paths=[Path(__file__),P/'PREFLIGHT.md',covpath,cache,packet,original/'bindings.json',original/'summary.json',R/'scripts/mond_atlas_aperture_scorer.py',R/'scripts/mond_atlas_actual_spectra.py',R/'scripts/mond_atlas_aperture_injection_gpu.py']+list(original.glob('boxcar*.json'))
 save(out/'bindings.json',{p.relative_to(R).as_posix():sha(p) for p in paths})
 with np.load(packet) as z:noise={k:z[k].copy() for k in z.files}
 records=[];errors=[]
 with (out/'fits.jsonl').open('w') as stream:
  for branch in ('boxcar_independent','boxcar_hanning_full','boxcar_hanning_decimated'):
   model=ActualSpectrumModel(cache,branch=branch);callback=GPUCallback(model);signal=callback.predict([0,10,1])
   for family in ('gaussian_background','real_background'):
    for trial,bg in enumerate(noise[family]):
     old=read(original/f'{branch}_{family}_{trial}.json')
     for variant in variants:
      mean=np.array(variant['mean']);C=np.array(variant['covariance']);fit=training_fit(callback,train,signal[train]+bg[train],mean,C)
      row=dict(branch=branch,family=family,trial=trial,omitted_western_index=variant['omitted_western_index'],fit=fit)
      if fit['evaluation_allowed']:
       frozen=freeze_predictions(callback,fit,held);row['frozen_prediction_sha256']=hashlib.sha256(frozen.values.tobytes()).hexdigest()
       row['frozen_predictions']=frozen.values.tolist();row['unknown_envelope']=callback.unknown_envelope(fit['parameters'])[held].tolist()
       # Persist numerical predictions before held residual calculation.
       stream.write(json.dumps(dict(phase='freeze',**row),allow_nan=False)+'\n');stream.flush()
       score=evaluate(frozen,signal[held]+bg[held],mean,C);res=np.array(score['residuals']);direct=np.einsum('ij,ji->i',res,np.linalg.solve(C,res.T))/42
       error=float(np.max(abs(direct-score['per_aperture_q_per_channel'])));errors.append(error);assert error<1e-10
       row={k:v for k,v in row.items() if k not in ('frozen_predictions','unknown_envelope')}
       row['parameter_shift']=(np.array(fit['parameters'])-old['fit']['parameters']).tolist()
       row['held_q_shift']=score['mean_q_per_channel']-old['evaluation']['mean_q_per_channel'];row['held_q']=score['mean_q_per_channel']
      stream.write(json.dumps(dict(phase='evaluation',**row),allow_nan=False)+'\n');stream.flush();records.append(row)
     print(branch,family,trial,'completed',flush=True)
 summaries=[]
 for family in ('gaussian_background','real_background'):
  rows=[r for r in records if r['family']==family];ok=[r for r in rows if 'parameter_shift'in r];shifts=np.abs([r['parameter_shift'] for r in ok]);q=np.abs([r['held_q_shift'] for r in ok])
  summaries.append(dict(family=family,fits=len(rows),identifiable=len(ok),median_absolute_parameter_shift=np.median(shifts,axis=0).tolist(),max_absolute_parameter_shift=shifts.max(axis=0).tolist(),median_absolute_held_q_shift=float(np.median(q)),max_absolute_held_q_shift=float(q.max())))
 save(out/'summary.json',dict(status='WESTERN_COVARIANCE_TRANSFER_EXECUTED',fits=len(records),variants=29,source_region_reads=0,gravity_law_comparisons=0,summaries=summaries,independent_quadratic_max_error=max(errors),seconds=time.monotonic()-start))
 print(json.dumps(read(out/'summary.json')))
if __name__=='__main__':run()
