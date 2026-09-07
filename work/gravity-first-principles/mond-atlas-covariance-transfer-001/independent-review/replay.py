"""Independent frozen-record and covariance-score replay; no optimizer rerun."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import hashlib,json,sys
import numpy as np
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R/'scripts'))
from mond_atlas_actual_spectra import ActualSpectrumModel
from mond_atlas_aperture_injection_gpu import GPUCallback
P=Path(__file__).resolve().parent.parent;O=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    summary=read(P/'run001/summary.json');bindings=read(P/'run001/bindings.json')
    assert all(sha(R/k)==v for k,v in bindings.items())
    original=R/'work/gravity-first-principles/mond-atlas-aperture-injection-001';base=read(original/'fit002/summary.json');train=base['training_indices'];held=base['evaluation_indices'];assert not set(train)&set(held)
    nr=read(original/'noise001/receipt.json')
    with np.load(R/nr['path']) as z:noise={k:z[k].copy() for k in z.files}
    variants={v['omitted_western_index']:v for v in read(R/'work/gravity-first-principles/mond-atlas-aperture-covariance-001/run001/western-leave-one-core-out.json')};assert len(variants)==29
    callbacks={b:GPUCallback(ActualSpectrumModel(R/'work/private/mond-atlas-actual-spectra-001/cache001/p0p03125_z48.npz',branch=b)) for b in ['boxcar_independent','boxcar_hanning_full','boxcar_hanning_decimated']}
    signals={b:cb.predict([0,10,1]) for b,cb in callbacks.items()};pending={};evaluated={};errors=[];startcount=0
    def key(r):return (r['branch'],r['family'],r['trial'],r['omitted_western_index'])
    for line in (P/'run001/fits.jsonl').read_text().splitlines():
        row=json.loads(line);k=key(row);fit=row['fit'];assert fit['training_indices']==train and len(fit['starts'])==3;startcount+=3 if row['phase']=='evaluation' else 0
        if row['phase']=='freeze':
            assert k not in pending and k not in evaluated
            pred=np.array(row['frozen_predictions'],dtype=np.float64);assert pred.shape==(5,42) and np.isfinite(pred).all()
            assert hashlib.sha256(pred.tobytes()).hexdigest()==row['frozen_prediction_sha256'];pending[k]=row
        else:
            assert k not in evaluated
            if fit['evaluation_allowed']:
                assert k in pending;frozen=pending.pop(k);assert frozen['fit']==fit
                pred=np.array(frozen['frozen_predictions']);v=variants[k[-1]];C=np.array(v['covariance']);mean=np.array(v['mean']);res=signals[k[0]][held]+noise[k[1]][k[2]][held]-mean-pred
                q=float(np.einsum('ij,ji->i',res,np.linalg.solve(C,res.T)).mean()/42);errors.append(abs(q-row['held_q']))
                old=read(original/f'fit002/{k[0]}_{k[1]}_{k[2]}.json');errors.append(abs(row['held_q_shift']-(q-old['evaluation']['mean_q_per_channel'])))
                assert np.allclose(np.array(fit['parameters'])-old['fit']['parameters'],row['parameter_shift'],rtol=0,atol=1e-14)
                assert fit['training_mean_q_per_channel']==min(s['training_mean_q_per_channel'] for s in fit['starts'] if s['success'])
                assert fit['scaled_singular_ratio']>1e-6
                assert np.array_equal(np.array(frozen['unknown_envelope']),callbacks[k[0]].unknown_envelope(fit['parameters'])[held])
            evaluated[k]=row
    assert not pending and len(evaluated)==1392 and startcount==4176
    for s in summary['summaries']:
        rows=[r for r in evaluated.values() if r['family']==s['family']];ok=[r for r in rows if 'parameter_shift' in r];shifts=np.abs([r['parameter_shift'] for r in ok]);q=np.abs([r['held_q_shift'] for r in ok])
        assert len(rows)==s['fits'] and len(ok)==s['identifiable']
        assert np.allclose(np.median(shifts,axis=0),s['median_absolute_parameter_shift'],rtol=0,atol=1e-14) and np.allclose(shifts.max(axis=0),s['max_absolute_parameter_shift'],rtol=0,atol=1e-14)
        assert float(np.median(q))==s['median_absolute_held_q_shift'] and float(q.max())==s['max_absolute_held_q_shift']
    assert max(errors)<1e-10
    result=dict(status='PASS_INDEPENDENT_COVARIANCE_TRANSFER_REPLAY',fits=len(evaluated),start_records_checked=startcount,max_score_error=max(errors),freeze_before_evaluation_in_record_order=True,source_region_spectra_read=0,scope='Reuses independently benchmarked source callback for three truth signals; independently solves every held covariance quadratic. No optimizer rerun.',bindings={str(p.relative_to(R)):sha(p) for p in [Path(__file__),P/'run001/summary.json',P/'run001/bindings.json',P/'run001/fits.jsonl']})
    with (O/'receipt.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='bindings'}))
if __name__=='__main__':main()
