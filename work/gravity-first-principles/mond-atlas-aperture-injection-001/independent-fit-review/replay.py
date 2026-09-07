"""Independent saved-score/provenance replay; no optimization or observed source spectra."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib,csv
import numpy as np
R=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(R/'scripts'))
from mond_atlas_actual_spectra import ActualSpectrumModel
from mond_atlas_aperture_injection_gpu import GPUCallback
P=Path(__file__).resolve().parent.parent
O=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
checked={}
def bind(p): checked[p.relative_to(R).as_posix()]=sha(p)
def check_bindings(obj):
    for path,digest in obj.items():
        p=R/path
        assert sha(p)==digest,(path,'hash mismatch')
        bind(p)
for sub in ('fit002','noise001','gpu001'):
    p=P/sub/'bindings.json'; bind(p); check_bindings(json.loads(p.read_text()))
review=R/'work/gravity-first-principles/mond-atlas-actual-spectra-001/independent-review/receipt.json'
check_bindings(json.loads(review.read_text())['bindings'])
receipt=json.loads((P/'noise001/receipt.json').read_text()); npath=R/receipt['path']
assert sha(npath)==receipt['sha256']; bind(npath)
with np.load(npath) as z: noise={k:z[k].copy() for k in z.files}
aps=R/'work/gravity-first-principles/mond-atlas-observation-admission-001/run004/frozen-geometric-apertures.csv'
with aps.open(newline='') as f: aps=list(csv.DictReader(f))
train=[i for i,r in enumerate(aps) if r['subset']=='training']; held=[i for i,r in enumerate(aps) if r['subset']=='evaluation']
assert len(train)==10 and len(held)==5 and not set(train)&set(held)
C=noise['covariance']; mean=noise['western_mean']; truth=np.array([0.,10.,1.])
def q(res): return np.einsum('ij,ji->i',res,np.linalg.solve(C,res.T))/42
errors=[]; count=0; starts=0; records=[]
cache=R/'work/private/mond-atlas-actual-spectra-001/cache001/p0p03125_z48.npz'
for branch in ('boxcar_independent','boxcar_hanning_full','boxcar_hanning_decimated'):
    cb=GPUCallback(ActualSpectrumModel(cache,branch=branch)); signal=cb.predict(truth)
    for p in sorted((P/'fit002').glob(branch+'_*.json')):
        bind(p); row=json.loads(p.read_text()); fit=row['fit']; par=np.array(fit['parameters'])
        assert fit['training_indices']==train and fit['evaluation_allowed']
        assert np.all(par>=[-30,3,.5]) and np.all(par<=[30,20,2])
        singular=np.array(fit['scaled_singular_values'])
        assert np.isclose(singular[-1]/singular[0],fit['scaled_singular_ratio']) and fit['scaled_singular_ratio']>1e-6
        assert len(fit['starts'])==3
        best=min(s['training_mean_q_per_channel'] for s in fit['starts'] if s['success'])
        assert best==fit['training_mean_q_per_channel']
        bg=np.broadcast_to(mean,(15,42)) if row['family']=='noiseless' else noise[row['family']][row['trial']]
        data=signal+bg
        for s in fit['starts']:
            val=float(q(cb.predict(s['parameters'])[train]+mean-data[train]).mean())
            errors.append(abs(val-s['training_mean_q_per_channel'])); starts+=1
        saved=R/row['frozen_prediction']['path']; assert sha(saved)==row['frozen_prediction']['sha256']; bind(saved)
        with np.load(saved) as z:
            pred=z['predictions'].copy(); envelope=z['unknown_envelope'].copy()
            assert z['indices'].tolist()==held and np.array_equal(z['parameters'],par)
        assert row['evaluation']['indices']==held
        errors.append(float(np.max(np.abs(pred-cb.predict(par)[held]))))
        res=data[held]-mean-pred; scores=q(res)
        errors.extend([float(np.max(np.abs(scores-row['evaluation']['per_aperture_q_per_channel']))),abs(float(scores.mean())-row['evaluation']['mean_q_per_channel']),abs(float(np.sqrt(np.mean(res**2)))-row['evaluation']['raw_rmse_mJy_beam'])])
        assert np.allclose(par-truth,row['parameter_error'],rtol=0,atol=1e-15)
        l2=float(np.linalg.norm(pred-signal[held])/np.linalg.norm(signal[held]))
        errors.append(abs(l2-row['held_signal_relative_L2']))
        assert np.array_equal(envelope,cb.unknown_envelope(par)[held])
        assert envelope.max()==row['max_unknown_envelope_mjy_beam']
        if row['family']=='noiseless': assert row['recovery_pass'] and np.max(np.abs(par-truth))<1e-5 and l2<1e-6
        records.append(row);count+=1
summary=json.loads((P/'fit002/summary.json').read_text()); bind(P/'fit002/summary.json')
for s in summary['summaries']:
    rows=[r for r in records if r['family']==s['family']]
    assert len(rows)==s['trials']==s['identifiable']
    assert np.allclose(np.median([np.abs(r['parameter_error']) for r in rows],axis=0),s['median_absolute_parameter_error'],rtol=0,atol=1e-15)
    assert np.median([r['evaluation']['mean_q_per_channel'] for r in rows])==s['median_held_q_per_channel']
    assert max(r['evaluation']['mean_q_per_channel'] for r in rows)==s['max_held_q_per_channel']
assert count==51 and starts==153 and max(errors)<1e-10
bind(Path(__file__))
result=dict(status='PASS_INDEPENDENT_SCORE_AND_PROVENANCE_REPLAY',trials=count,starts_replayed=starts,max_absolute_arithmetic_error=max(errors),nested_actual_review_bindings_verified=True,training_indices=train,evaluation_indices=held,source_region_spectra_read=0,limitations=['GPU callback reused for source predictions; independent dense covariance solve for all scores, previous independent CPU signal audit retained.','Runtime freeze ordering verified in source code; immutable timestamp ordering not independently established.','Runner does not recursively verify nested actual-review bindings; independently verified here after execution.','Valid-emitter conditional template only; globally steady source fails; shared background trials are not independent.','One optimizer start equals injected truth; noiseless recovery alone is not a general optimizer robustness claim.'],bindings=checked)
(O/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='bindings'}))
