import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,csv,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'AGENTS.md').exists());RUN=P/'run002'

def main():
    cfg=json.loads((ROOT/'configs/mond_atlas_aperture_noise_v1.json').read_text(encoding='utf-8'));source=ROOT/cfg['input'];assert hashlib.sha256(source.read_bytes()).hexdigest()==cfg['input_sha256']
    with np.load(source) as z:west=z['training'];east=z['validation']
    frozen=json.loads((RUN/'models-before-east.json').read_text(encoding='utf-8'));mean=west.mean(axis=(0,1,2));e=(west-mean).reshape(29,576,42)
    sample=e.reshape(-1,42).T@e.reshape(-1,42)/(29*576);C=.7*sample+.3*np.diag(np.diag(sample));ci=np.linalg.inv(C)
    rawK=sum(core@ci@core.T for core in e)/(29*42);K=.9*rawK+.1*np.diag(np.diag(rawK));ki=np.linalg.inv(K)
    savedC=np.loadtxt(RUN/'C.csv',delimiter=',');savedK=np.loadtxt(RUN/'K-0.1.csv',delimiter=',')
    errors=dict(C=float(np.max(abs(savedC-C))),K=float(np.max(abs(savedK-K))))
    eastflat=(east-mean).reshape(27,576,42);q=np.array([np.sum(core*(ki@core@ci)) for core in eastflat])/(576*42)
    ld=42*np.linalg.slogdet(K)[1]+576*np.linalg.slogdet(C)[1];lp=-.5*(q*576*42+ld+576*42*np.log(2*np.pi))/(576*42)
    with (RUN/'core-scores.csv').open(encoding='utf-8',newline='') as f:rows=[r for r in csv.DictReader(f) if float(r['alpha'])==.1]
    errors['joint_q']=float(max(abs(q-np.array([float(r['q']) for r in rows]))));errors['joint_logpdf']=float(max(abs(lp-np.array([float(r['logpdf']) for r in rows]))))
    summary=json.loads((RUN/'summary.json').read_text(encoding='utf-8'));apertures=[]
    for side in [1,2,4,8,12,24]:
        variances=[];observed=[]
        for y in range(0,24,side):
            for x in range(0,24,side):
                indices=np.array([iy*24+ix for iy in range(y,y+side) for ix in range(x,x+side)])
                variances.append(K[np.ix_(indices,indices)].sum()/side**4)
                observed.append((east-mean)[:,y:y+side,x:x+side].mean(axis=(1,2)))
        obs=np.stack(observed,axis=1);var=np.array(variances)
        aq=float(np.mean(np.einsum('btc,cd,btd->bt',obs,ci,obs)/var[None,:]/42))
        tr=float(np.mean(np.sum(obs*obs,axis=2))/(np.trace(C)*var.mean()))
        ref=next(r for r in summary['apertures'] if r['selected'] and r['side']==side)
        apertures.append(dict(side=side,q=aq,trace_ratio=tr));errors['aperture_'+str(side)]=max(abs(aq-ref['q']),abs(tr-ref['trace_ratio']))
    identical={name:hashlib.sha256((P/'run001'/name).read_bytes()).hexdigest()==hashlib.sha256((RUN/name).read_bytes()).hexdigest() for name in ['C.csv','western-cv.csv','core-scores.csv','aperture-scores.csv']}
    eigen=np.linalg.eigvalsh(K)
    out=P/'independent-review';out.mkdir(exist_ok=False)
    receipt=dict(errors=errors,selected_joint_q=float(q.mean()),apertures=apertures,technical_rerun_byte_identical=identical,K_effective_rank=float(np.trace(K)**2/np.sum(eigen**2)),K_condition=float(eigen[-1]/eigen[0]),passed=bool(max(errors.values())<1e-10 and all(identical.values())),note='Independent second moments, inverse trace joint score and direct block-sum aperture variances. No full 24192-square matrix allocated.')
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt,indent=2))
    if not receipt['passed']:raise RuntimeError('Independent verification failed')

if __name__=='__main__':main()
