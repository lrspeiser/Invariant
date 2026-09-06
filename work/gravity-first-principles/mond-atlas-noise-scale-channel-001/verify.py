"""Manual cosine-basis transform, covariance refit and inverse score replay."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'AGENTS.md').exists());RUN=P/'run001'

def main():
    cfg=json.loads((ROOT/'configs/mond_atlas_aperture_noise_v1.json').read_text(encoding='utf-8'));source=ROOT/cfg['input'];assert hashlib.sha256(source.read_bytes()).hexdigest()==cfg['input_sha256']
    with np.load(source) as z:west=z['training'];east=z['validation']
    frozen=json.loads((RUN/'models-before-east.json').read_text(encoding='utf-8'));mean=west.mean(axis=(0,1,2));j=np.arange(24);k=j[:,None]
    D=np.sqrt(2/24)*np.cos(np.pi*k*(2*j+1)/(2*24));D[0]/=np.sqrt(2)
    transform=lambda a:np.einsum('iy,byxc,jx->bijc',D,a-mean,D,optimize=True).reshape(len(a),576,42)
    train=transform(west);test=transform(east);ky,kx=np.meshgrid(j,j,indexing='ij');r2=(kx*kx+ky*ky).ravel();bands={'DC':r2==0,'low':(r2>=1)&(r2<=9),'middle':(r2>=10)&(r2<=64),'high':r2>64}
    covs={};maxC=0.;jointq=np.zeros(27);jointlp=np.zeros(27);diagnostics=[]
    for band,mask in bands.items():
        a=train[:,mask].reshape(-1,42);raw=a.T@a/len(a);diag=raw.diagonal().copy();floor=max(1e-12,1e-8*np.median(diag));raw+=np.diag(np.maximum(diag,floor)-diag);alpha=frozen['selected'][band];C=(1-alpha)*raw+alpha*np.diag(np.diag(raw));covs[band]=C;maxC=max(maxC,float(np.max(abs(C-np.array(frozen['matrices'][band][str(alpha)])))))
        inv=np.linalg.inv(C);q=np.einsum('bmi,ij,bmj->bm',test[:,mask],inv,test[:,mask]);jointq+=q.sum(axis=1);jointlp+=-.5*(q.sum(axis=1)+mask.sum()*(np.linalg.slogdet(C)[1]+42*np.log(2*np.pi)))
        eig,V=np.linalg.eigh(C);power=(test[:,mask]@V)**2/eig;diagnostics.append(dict(band=band,group='all',q=float(power.mean())))
        if band!='DC':
            local=r2[mask];median=np.median(local)
            for label,keep in [('frequency_lower',local<=median),('frequency_upper',local>median)]:diagnostics.append(dict(band=band,group=label,q=float(power[:,keep].mean())))
        for index,indices in enumerate(np.array_split(np.arange(42),4)):diagnostics.append(dict(band=band,group=f'channel_eigen_quartile_{index+1}',q=float(power[:,:,indices].mean())))
    jointq/=576*42;jointlp/=576*42
    with (RUN/'core-scores.csv').open(encoding='utf-8',newline='') as f:rows=[r for r in csv.DictReader(f) if r['model']=='selected']
    errors=dict(covariance=maxC,joint_q=float(np.max(abs(jointq-[float(r['q']) for r in rows]))),joint_logpdf=float(np.max(abs(jointlp-[float(r['logpdf']) for r in rows]))))
    summary=json.loads((RUN/'summary.json').read_text(encoding='utf-8'))
    for side in (1,2,4,8,12,24):
        scores=[];traces=[]
        for y in range(0,24,side):
            for x in range(0,24,side):
                by=D[:,y:y+side].sum(axis=1)/side;bx=D[:,x:x+side].sum(axis=1)/side;weight=np.outer(by,bx).ravel()**2
                C=sum(weight[mask].sum()*covs[band] for band,mask in bands.items());obs=(east-mean)[:,y:y+side,x:x+side].mean(axis=(1,2));scores.append(np.einsum('bi,ij,bj->b',obs,np.linalg.inv(C),obs)/42);traces.append(np.sum(obs*obs,axis=1)/np.trace(C))
        q=float(np.mean(scores));tr=float(np.mean(traces));ref=next(r for r in summary['selected_apertures'] if r['side']==side);errors['aperture_'+str(side)]=max(abs(q-ref['q']),abs(tr-ref['trace_ratio']))
    errors['diagnostics']=max(abs(r['q']-next(v['q'] for v in summary['selected_diagnostics'] if v['band']==r['band'] and v['group']==r['group'])) for r in diagnostics)
    with (RUN/'western-cv.csv').open(encoding='utf-8',newline='') as f:cv=list(csv.DictReader(f))
    selected={b:max([.1,.3,.6,1.],key=lambda a:np.mean([float(r['logpdf']) for r in cv if r['band']==b and float(r['alpha'])==a])) for b in bands}
    receipt=dict(errors=errors,western_selection_matches=selected==frozen['selected'],manual_cosine_basis=True,passed=bool(max(errors.values())<1e-10 and selected==frozen['selected']),diagnostic_groups=len(diagnostics),selected_core_scores=27,selected_joint_q=float(jointq.mean()),limitations='Shared raw packet; independent transform, moments, inverse scores and aperture projections. Cross-mode covariance not validated.')
    out=P/'independent-review';out.mkdir(exist_ok=False);(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt,indent=2))
    if not receipt['passed']:raise RuntimeError('Independent replay failed')

if __name__=='__main__':main()
