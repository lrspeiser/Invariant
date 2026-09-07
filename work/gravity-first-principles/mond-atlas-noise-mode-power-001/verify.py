"""Independent manual cosine transform, selected power/C refit and scores."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'AGENTS.md').exists());RUN=P/'run001'

def main():
    cfg=json.loads((ROOT/'configs/mond_atlas_aperture_noise_v1.json').read_text(encoding='utf-8'));source=ROOT/cfg['input'];assert hashlib.sha256(source.read_bytes()).hexdigest()==cfg['input_sha256']
    with np.load(source) as z:west=z['training'];east=z['validation']
    frozen=json.loads((RUN/'models-before-east.json').read_text(encoding='utf-8'));mean=west.mean(axis=(0,1,2));j=np.arange(24);D=np.sqrt(2/24)*np.cos(np.pi*j[:,None]*(2*j+1)/48);D[0]/=np.sqrt(2)
    trans=lambda a:np.einsum('iy,byxc,jx->bijc',D,a-mean,D,optimize=True).reshape(len(a),576,42)
    train=trans(west);test=trans(east);coords=np.array([(y,x) for y in range(24) for x in range(24)]);r2=np.sum(coords*coords,axis=1);bands={'DC':r2==0,'low':(r2>=1)&(r2<=9),'middle':(r2>=10)&(r2<=64),'high':r2>64}
    covs={};powers={};qtotal=np.zeros(27);lptotal=np.zeros(27);errors=dict(power=0.,covariance=0.);modes={};diagnostics={};whitened=np.zeros_like(test)
    for band,mask in bands.items():
        ref=frozen['models'][band][frozen['choice'][band]];raw=np.mean(train[:,mask]**2,axis=(0,2));xy=coords[mask];pooled=np.array([raw[np.sum((xy-point)**2,axis=1)<=ref['radius']**2].mean() for point in xy]);floor=max(1e-300,1e-12*np.median(raw[raw>0]));pooled=np.maximum(pooled,floor);power=pooled/pooled.mean();powers[band]=power
        a=(train[:,mask]/np.sqrt(power)[None,:,None]).reshape(-1,42);sample=a.T@a/len(a);diagonal=sample.diagonal().copy();cf=max(1e-12,1e-8*np.median(diagonal));sample+=np.diag(np.maximum(diagonal,cf)-diagonal);C=(1-ref['alpha'])*sample+ref['alpha']*np.diag(np.diag(sample));covs[band]=C
        errors['power']=max(errors['power'],float(np.max(abs(power-np.array(ref['power'])))));errors['covariance']=max(errors['covariance'],float(np.max(abs(C-np.array(ref['C'])))))
        normalized=test[:,mask]/np.sqrt(power)[None,:,None];inv=np.linalg.inv(C);q=np.einsum('bmi,ij,bmj->bm',normalized,inv,normalized);qtotal+=q.sum(axis=1);lptotal+=-.5*(q.sum(axis=1)+mask.sum()*np.linalg.slogdet(C)[1]+42*np.log(power).sum()+mask.sum()*42*np.log(2*np.pi))
        for index,value in zip(np.flatnonzero(mask),q.mean(axis=0)/42):modes[int(index)]=float(value)
        L=np.linalg.cholesky(C);whitened[:,mask]=normalized@np.linalg.inv(L).T
        eig,V=np.linalg.eigh(C);ep=(normalized@V)**2/eig;diagnostics[(band,'all')]=float(ep.mean())
        if band!='DC':
            local=r2[mask];median=np.median(local)
            diagnostics[(band,'frequency_lower')]=float(ep[:,local<=median].mean());diagnostics[(band,'frequency_upper')]=float(ep[:,local>median].mean())
        for k,indices in enumerate(np.array_split(np.arange(42),4)):diagnostics[(band,f'channel_eigen_quartile_{k+1}')]=float(ep[:,:,indices].mean())
    read=lambda f:list(csv.DictReader((RUN/f).open(encoding='utf-8',newline='')))
    scores=read('core-scores.csv');qtotal/=576*42;lptotal/=576*42;errors['joint_q']=float(np.max(abs(qtotal-[float(r['q']) for r in scores])));errors['joint_logpdf']=float(np.max(abs(lptotal-[float(r['logpdf']) for r in scores])))
    errors['per_mode']=max(abs(modes[int(r['mode'])]-float(r['q'])) for r in read('mode-scores.csv'));errors['diagnostics']=max(abs(diagnostics[(r['band'],r['group'])]-float(r['q'])) for r in read('diagnostics.csv'))
    errors['cross_mode']=0.
    for row in read('cross-mode-products.csv'):
        a=whitened[:,int(row['mode_a'])].ravel();b=whitened[:,int(row['mode_b'])].ravel();value=float(a@b/np.sqrt((a@a)*(b@b)));errors['cross_mode']=max(errors['cross_mode'],abs(value-float(row['normalized_product'])))
    summary=json.loads((RUN/'summary.json').read_text(encoding='utf-8'))
    for side in (1,2,4,8,12,24):
        scores=[];traces=[]
        for y in range(0,24,side):
            for x in range(0,24,side):
                by=D[:,y:y+side].sum(axis=1)/side;bx=D[:,x:x+side].sum(axis=1)/side;weights=np.outer(by,bx).ravel()**2;C=sum(np.sum(weights[mask]*powers[band])*covs[band] for band,mask in bands.items());v=(east-mean)[:,y:y+side,x:x+side].mean(axis=(1,2));scores.append(np.einsum('bi,ij,bj->b',v,np.linalg.inv(C),v)/42);traces.append(np.sum(v*v,axis=1)/np.trace(C))
        ref=next(r for r in summary['apertures'] if r['side']==side);errors['aperture_'+str(side)]=max(abs(float(np.mean(scores))-ref['q']),abs(float(np.mean(traces))-ref['trace_ratio']))
    cv=read('western-cv.csv');selected={b:max(frozen['models'][b],key=lambda candidate:np.mean([float(r['logpdf']) for r in cv if r['band']==b and r['candidate']==candidate])) for b in bands}
    result=dict(errors=errors,manual_basis_and_source_power_refit=True,selected_matches=selected==frozen['choice'],passed=bool(max(errors.values())<1e-10 and selected==frozen['choice']),mode_scores_replayed=576,crossmode_pairs_replayed=1104,diagnostic_groups_replayed=26,observed_gravity_scores=0)
    out=P/'independent-review';out.mkdir(exist_ok=False);(out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
    if not result['passed']:raise RuntimeError('Independent replay failed')

if __name__=='__main__':main()
