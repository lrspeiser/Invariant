"""Manual source transform and independent predictive-variance score replay."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'AGENTS.md').exists());RUN=P/'run001'

def main():
    cfg=json.loads((ROOT/'configs/mond_atlas_aperture_noise_v1.json').read_text(encoding='utf-8'));source=ROOT/cfg['input'];assert hashlib.sha256(source.read_bytes()).hexdigest()==cfg['input_sha256']
    with np.load(source) as z:west=z['training'];east=z['validation']
    frozen=json.loads((RUN/'models-before-east.json').read_text(encoding='utf-8'));mean=west.mean(axis=(0,1,2));j=np.arange(24);D=np.sqrt(2/24)*np.cos(np.pi*j[:,None]*(2*j+1)/48);D[0]/=np.sqrt(2)
    trans=lambda a:np.einsum('iy,byxc,jx->bijc',D,a-mean,D,optimize=True).reshape(len(a),576,42)
    train=trans(west);test=trans(east);coords=np.array([(y,x) for y in range(24) for x in range(24)]);r2=np.sum(coords*coords,axis=1);bands={'DC':r2==0,'low':(r2>=1)&(r2<=9),'middle':(r2>=10)&(r2<=64),'high':r2>64};models={};error=0.
    for band,mask in bands.items():
        ref=frozen['base_models'][band];raw=np.mean(train[:,mask]**2,axis=(0,2));power=raw/raw.mean();a=(train[:,mask]/np.sqrt(power)[None,:,None]).reshape(-1,42);C=a.T@a/len(a);diag=C.diagonal().copy();floor=max(1e-12,1e-8*np.median(diag));C+=np.diag(np.maximum(diag,floor)-diag);C=(1-ref['alpha'])*C+ref['alpha']*np.diag(np.diag(C));models[band]=dict(power=power,C=C);error=max(error,float(np.max(abs(C-np.array(ref['C'])))))
    S=np.diag(models['DC']['C']);low=np.diag(models['low']['C']);n=len(west);candidates={};maxvariance=0.
    for name,ref in frozen['dc_candidates'].items():
        if name=='baseline':variance=S
        else:
            target=np.full(42,S.mean()) if ref['target']=='constant' else low*S.sum()/low.sum()
            # Separate unbiased-sample and mean-error factors, rather than stored correction.
            variance=((1-ref['weight'])*S+ref['weight']*target)*n/(n-1)*(1+1/n)
        candidates[name]=variance;maxvariance=max(maxvariance,float(np.max(abs(variance-np.array(ref['variance'])))))
    read=lambda name:list(csv.DictReader((RUN/name).open(encoding='utf-8',newline='')))
    dcerror=0.
    # Source geometry order is inherited and binds eastern row order.
    geo=ROOT/'work/gravity-first-principles/mond-atlas-native-covariance-001/run-001/block-geometry.csv'
    with geo.open(encoding='utf-8',newline='') as f:er=[r for r in csv.DictReader(f) if r['region']=='validation']
    index={r['block_id']:i for i,r in enumerate(er)}
    for row in read('dc-candidate-scores.csv'):
        variance=candidates[row['candidate']];e=test[index[row['core']],0];q=float(np.sum(e*e/variance)/42);lp=-.5*(q*42+np.log(variance).sum()+42*np.log(2*np.pi))/42;dcerror=max(dcerror,abs(q-float(row['q'])),abs(lp-float(row['logpdf'])))
    models['DC']['C']=np.diag(candidates[frozen['selected']]);totalq=np.zeros(27);totallp=np.zeros(27);modes={};diagnostics={}
    for band,mask in bands.items():
        C=models[band]['C'];power=models[band]['power'];a=test[:,mask]/np.sqrt(power)[None,:,None];q=np.einsum('bmi,ij,bmj->bm',a,np.linalg.inv(C),a);totalq+=q.sum(axis=1);totallp+=-.5*(q.sum(axis=1)+mask.sum()*np.linalg.slogdet(C)[1]+42*np.log(power).sum()+42*mask.sum()*np.log(2*np.pi))
        for mode,v in zip(np.flatnonzero(mask),q.mean(axis=0)/42):modes[int(mode)]=float(v)
        eig,V=np.linalg.eigh(C);ep=(a@V)**2/eig;diagnostics[(band,'all')]=float(ep.mean())
        if band!='DC':
            local=r2[mask];median=np.median(local)
            for name,keep in [('frequency_lower',local<=median),('frequency_upper',local>median)]:diagnostics[(band,name)]=float(ep[:,keep].mean())
        for k,indices in enumerate(np.array_split(np.arange(42),4)):diagnostics[(band,f'channel_eigen_quartile_{k+1}')]=float(ep[:,:,indices].mean())
    channel_q=np.mean(test[:,0]**2,axis=0)/candidates[frozen['selected']]
    for k,indices in enumerate(np.array_split(np.argsort(S),4)):diagnostics[('DC',f'fixed_original_channel_quartile_{k+1}')]=float(channel_q[indices].mean())
    rows=[r for r in read('joint-core-scores.csv') if r['model']=='selected'];totalq/=576*42;totallp/=576*42
    errors=dict(base_covariance=error,dc_variance=maxvariance,all_dc_scores=dcerror,joint_q=float(np.max(abs(totalq-[float(r['q']) for r in rows]))),joint_logpdf=float(np.max(abs(totallp-[float(r['logpdf']) for r in rows]))),mode_scores=max(abs(modes[int(r['mode'])]-float(r['q'])) for r in read('mode-scores.csv')),channel_scores=max(abs(channel_q[int(r['channel'])]-float(r['q'])) for r in read('channel-scores.csv')),diagnostics=max(abs(diagnostics[(r['band'],r['group'])]-float(r['q'])) for r in read('diagnostics.csv')))
    for side in (1,2,4,8,12,24):
        qvals=[];traces=[]
        for y in range(0,24,side):
            for x in range(0,24,side):
                weights=np.outer(D[:,y:y+side].sum(axis=1)/side,D[:,x:x+side].sum(axis=1)/side).ravel()**2;C=sum(np.sum(weights[mask]*models[b]['power'])*models[b]['C'] for b,mask in bands.items());a=(east-mean)[:,y:y+side,x:x+side].mean(axis=(1,2));qvals.append(np.einsum('bi,ij,bj->b',a,np.linalg.inv(C),a)/42);traces.append(np.sum(a*a,axis=1)/np.trace(C))
        ref=next(r for r in read('apertures.csv') if r['model']=='selected' and int(r['side'])==side);errors['aperture_'+str(side)]=max(abs(float(np.mean(qvals))-float(ref['q'])),abs(float(np.mean(traces))-float(ref['trace_ratio'])))
    cv=read('western-cv.csv');selected=max(candidates,key=lambda c:np.mean([float(r['logpdf']) for r in cv if r['candidate']==c]));receipt=dict(errors=errors,western_selection_matches=selected==frozen['selected'],all_dc_candidate_core_scores=297,selected_core_scores=27,mode_scores=576,channel_scores=42,diagnostic_groups=30,passed=bool(max(errors.values())<1e-10 and selected==frozen['selected']),note='Independent manual DCT, moments, correction factors, inverse joint/DC scores and aperture projections. Source packet shared; optimizer selection replayed from western scores.')
    out=P/'independent-review';out.mkdir(exist_ok=False);(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt,indent=2))
    if not receipt['passed']:raise RuntimeError('Independent replay failed')

if __name__=='__main__':main()
