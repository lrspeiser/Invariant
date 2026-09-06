"""Independent moment fit, inverse joint scores and aperture replay."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,csv,sys,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.distance import cdist
P=Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'AGENTS.md').exists());RUN=P/'run001'

def main():
    cfg=json.loads((ROOT/'configs/mond_atlas_aperture_noise_v1.json').read_text(encoding='utf-8'));source=ROOT/cfg['input'];assert hashlib.sha256(source.read_bytes()).hexdigest()==cfg['input_sha256']
    with np.load(source) as z:west=z['training'];east=z['validation']
    frozen=json.loads((RUN/'models-before-east.json').read_text(encoding='utf-8'));name=frozen['selected'];model=frozen['models'][name];spec=model['spec']
    mean=west.mean(axis=(0,1,2));res=west-mean;flat=res.reshape(-1,42);raw=flat.T@flat/len(flat);C=.7*raw+.3*np.diag(np.diag(raw));ci=np.linalg.inv(C);weighted=res@ci
    moments=[]
    for dy in range(spec['cap']+1):
        for dx in range(-spec['cap'],spec['cap']+1):
            if dy==0 and dx<0:continue
            # Explicit coordinate arrays provide independent displacement indexing.
            yy,xx=np.meshgrid(np.arange(24-dy),np.arange(max(0,-dx),min(24,24-dx)),indexing='ij')
            value=float(np.mean(weighted[:,yy,xx]*res[:,yy+dy,xx+dx]))
            moments.append((dy,dx,value,(24-dy)*(24-abs(dx))))
    widths=np.array([.5,1,2,4,8]);d2=np.array([dy*dy+dx*dx for dy,dx,_,_ in moments]);target=np.array([v for _,_,v,_ in moments]);weight=np.sqrt([n for *_,n in moments]);design=np.column_stack((d2==0,np.exp(-.5*d2[:,None]/widths**2)))
    floor=target[0]*spec['floor'];solution=lsq_linear(design*weight[:,None],(target-floor*(d2==0))*weight,bounds=(0,np.inf),method='bvls',tol=1e-12)
    coefficient=solution.x;coefficient[0]+=floor
    errors=dict(C=float(np.max(abs(C-np.array(frozen['C'])))),coefficients=float(np.max(abs(coefficient-np.array(model['coefficients'])))))
    coords=np.array([(y,x) for y in range(24) for x in range(24)]);distance=cdist(coords,coords,'sqeuclidean');K=np.eye(576)*coefficient[0]
    for amplitude,width in zip(coefficient[1:],widths):K+=amplitude*np.exp(-distance/(2*width*width))
    ki=np.linalg.inv(K);e=(east-mean).reshape(27,576,42);q=np.array([np.sum(a*(ki@a@ci)) for a in e])/(576*42);ld=42*np.linalg.slogdet(K)[1]+576*np.linalg.slogdet(C)[1];lp=-.5*(q*576*42+ld+576*42*np.log(2*np.pi))/(576*42)
    with (RUN/'east-core-scores.csv').open(encoding='utf-8',newline='') as f:rows=[r for r in csv.DictReader(f) if r['model']==name]
    errors['joint_q']=float(np.max(abs(q-[float(r['q']) for r in rows])));errors['joint_logpdf']=float(np.max(abs(lp-[float(r['logpdf']) for r in rows])))
    summary=json.loads((RUN/'summary.json').read_text(encoding='utf-8'));aps=[]
    for side in (1,2,4,8,12,24):
        variance=[];obs=[]
        for y in range(0,24,side):
            for x in range(0,24,side):
                idx=np.array([iy*24+ix for iy in range(y,y+side) for ix in range(x,x+side)]);variance.append(K[np.ix_(idx,idx)].sum()/side**4);obs.append((east-mean)[:,y:y+side,x:x+side].mean(axis=(1,2)))
        values=np.stack(obs,axis=1);variance=np.array(variance);aq=float(np.mean(np.einsum('bti,ij,btj->bt',values,ci,values)/variance[None,:]/42));tr=float(np.mean(np.sum(values*values,axis=2))/(np.trace(C)*variance.mean()));ref=next(r for r in summary['selected_apertures'] if r['side']==side)
        errors['aperture_'+str(side)]=max(abs(aq-ref['q']),abs(tr-ref['trace_ratio']));aps.append(dict(side=side,q=aq,trace_ratio=tr))
    with (RUN/'western-cv.csv').open(encoding='utf-8',newline='') as f:cv=list(csv.DictReader(f))
    modelnames=list(frozen['models']);picked=max(modelnames,key=lambda n:np.mean([float(r['logpdf']) for r in cv if r['model']==n]))
    # Descriptive diagnostic only: no model modification after seeing these modes.
    eigen,U=np.linalg.eigh(K);projected=np.einsum('is,bsc->bic',U.T,e);mode_q=np.mean(np.einsum('bsi,ij,bsj->bs',projected,ci,projected),axis=0)/(42*eigen)
    diagnostic=[dict(eigenvalue_quartile=i+1,mean_q=float(mode_q[indices].mean()),minimum_eigenvalue=float(eigen[indices].min()),maximum_eigenvalue=float(eigen[indices].max())) for i,indices in enumerate(np.array_split(np.arange(576),4))]
    result=dict(errors=errors,independent_optimizer_success=bool(solution.success),western_selection_matches=picked==name,passed=bool(max(errors.values())<1e-10 and picked==name and solution.success),selected_joint_q=float(q.mean()),apertures=aps,spatial_mode_diagnostic=diagnostic,note='Independent inverse scores and bounded least-squares moment fit. Mode diagnostics are descriptive and did not tune model.')
    out=P/'independent-review';out.mkdir(exist_ok=False);(out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
    if not result['passed']:raise RuntimeError('Independent replay failed')

if __name__=='__main__':main()
