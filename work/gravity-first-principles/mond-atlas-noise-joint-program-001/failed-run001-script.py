"""Separable within-core spatial and spectral background model."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import sys,json,csv,subprocess
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
from mond_atlas_native_covariance import regularized_covariance
P=ROOT/'work/gravity-first-principles/mond-atlas-noise-joint-program-001'
ALPHAS=[.1,.3,.6,1.]

def fit(data):
    data=np.asarray(data,float);b,h,w,c=data.shape
    if b<2 or not np.isfinite(data).all():raise ValueError('Finite multiple cores required')
    mean=data.mean(axis=(0,1,2));e=(data-mean).reshape(b,h*w,c)
    C=regularized_covariance(e.reshape(-1,c),dict(kind='full',shrinkage=.3));L=np.linalg.cholesky(C)
    z=solve_triangular(L,e.reshape(-1,c).T,lower=True).T.reshape(b,h*w,c)
    K=np.einsum('bic,bjc->ij',z,z)/(b*c);K=(K+K.T)/2
    floor=max(1e-12,np.median(np.diag(K))*1e-8);K+=np.diag(np.maximum(np.diag(K),floor)-np.diag(K))
    covs={alpha:(1-alpha)*K+alpha*np.diag(np.diag(K)) for alpha in ALPHAS}
    return mean,C,covs

def statistics(data,mean,C,K):
    e=(data-mean).reshape(len(data),len(K),len(C));lc=np.linalg.cholesky(C);lk=np.linalg.cholesky(K)
    z=solve_triangular(lc,e.reshape(-1,len(C)).T,lower=True).T.reshape(e.shape)
    spatial=z.transpose(1,0,2).reshape(len(K),-1)
    whitened=solve_triangular(lk,spatial,lower=True).reshape(len(K),len(data),len(C))
    q=np.sum(whitened**2,axis=(0,2));n=len(K)*len(C)
    logdet=len(C)*2*np.log(np.diag(lk)).sum()+len(K)*2*np.log(np.diag(lc)).sum()
    return q/n,-.5*(q+logdet+n*np.log(2*np.pi))/n

def aperture_operator(h,w,side):
    rows=[]
    for y in range(0,h,side):
        for x in range(0,w,side):
            a=np.zeros((h,w));a[y:y+side,x:x+side]=1/side**2;rows.append(a.ravel())
    return np.array(rows)

def aperture_score(data,mean,C,K,side):
    b,h,w,c=data.shape;A=aperture_operator(h,w,side);e=(data-mean).reshape(b,h*w,c)
    v=np.einsum('ts,bsc->btc',A,e);spatial=A@K@A.T;variance=np.diag(spatial)
    lc=np.linalg.cholesky(C);z=solve_triangular(lc,v.reshape(-1,c).T,lower=True).T.reshape(v.shape)
    q=np.sum(z*z,axis=2)/variance[None,:]/c
    trace=np.sum(v*v,axis=2).mean(axis=1)/(np.trace(C)*variance.mean())
    return q.mean(axis=1),trace

def run():
    out=P/'run001';out.mkdir(parents=True,exist_ok=False)
    try:
        cfg=read_json(ROOT/'configs/mond_atlas_aperture_noise_v1.json');source=ROOT/cfg['input']
        assert digest(source)==cfg['input_sha256']
        geometry=ROOT/'work/gravity-first-principles/mond-atlas-native-covariance-001/run-001/block-geometry.csv'
        paths=[source,geometry,Path(__file__),P/'test_joint.py',P/'PREFLIGHT.md',ROOT/'scripts/mond_atlas_native_covariance.py']
        write_json(out/'bindings.json',dict(previous_exposure=True,files={p.relative_to(ROOT).as_posix():digest(p) for p in paths}))
        test=subprocess.run([sys.executable,str(P/'test_joint.py')],capture_output=True,text=True,cwd=ROOT);(out/'tests.log').write_text(test.stdout+test.stderr,encoding='utf-8')
        if test.returncode:raise RuntimeError('Pre-access test failure')
        with geometry.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        westrows=[r for r in rows if r['region']=='training'];eastrows=[r for r in rows if r['region']=='validation'];folds=np.array([int(r['fold']) for r in westrows])
        with np.load(source) as z:west=z['training']
        assert west.shape==(29,24,24,42)
        cv=[];rank={a:[] for a in ALPHAS}
        for fold in sorted(set(folds)):
            mean,C,covs=fit(west[folds!=fold])
            for alpha,K in covs.items():
                q,lp=statistics(west[folds==fold],mean,C,K)
                for i,qn,l in zip(np.flatnonzero(folds==fold),q,lp):cv.append(dict(fold=int(fold),alpha=alpha,core=westrows[i]['block_id'],q=float(qn),logpdf=float(l)));rank[alpha].append(float(l))
        selected=max(ALPHAS,key=lambda a:np.mean(rank[a]));mean,C,covs=fit(west)
        # Numerical model parameters are compact generated artifacts, not raw pixels.
        for alpha,K in covs.items():
            np.savetxt(out/f'K-{alpha}.csv',K,delimiter=',',fmt='%.17g')
        np.savetxt(out/'C.csv',C,delimiter=',',fmt='%.17g')
        before=out/'models-before-east.json';write_json(before,dict(mean=mean.tolist(),selected_alpha=selected,western_ranking={str(a):float(np.mean(rank[a])) for a in ALPHAS},matrix_hashes={p.name:digest(p) for p in out.glob('*.csv')},east_read_this_run=False))
        beforehash=digest(before);write_csv(out/'western-cv.csv',cv)
        with np.load(source) as z:east=z['validation']
        assert east.shape==(27,24,24,42)
        results=[];apertures=[];cores=[]
        for alpha,K in covs.items():
            q,lp=statistics(east,mean,C,K);meanq=float(q.mean())
            results.append(dict(alpha=alpha,selected=alpha==selected,q=meanq,logpdf=float(lp.mean()),q_pass=.8<=meanq<=1.2))
            for i in range(27):cores.append(dict(alpha=alpha,core=eastrows[i]['block_id'],q=float(q[i]),logpdf=float(lp[i])))
            for side in [1,2,4,8,12,24]:
                aq,tr=aperture_score(east,mean,C,K,side)
                apertures.append(dict(alpha=alpha,selected=alpha==selected,side=side,q=float(aq.mean()),trace_ratio=float(tr.mean()),q_pass=.8<=aq.mean()<=1.2))
        assert digest(before)==beforehash
        write_csv(out/'core-scores.csv',cores);write_csv(out/'aperture-scores.csv',apertures)
        write_json(out/'summary.json',dict(status='JOINT_WITHIN_BACKGROUND_CORE_ONLY',selected_alpha=selected,models=results,apertures=apertures,observed_gravity_scores=0,emission_likelihood_admitted=False,cross_core_covariance=False))
        print(json.dumps(dict(models=results,selected_apertures=[r for r in apertures if r['selected']]),indent=2))
    except Exception as exc:write_json(out/'failure.json',dict(error=repr(exc)));raise

if __name__=='__main__':run()
