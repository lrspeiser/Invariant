"""Western-selected stationary positive Gaussian covariance mixture."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,csv,subprocess
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import nnls
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
from mond_atlas_native_covariance import regularized_covariance
from mond_atlas_noise_joint_program import statistics,aperture_score
P=ROOT/'work/gravity-first-principles/mond-atlas-noise-stationary-001'
WIDTHS=np.array([.5,1.,2.,4.,8.])
SPECS=[dict(cap=c,floor=f,id=f'cap{c}_floor{f}') for c in (4,8,12) for f in (1e-6,1e-4,.01)]

def lag_products(z,cap):
    b,h,w,c=z.shape;rows=[]
    for dy in range(cap+1):
        for dx in range(-cap,cap+1):
            if dy==0 and dx<0:continue
            if dx>=0:a=z[:,:h-dy,:w-dx];v=z[:,dy:,dx:]
            else:a=z[:,:h-dy,-dx:];v=z[:,dy:,:w+dx]
            rows.append(dict(dy=dy,dx=dx,pairs=(h-dy)*(w-abs(dx)),product=float(np.mean(a*v))))
    return rows

def coefficient_fit(lags,spec):
    rows=[r for r in lags if r['dy']<=spec['cap'] and abs(r['dx'])<=spec['cap']]
    d2=np.array([r['dx']**2+r['dy']**2 for r in rows]);target=np.array([r['product'] for r in rows]);weight=np.sqrt([r['pairs'] for r in rows])
    design=np.column_stack((d2==0,np.exp(-d2[:,None]/(2*WIDTHS[None,:]**2))))
    variance=float(target[d2==0][0]);floor=spec['floor']*variance
    coefficient,residual=nnls(weight[:,None]*design,weight*(target-floor*design[:,0]));coefficient[0]+=floor
    return dict(coefficients=coefficient.tolist(),white_minimum=floor,weighted_lag_residual=float(residual),lag_count=len(rows),source_zero_lag=variance)

def covariance(coefficients,h=24,w=24):
    x,y=np.meshgrid(np.arange(h),np.arange(w),indexing='ij');xy=np.column_stack((x.ravel(),y.ravel()));d2=np.sum((xy[:,None,:]-xy[None,:,:])**2,axis=2)
    K=np.eye(h*w)*coefficients[0]
    for amplitude,width in zip(coefficients[1:],WIDTHS):K+=amplitude*np.exp(-d2/(2*width**2))
    return K

def fit(data):
    a=np.asarray(data,float);mean=a.mean(axis=(0,1,2));e=a-mean
    C=regularized_covariance(e.reshape(-1,a.shape[-1]),dict(kind='full',shrinkage=.3));L=np.linalg.cholesky(C)
    z=solve_triangular(L,e.reshape(-1,len(C)).T,lower=True).T.reshape(e.shape)
    lags=lag_products(z,12);fitted={spec['id']:dict(spec=spec,**coefficient_fit(lags,spec)) for spec in SPECS}
    return mean,C,fitted,lags

def run():
    out=P/'run001';out.mkdir(parents=True,exist_ok=False)
    try:
        cfg=read_json(ROOT/'configs/mond_atlas_aperture_noise_v1.json');source=ROOT/cfg['input'];assert digest(source)==cfg['input_sha256']
        geo=ROOT/'work/gravity-first-principles/mond-atlas-native-covariance-001/run-001/block-geometry.csv'
        assert digest(geo)==read_json(geo.parent/'geometry-freeze.json')['block_geometry_sha256']
        deps=[Path(__file__),P/'PREFLIGHT.md',P/'test_stationary.py',source,geo,ROOT/'scripts/mond_atlas_noise_joint_program.py',ROOT/'scripts/mond_atlas_native_covariance.py']
        write_json(out/'pre-access-bindings.json',dict(previous_exposure=True,specs=SPECS,widths_native_pixels=WIDTHS.tolist(),files={p.relative_to(ROOT).as_posix():digest(p) for p in deps}))
        test=subprocess.run([sys.executable,str(P/'test_stationary.py')],cwd=ROOT,capture_output=True,text=True);(out/'tests.log').write_text(test.stdout+test.stderr,encoding='utf-8')
        if test.returncode:raise RuntimeError('Pre-access tests failed')
        with geo.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        wr=[r for r in rows if r['region']=='training'];er=[r for r in rows if r['region']=='validation'];folds=np.array([int(r['fold']) for r in wr])
        with np.load(source) as z:west=z['training']
        assert west.shape==(29,24,24,42)
        cv=[];fits=[];lagrows=[];ranking={s['id']:[] for s in SPECS}
        for fold in sorted(set(folds)):
            mean,C,models,lags=fit(west[folds!=fold]);lagrows.extend(dict(fold=int(fold),**l) for l in lags)
            for name,model in models.items():
                K=covariance(model['coefficients']);q,lp=statistics(west[folds==fold],mean,C,K);fits.append(dict(fold=int(fold),model=name,**model))
                for j,v,l in zip(np.flatnonzero(folds==fold),q,lp):cv.append(dict(fold=int(fold),model=name,core=wr[j]['block_id'],q=float(v),logpdf=float(l)));ranking[name].append(float(l))
        selected=max(ranking,key=lambda name:np.mean(ranking[name]));mean,C,models,lags=fit(west)
        before=out/'models-before-east.json';write_json(before,dict(selected=selected,mean=mean.tolist(),C=C.tolist(),models=models,ranking={n:float(np.mean(v)) for n,v in ranking.items()},east_opened_this_run=False));modelhash=digest(before)
        write_json(out/'western-fit-models.json',fits);write_csv(out/'western-cv.csv',cv);write_csv(out/'western-lags.csv',lagrows);write_csv(out/'full-west-lags.csv',lags)
        with np.load(source) as z:east=z['validation']
        assert east.shape==(27,24,24,42)
        joint=[];aps=[];cores=[]
        for name,model in models.items():
            K=covariance(model['coefficients']);q,lp=statistics(east,mean,C,K);qm=float(q.mean())
            joint.append(dict(model=name,selected=name==selected,q=qm,logpdf=float(lp.mean()),q_pass=bool(.8<=qm<=1.2),white_fraction=float(model['coefficients'][0]/sum(model['coefficients'])),coefficients=model['coefficients']))
            for j in range(27):cores.append(dict(model=name,core=er[j]['block_id'],q=float(q[j]),logpdf=float(lp[j])))
            for side in [1,2,4,8,12,24]:
                aq,tr=aperture_score(east,mean,C,K,side);aps.append(dict(model=name,selected=name==selected,side=side,q=float(aq.mean()),trace_ratio=float(tr.mean()),q_pass=bool(.8<=aq.mean()<=1.2)))
        assert digest(before)==modelhash
        write_csv(out/'east-core-scores.csv',cores);write_csv(out/'aperture-scores.csv',aps)
        write_json(out/'summary.json',dict(status='PHENOMENOLOGICAL_STATIONARY_BACKGROUND_ONLY',selected=selected,joint=joint,selected_apertures=[r for r in aps if r['selected']],new_raw_bytes=0,emission_likelihood_admitted=False,observed_gravity_scores=0))
        print(json.dumps(dict(selected=next(r for r in joint if r['selected']),apertures=[r for r in aps if r['selected']]),indent=2))
    except Exception as exc:write_json(out/'failure.json',dict(error=repr(exc)));raise

if __name__=='__main__':run()
