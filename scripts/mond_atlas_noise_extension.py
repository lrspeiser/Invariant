"""Training-only channel covariance regularization across fixed apertures."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,csv,subprocess
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
from mond_atlas_aperture_noise import tiles,scores
from mond_atlas_native_covariance import regularized_covariance

P=ROOT/'work/gravity-first-principles/mond-atlas-noise-extension-001'
SIDES=[1,2,4,8,12,24]

def models(data,side):
    a=np.asarray(data,float)
    if a.ndim!=4 or len(a)<2 or not np.isfinite(a).all():raise ValueError('Finite core,y,x,channel input required')
    mean=a.mean(axis=(0,1,2));res=a-mean;sample=tiles(res,side).reshape(-1,a.shape[-1])
    result={}
    for alpha in (.1,.3,.6,1.):result[f'full_{alpha}']=regularized_covariance(sample,dict(kind='full',shrinkage=alpha))
    for lag in (1,3,7):
        for alpha in (.1,.3):result[f'bartlett_{lag}_{alpha}']=regularized_covariance(sample,dict(kind='bartlett',max_lag=lag,shrinkage=alpha))
    pixel=regularized_covariance(res.reshape(-1,a.shape[-1]),dict(kind='full',shrinkage=.1))
    corr=pixel/np.sqrt(np.outer(np.diag(pixel),np.diag(pixel)))
    diagonal=np.diag(result['full_0.1'])
    target=corr*np.sqrt(np.outer(diagonal,diagonal))
    for weight in (0.,.25,.5,.75,1.):result[f'pixelcorr_mixture_{weight}']=(1-weight)*target+weight*result['full_0.1']
    for cov in result.values():np.linalg.cholesky(cov)
    return mean,result

def run():
    out=P/'run001';out.mkdir(parents=True,exist_ok=False)
    try:
        cfg=read_json(ROOT/'configs/mond_atlas_aperture_noise_v1.json');source=ROOT/cfg['input']
        native=ROOT/'work/gravity-first-principles/mond-atlas-native-covariance-001/run-001'
        geometry=native/'block-geometry.csv';freeze=read_json(native/'geometry-freeze.json')
        if digest(source)!=cfg['input_sha256'] or digest(geometry)!=freeze['block_geometry_sha256']:raise RuntimeError('Source hash mismatch')
        paths=[source,geometry,P/'PREFLIGHT.md',P/'test_extension.py',Path(__file__),ROOT/'scripts/mond_atlas_aperture_noise.py',ROOT/'scripts/mond_atlas_native_covariance.py']
        write_json(out/'pre-access-bindings.json',dict(status='BACKGROUND_ONLY_SOURCE_LIKELIHOOD_BLOCKED',previous_exposure=True,files={p.relative_to(ROOT).as_posix():digest(p) for p in paths}))
        t=subprocess.run([sys.executable,str(P/'test_extension.py')],cwd=ROOT,capture_output=True,text=True);(out/'tests.log').write_text(t.stdout+t.stderr,encoding='utf-8')
        if t.returncode:raise RuntimeError('Pre-access tests failed')
        with geometry.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        westrows=[r for r in rows if r['region']=='training'];eastrows=[r for r in rows if r['region']=='validation']
        if len(westrows)!=29 or len(eastrows)!=27:raise RuntimeError('Geometry count mismatch')
        boxes=[tuple(int(r[k]) for k in ('y0','y1','x0','x1')) for r in rows]
        for i,(y0,y1,x0,x1) in enumerate(boxes):
            if y1-y0!=24 or x1-x0!=24:raise RuntimeError('Core shape')
            for a,b,c,d in boxes[:i]:
                if max(a,y0)<min(b,y1) and max(c,x0)<min(d,x1):raise RuntimeError('Core overlap')
        write_json(out/'geometry-audit.json',dict(cores=len(rows),pairwise_nonoverlap=True,original_geometry_hash_matched=True,ordering='filter original CSV by region, same extraction order as original script',mask_cleanliness_certified=False))
        with np.load(source) as packet:west=packet['training']
        if west.shape!=(29,24,24,42):raise RuntimeError('Western shape')
        folds=np.array([int(r['fold']) for r in westrows]);cv=[];fitted={};rankings=[];chosen={}
        for side in SIDES:
            bymodel={}
            for fold in sorted(set(folds)):
                mean,covs=models(west[folds!=fold],side)
                for name,cov in covs.items():
                    result=scores(west[folds==fold],mean,cov,side)
                    for j,q,lp in zip(np.flatnonzero(folds==fold),result['q_over_n'],result['logpdf_per_channel']):
                        cv.append(dict(side=side,fold=int(fold),model=name,core=westrows[j]['block_id'],q=float(q),logpdf=float(lp)));bymodel.setdefault(name,[]).append(float(lp))
            ranking=sorted(bymodel,key=lambda name:-np.mean(bymodel[name]));chosen[side]=ranking[0]
            rankings.extend(dict(side=side,rank=i+1,model=name,mean_west_held_logpdf=float(np.mean(bymodel[name]))) for i,name in enumerate(ranking))
            mean,covs=models(west,side);fitted[side]=dict(mean=mean,covariances=covs)
        write_csv(out/'western-cv.csv',cv);write_csv(out/'western-rankings.csv',rankings)
        before=out/'models-before-east.json'
        write_json(before,dict(chosen=chosen,models={str(side):dict(mean=v['mean'].tolist(),covariances={k:c.tolist() for k,c in v['covariances'].items()}) for side,v in fitted.items()},east_opened_this_run=False))
        beforehash=digest(before)
        with np.load(source) as packet:east=packet['validation']
        if east.shape!=(27,24,24,42):raise RuntimeError('Eastern shape')
        core=[];summary=[]
        for side,v in fitted.items():
            for name,cov in v['covariances'].items():
                result=scores(east,v['mean'],cov,side);q=float(result['q_over_n'].mean())
                summary.append(dict(side=side,model=name,selected=name==chosen[side],q=q,trace_ratio=float(result['trace_second_moment'].mean()/np.trace(cov)),logpdf=float(result['logpdf_per_channel'].mean()),descriptive_q_pass=.8<=q<=1.2,covariance_condition=float(np.linalg.cond(cov))))
                for j,row in enumerate(eastrows):core.append(dict(side=side,model=name,core=row['block_id'],**{k:float(a[j]) for k,a in result.items()}))
        if digest(before)!=beforehash:raise RuntimeError('Model freeze changed')
        write_csv(out/'all-east-core-scores.csv',core);write_csv(out/'all-scale-results.csv',summary)
        write_json(out/'summary.json',dict(status='BACKGROUND_MARGINAL_TRANSFER_ONLY',source_likelihood_admitted=False,models_frozen_before_east_sha256=beforehash,selected=[s for s in summary if s['selected']],new_raw_bytes=0,gravity_scores=0,mask_admitted=False))
        print(json.dumps([s for s in summary if s['selected']],indent=2))
    except Exception as exc:write_json(out/'failure.json',dict(error=repr(exc)));raise

if __name__=='__main__':run()
