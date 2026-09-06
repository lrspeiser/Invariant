"""Band-specific channel covariance in a complete orthonormal spatial DCT."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,csv,subprocess
from pathlib import Path
import numpy as np
from scipy.fft import dct,dctn
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
from mond_atlas_native_covariance import regularized_covariance,gaussian_statistics
P=ROOT/'work/gravity-first-principles/mond-atlas-noise-scale-channel-001'
ALPHAS=[.1,.3,.6,1.]

def geometry(h,w):
    ky,kx=np.meshgrid(np.arange(h),np.arange(w),indexing='ij');r2=(kx*kx+ky*ky).ravel()
    bands={'DC':r2==0,'low':(r2>=1)&(r2<=9),'middle':(r2>=10)&(r2<=64),'high':r2>64}
    U=np.kron(dct(np.eye(h),type=2,norm='ortho',axis=0).T,dct(np.eye(w),type=2,norm='ortho',axis=0).T)
    return {k:v for k,v in bands.items() if v.any()},U,r2

def transform(data,mean):return dctn(data-mean,type=2,norm='ortho',axes=(1,2)).reshape(len(data),-1,data.shape[-1])

def fit(data):
    mean=data.mean(axis=(0,1,2));coeff=transform(data,mean);bands,_,_=geometry(*data.shape[1:3]);models={}
    for name,mask in bands.items():models[name]={alpha:regularized_covariance(coeff[:,mask].reshape(-1,data.shape[-1]),dict(kind='full',shrinkage=alpha)) for alpha in ALPHAS}
    return mean,models

def score_coeff(coeff,bands,covs):
    q=np.zeros(len(coeff));lp=np.zeros(len(coeff));byband={}
    for band,mask in bands.items():
        _,a,b,_=gaussian_statistics(coeff[:,mask],covs[band]);q+=a.sum(axis=1);lp+=b.sum(axis=1);byband[band]=(a.mean(axis=1)/coeff.shape[-1],b.mean(axis=1)/coeff.shape[-1])
    return q/(coeff.shape[1]*coeff.shape[2]),lp/(coeff.shape[1]*coeff.shape[2]),byband

def aperture(data,mean,bands,U,covs,side):
    b,h,w,c=data.shape;res=data-mean;q=np.zeros(b);tr=np.zeros(b);count=0
    for y in range(0,h,side):
        for x in range(0,w,side):
            operator=np.zeros((h,w));operator[y:y+side,x:x+side]=1/side**2;weights=(operator.ravel()@U)**2
            covariance=sum(weights[mask].sum()*covs[band] for band,mask in bands.items())
            v=res[:,y:y+side,x:x+side].mean(axis=(1,2));_,quadratic,_,_=gaussian_statistics(v,covariance)
            q+=quadratic/c;tr+=np.sum(v*v,axis=1)/np.trace(covariance);count+=1
    return q/count,tr/count

def run():
    out=P/'run001';out.mkdir(parents=True,exist_ok=False)
    try:
        cfg=read_json(ROOT/'configs/mond_atlas_aperture_noise_v1.json');source=ROOT/cfg['input'];assert digest(source)==cfg['input_sha256']
        geo=ROOT/'work/gravity-first-principles/mond-atlas-native-covariance-001/run-001/block-geometry.csv';assert digest(geo)==read_json(geo.parent/'geometry-freeze.json')['block_geometry_sha256']
        deps=[source,geo,Path(__file__),P/'test_scale_channel.py',P/'PREFLIGHT.md',P/'DIAGNOSTIC_ADDENDUM.md',ROOT/'scripts/mond_atlas_native_covariance.py']
        write_json(out/'bindings.json',dict(previous_exposure=True,files={p.relative_to(ROOT).as_posix():digest(p) for p in deps}))
        t=subprocess.run([sys.executable,str(P/'test_scale_channel.py')],cwd=ROOT,capture_output=True,text=True);(out/'tests.log').write_text(t.stdout+t.stderr,encoding='utf-8')
        if t.returncode:raise RuntimeError('Pre-access tests failed')
        with geo.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        wr=[r for r in rows if r['region']=='training'];er=[r for r in rows if r['region']=='validation'];folds=np.array([int(r['fold']) for r in wr])
        with np.load(source) as z:west=z['training']
        assert west.shape==(29,24,24,42)
        bands,U,r2=geometry(24,24);cv=[];ranking={band:{a:[] for a in ALPHAS} for band in bands}
        for fold in sorted(set(folds)):
            mean,models=fit(west[folds!=fold]);coeff=transform(west[folds==fold],mean)
            for band,mask in bands.items():
                for alpha,C in models[band].items():
                    _,q,lp,_=gaussian_statistics(coeff[:,mask],C)
                    for j,v,l in zip(np.flatnonzero(folds==fold),q.mean(axis=1)/42,lp.mean(axis=1)/42):cv.append(dict(fold=int(fold),band=band,alpha=alpha,core=wr[j]['block_id'],q=float(v),logpdf=float(l)));ranking[band][alpha].append(float(l))
        selected={band:max(ALPHAS,key=lambda a:np.mean(ranking[band][a])) for band in bands};mean,models=fit(west)
        before=out/'models-before-east.json';write_json(before,dict(mean=mean.tolist(),selected=selected,band_counts={b:int(mask.sum()) for b,mask in bands.items()},matrices={b:{str(a):C.tolist() for a,C in v.items()} for b,v in models.items()},western_ranking={b:{str(a):float(np.mean(v)) for a,v in r.items()} for b,r in ranking.items()},east_read=False));beforehash=digest(before);write_csv(out/'western-cv.csv',cv)
        with np.load(source) as z:east=z['validation']
        assert east.shape==(27,24,24,42)
        coeff=transform(east,mean);compositions={'selected':selected,**{f'uniform_{a}':{b:a for b in bands} for a in ALPHAS}};results=[];aps=[];cores=[];diagnostics=[]
        for name,selection in compositions.items():
            covs={b:models[b][a] for b,a in selection.items()};q,lp,byband=score_coeff(coeff,bands,covs);qm=float(q.mean());results.append(dict(model=name,q=qm,logpdf=float(lp.mean()),q_pass=bool(.8<=qm<=1.2)))
            for j in range(27):cores.append(dict(model=name,core=er[j]['block_id'],q=float(q[j]),logpdf=float(lp[j])))
            for side in (1,2,4,8,12,24):
                aq,tr=aperture(east,mean,bands,U,covs,side);aps.append(dict(model=name,side=side,q=float(aq.mean()),trace_ratio=float(tr.mean()),q_pass=bool(.8<=aq.mean()<=1.2)))
            if name=='selected':
                for band,mask in bands.items():
                    C=covs[band];eig,V=np.linalg.eigh(C);spectral=coeff[:,mask]@V;power=spectral*spectral/eig[None,None,:]
                    diagnostics.append(dict(band=band,group='all',spatial_modes=int(mask.sum()),channel_modes=42,q=float(power.mean())))
                    if band!='DC':
                        local=r2[mask];middle=np.median(local)
                        for group,keep in [('frequency_lower',local<=middle),('frequency_upper',local>middle)]:
                            if keep.any():diagnostics.append(dict(band=band,group=group,spatial_modes=int(keep.sum()),channel_modes=42,q=float(power[:,keep].mean())))
                    for k,indices in enumerate(np.array_split(np.arange(42),4)):diagnostics.append(dict(band=band,group=f'channel_eigen_quartile_{k+1}',spatial_modes=int(mask.sum()),channel_modes=len(indices),q=float(power[:,:,indices].mean())))
        assert digest(before)==beforehash
        write_csv(out/'core-scores.csv',cores);write_csv(out/'aperture-scores.csv',aps);write_csv(out/'diagnostics.csv',diagnostics)
        write_json(out/'summary.json',dict(status='INDEPENDENT_DCT_MODE_BACKGROUND_MODEL',selected_shrinkages=selected,joint=results,selected_apertures=[r for r in aps if r['model']=='selected'],selected_diagnostics=diagnostics,observed_gravity_scores=0,source_likelihood_admitted=False,crossmode_covariance_modeled=False,new_raw_bytes=0));print(json.dumps(dict(selected=selected,joint=results,apertures=[r for r in aps if r['model']=='selected'],band_scores=[r for r in diagnostics if r['group']=='all']),indent=2))
    except Exception as exc:write_json(out/'failure.json',dict(error=repr(exc)));raise

if __name__=='__main__':run()
