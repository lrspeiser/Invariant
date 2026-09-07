"""Positive per-DCT-mode power times band-specific channel covariance."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,csv,json,subprocess
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
from mond_atlas_native_covariance import regularized_covariance,gaussian_statistics
from mond_atlas_noise_scale_channel import geometry,transform
P=ROOT/'work/gravity-first-principles/mond-atlas-noise-mode-power-001'
ALPHAS=[.1,.3,.6,1.]

def pool_power(raw,coordinates,radius):
    raw=np.asarray(raw,float);distance=np.sum((coordinates[:,None]-coordinates[None,:])**2,axis=2);weights=distance<=radius**2
    pooled=weights@raw/weights.sum(axis=1);positive=raw[raw>0]
    if not len(positive):raise ValueError('Entire band zero; no silent exclusion')
    floor=max(1e-300,1e-12*float(np.median(positive)));hits=int(np.sum(pooled<floor));pooled=np.maximum(pooled,floor)
    return pooled/pooled.mean(),dict(floor=floor,floor_hits=hits,raw_min=float(raw.min()),raw_max=float(raw.max()),pooled_min=float(pooled.min()),pooled_max=float(pooled.max()))

def fit(data):
    mean=data.mean(axis=(0,1,2));coeff=transform(data,mean);h,w=data.shape[1:3];bands,_,_=geometry(h,w);coords=np.array([(y,x) for y in range(h) for x in range(w)]);models={};debug={}
    for band,mask in bands.items():
        raw=np.mean(coeff[:,mask]**2,axis=(0,2));models[band]={};debug[band]=dict(raw_power=raw.tolist(),mode_indices=np.flatnonzero(mask).tolist())
        for radius in (0,1,2):
            power,info=pool_power(raw,coords[mask],radius);normalized=coeff[:,mask]/np.sqrt(power)[None,:,None]
            for alpha in ALPHAS:
                C=regularized_covariance(normalized.reshape(-1,data.shape[-1]),dict(kind='full',shrinkage=alpha));models[band][f'r{radius}_a{alpha}']=dict(radius=radius,alpha=alpha,power=power,C=C,**info)
    return mean,models,debug

def band_score(coeff,power,C):
    _,q,lp,_=gaussian_statistics(coeff/np.sqrt(power)[None,:,None],C)
    lp=lp-.5*len(C)*np.log(power)[None,:]
    return q,lp

def aperture(data,mean,bands,U,models,side):
    b,h,w,c=data.shape;res=data-mean;q=np.zeros(b);tr=np.zeros(b);count=0
    for y in range(0,h,side):
        for x in range(0,w,side):
            A=np.zeros((h,w));A[y:y+side,x:x+side]=1/side**2;weights=(A.ravel()@U)**2
            C=sum(np.sum(weights[mask]*models[band]['power'])*models[band]['C'] for band,mask in bands.items());v=res[:,y:y+side,x:x+side].mean(axis=(1,2));_,a,_,_=gaussian_statistics(v,C);q+=a/c;tr+=np.sum(v*v,axis=1)/np.trace(C);count+=1
    return q/count,tr/count

def serial(models):return {b:{name:{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in m.items()} for name,m in candidates.items()} for b,candidates in models.items()}

def run():
    out=P/'run001';out.mkdir(parents=True,exist_ok=False)
    try:
        cfg=read_json(ROOT/'configs/mond_atlas_aperture_noise_v1.json');source=ROOT/cfg['input'];assert digest(source)==cfg['input_sha256'];geo=ROOT/'work/gravity-first-principles/mond-atlas-native-covariance-001/run-001/block-geometry.csv';assert digest(geo)==read_json(geo.parent/'geometry-freeze.json')['block_geometry_sha256']
        deps=[source,geo,P/'PREFLIGHT.md',P/'test_mode_power.py',Path(__file__),ROOT/'scripts/mond_atlas_noise_scale_channel.py',ROOT/'scripts/mond_atlas_native_covariance.py'];write_json(out/'bindings.json',dict(previous_exposure=True,files={p.relative_to(ROOT).as_posix():digest(p) for p in deps}))
        t=subprocess.run([sys.executable,str(P/'test_mode_power.py')],cwd=ROOT,capture_output=True,text=True);(out/'tests.log').write_text(t.stdout+t.stderr,encoding='utf-8')
        if t.returncode:raise RuntimeError('Pre-access tests failed')
        with geo.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        wr=[r for r in rows if r['region']=='training'];er=[r for r in rows if r['region']=='validation'];folds=np.array([int(r['fold']) for r in wr])
        with np.load(source) as z:west=z['training']
        assert west.shape==(29,24,24,42);bands,U,r2=geometry(24,24);cv=[];ranking={b:{} for b in bands};fold_debug=[]
        for fold in sorted(set(folds)):
            mean,models,debug=fit(west[folds!=fold]);coeff=transform(west[folds==fold],mean);fold_debug.append(dict(fold=int(fold),raw_power=debug,models=serial(models)))
            for band,mask in bands.items():
                for name,m in models[band].items():
                    q,lp=band_score(coeff[:,mask],m['power'],m['C']);ranking[band].setdefault(name,[])
                    for j,v,l in zip(np.flatnonzero(folds==fold),q.mean(axis=1)/42,lp.mean(axis=1)/42):cv.append(dict(fold=int(fold),band=band,candidate=name,core=wr[j]['block_id'],q=float(v),logpdf=float(l)));ranking[band][name].append(float(l))
        choice={b:max(r,key=lambda name:np.mean(r[name])) for b,r in ranking.items()};mean,models,debug=fit(west);selected={b:models[b][choice[b]] for b in bands};before=out/'models-before-east.json';write_json(before,dict(mean=mean.tolist(),choice=choice,models=serial(models),raw_power=debug,ranking={b:{n:float(np.mean(v)) for n,v in r.items()} for b,r in ranking.items()},east_read=False));modelhash=digest(before);write_json(out/'western-fit-models.json',fold_debug);write_csv(out/'western-cv.csv',cv)
        with np.load(source) as z:east=z['validation']
        assert east.shape==(27,24,24,42);coeff=transform(east,mean);totalq=np.zeros(27);totallp=np.zeros(27);mode_rows=[];diagnostics=[];whitened=np.zeros_like(coeff);range_rows=[]
        largest=max(max(d['raw_power']) for d in debug.values())
        for band,mask in bands.items():
            m=selected[band];power=m['power'];C=m['C'];q,lp=band_score(coeff[:,mask],power,C);totalq+=q.sum(axis=1);totallp+=lp.sum(axis=1);indices=np.flatnonzero(mask)
            for mode,v in zip(indices,q.mean(axis=0)/42):mode_rows.append(dict(band=band,mode=int(mode),ky=int(mode//24),kx=int(mode%24),q=float(v),q_pass=bool(.8<=v<=1.2)))
            normalized=coeff[:,mask]/np.sqrt(power)[None,:,None];L=np.linalg.cholesky(C);whitened[:,mask]=solve_triangular(L,normalized.reshape(-1,42).T,lower=True).T.reshape(normalized.shape)
            eig,V=np.linalg.eigh(C);ep=(normalized@V)**2/eig;diagnostics.append(dict(band=band,group='all',spatial_modes=int(mask.sum()),channel_modes=42,q=float(ep.mean())))
            if band!='DC':
                local=r2[mask];median=np.median(local)
                for group,keep in [('frequency_lower',local<=median),('frequency_upper',local>median)]:diagnostics.append(dict(band=band,group=group,spatial_modes=int(keep.sum()),channel_modes=42,q=float(ep[:,keep].mean())))
            for i,ci in enumerate(np.array_split(np.arange(42),4)):diagnostics.append(dict(band=band,group=f'channel_eigen_quartile_{i+1}',spatial_modes=int(mask.sum()),channel_modes=len(ci),q=float(ep[:,:,ci].mean())))
            range_rows.append(dict(band=band,selected=choice[band],floor_hits=m['floor_hits'],raw_min=m['raw_min'],raw_max=m['raw_max'],pooled_min=m['pooled_min'],pooled_max=m['pooled_max'],min_relative_to_epsilon_times_globalmax=m['raw_min']/(np.finfo(float).eps*largest)))
        pairs=[]
        for y in range(24):
            for x in range(24):
                a=y*24+x
                for yy,xx in [(y+1,x),(y,x+1)]:
                    if yy>=24 or xx>=24:continue
                    b=yy*24+xx;first=whitened[:,a].ravel();second=whitened[:,b].ravel();corr=float(first@second/np.sqrt((first@first)*(second@second)));pairs.append(dict(mode_a=a,mode_b=b,normalized_product=corr,abs_above_point1=abs(corr)>.1))
        aps=[]
        for side in (1,2,4,8,12,24):
            q,tr=aperture(east,mean,bands,U,selected,side);aps.append(dict(side=side,q=float(q.mean()),trace_ratio=float(tr.mean()),q_pass=bool(.8<=q.mean()<=1.2)))
        totalq/=576*42;totallp/=576*42;cores=[dict(core=er[j]['block_id'],q=float(totalq[j]),logpdf=float(totallp[j])) for j in range(27)]
        assert digest(before)==modelhash
        for filename,data in [('core-scores.csv',cores),('aperture-scores.csv',aps),('mode-scores.csv',mode_rows),('diagnostics.csv',diagnostics),('cross-mode-products.csv',pairs),('power-range.csv',range_rows)]:write_csv(out/filename,data)
        summary=dict(status='PER_MODE_POWER_BACKGROUND_ONLY',choices=choice,joint_q=float(totalq.mean()),joint_logpdf=float(totallp.mean()),joint_q_pass=bool(.8<=totalq.mean()<=1.2),apertures=aps,diagnostics=diagnostics,mode_q_pass_count=sum(r['q_pass'] for r in mode_rows),mode_count=576,crossmode_pairs=len(pairs),crossmode_abs_gt_point1=sum(r['abs_above_point1'] for r in pairs),crossmode_mean_abs=float(np.mean([abs(r['normalized_product']) for r in pairs])),power_ranges=range_rows,observed_gravity_scores=0,source_likelihood_admitted=False,new_raw_bytes=0);write_json(out/'summary.json',summary);print(json.dumps(summary,indent=2))
    except Exception as exc:write_json(out/'failure.json',dict(error=repr(exc)));raise

if __name__=='__main__':run()
