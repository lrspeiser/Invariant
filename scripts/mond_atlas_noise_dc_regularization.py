"""Bounded DC variance shrinkage with finite training-mean correction."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,csv,json,subprocess,time
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
import mond_atlas_noise_mode_power as mp
from mond_atlas_native_covariance import gaussian_statistics
P=ROOT/'work/gravity-first-principles/mond-atlas-noise-dc-regularization-001'
FIXED={'DC':'r0_a1.0','low':'r0_a0.6','middle':'r0_a0.3','high':'r0_a0.1'}

def candidates(diagonal,low_diagonal,n):
    S=np.asarray(diagonal,float);low=np.asarray(low_diagonal,float)
    if n<2 or np.any(S<=0) or np.any(low<=0):raise ValueError('Positive variances and multiple fitting cores required')
    correction=(n+1)/(n-1);result={'baseline':dict(variance=S,correction=1.,weight=0.,target='none')}
    for name,T in [('constant',np.full_like(S,S.mean())),('low_band',low*S.sum()/low.sum())]:
        for w in (0.,.25,.5,.75,1.):result[f'{name}_w{w}']=dict(variance=correction*((1-w)*S+w*T),correction=correction,weight=w,target=name)
    return result

def fit(data):
    mean,allmodels,_=mp.fit(data);selected={b:allmodels[b][name] for b,name in FIXED.items() if b in allmodels};dc=candidates(np.diag(selected['DC']['C']),np.diag(selected['low']['C']),len(data))
    return mean,selected,dc

def run():
    started=time.monotonic();out=P/'run001';out.mkdir(parents=True,exist_ok=False)
    try:
        cfg=read_json(ROOT/'configs/mond_atlas_aperture_noise_v1.json');source=ROOT/cfg['input'];assert digest(source)==cfg['input_sha256'];geo=ROOT/'work/gravity-first-principles/mond-atlas-native-covariance-001/run-001/block-geometry.csv';assert digest(geo)==read_json(geo.parent/'geometry-freeze.json')['block_geometry_sha256']
        deps=[source,geo,Path(__file__),P/'PREFLIGHT.md',P/'test_dc.py',ROOT/'scripts/mond_atlas_noise_mode_power.py',ROOT/'scripts/mond_atlas_noise_scale_channel.py',ROOT/'scripts/mond_atlas_native_covariance.py'];write_json(out/'bindings.json',dict(previous_exposure=True,files={p.relative_to(ROOT).as_posix():digest(p) for p in deps}))
        t=subprocess.run([sys.executable,str(P/'test_dc.py')],cwd=ROOT,capture_output=True,text=True);(out/'tests.log').write_text(t.stdout+t.stderr,encoding='utf-8')
        if t.returncode:raise RuntimeError('Pre-access tests failed')
        with geo.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        wr=[r for r in rows if r['region']=='training'];er=[r for r in rows if r['region']=='validation'];folds=np.array([int(r['fold']) for r in wr])
        with np.load(source) as z:west=z['training']
        assert west.shape==(29,24,24,42);bands,U,r2=mp.geometry(24,24);cv=[];ranking={};fitted_folds=[]
        for fold in sorted(set(folds)):
            mean,models,dc=fit(west[folds!=fold]);coeff=mp.transform(west[folds==fold],mean)[:,0]
            fitted_folds.append(dict(fold=int(fold),candidates={n:{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in d.items()} for n,d in dc.items()}))
            for name,d in dc.items():
                _,q,lp,_=gaussian_statistics(coeff,np.diag(d['variance']));ranking.setdefault(name,[])
                for j,v,l in zip(np.flatnonzero(folds==fold),q/42,lp/42):cv.append(dict(fold=int(fold),candidate=name,core=wr[j]['block_id'],q=float(v),logpdf=float(l)));ranking[name].append(float(l))
        chosen=max(ranking,key=lambda n:np.mean(ranking[n]));mean,models,dc=fit(west);before=out/'models-before-east.json'
        write_json(before,dict(mean=mean.tolist(),selected=chosen,dc_candidates={n:{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in d.items()} for n,d in dc.items()},base_models={b:{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in m.items()} for b,m in models.items()},ranking={n:float(np.mean(v)) for n,v in ranking.items()},fixed_original_channel_order=np.argsort(dc['baseline']['variance']).tolist(),east_read=False));frozenhash=digest(before);write_csv(out/'western-cv.csv',cv);write_json(out/'western-fit-candidates.json',fitted_folds)
        if time.monotonic()-started>300:raise TimeoutError('Five-minute execution cap')
        with np.load(source) as z:east=z['validation']
        assert east.shape==(27,24,24,42);coeff=mp.transform(east,mean);dc_rows=[];dc_summary=[]
        for name,d in dc.items():
            _,q,lp,_=gaussian_statistics(coeff[:,0],np.diag(d['variance']));dc_summary.append(dict(candidate=name,selected=name==chosen,q=float(q.mean()/42),logpdf=float(lp.mean()/42)))
            for j in range(27):dc_rows.append(dict(candidate=name,core=er[j]['block_id'],q=float(q[j]/42),logpdf=float(lp[j]/42)))
        results=[];apertures=[];diagnostics=[];core_rows=[];mode_rows=[];pair_rows=[];channel_rows=[]
        for label,name in [('baseline','baseline'),('selected',chosen)]:
            current={b:dict(m) for b,m in models.items()};current['DC']['C']=np.diag(dc[name]['variance']);totalq=np.zeros(27);totallp=np.zeros(27);white=np.zeros_like(coeff)
            for band,mask in bands.items():
                m=current[band];q,lp=mp.band_score(coeff[:,mask],m['power'],m['C']);totalq+=q.sum(axis=1);totallp+=lp.sum(axis=1)
                if label!='selected':continue
                normalized=coeff[:,mask]/np.sqrt(m['power'])[None,:,None];L=np.linalg.cholesky(m['C']);white[:,mask]=solve_triangular(L,normalized.reshape(-1,42).T,lower=True).T.reshape(normalized.shape);eig,V=np.linalg.eigh(m['C']);ep=(normalized@V)**2/eig
                diagnostics.append(dict(band=band,group='all',q=float(ep.mean())))
                for mode,v in zip(np.flatnonzero(mask),q.mean(axis=0)/42):mode_rows.append(dict(mode=int(mode),band=band,q=float(v),q_pass=bool(.8<=v<=1.2)))
                if band!='DC':
                    local=r2[mask];mid=np.median(local)
                    for group,keep in [('frequency_lower',local<=mid),('frequency_upper',local>mid)]:diagnostics.append(dict(band=band,group=group,q=float(ep[:,keep].mean())))
                for k,indices in enumerate(np.array_split(np.arange(42),4)):diagnostics.append(dict(band=band,group=f'channel_eigen_quartile_{k+1}',q=float(ep[:,:,indices].mean())))
            totalq/=576*42;totallp/=576*42;results.append(dict(model=label,candidate=name,joint_q=float(totalq.mean()),joint_logpdf=float(totallp.mean())))
            for j in range(27):core_rows.append(dict(model=label,core=er[j]['block_id'],q=float(totalq[j]),logpdf=float(totallp[j])))
            for side in (1,2,4,8,12,24):
                q,tr=mp.aperture(east,mean,bands,U,current,side);apertures.append(dict(model=label,side=side,q=float(q.mean()),trace_ratio=float(tr.mean())))
            if label=='selected':
                channel_q=np.mean(coeff[:,0]**2,axis=0)/dc[name]['variance']
                for k,v in enumerate(channel_q):channel_rows.append(dict(channel=k,q=float(v),q_pass=bool(.8<=v<=1.2)))
                order=np.argsort(dc['baseline']['variance'])
                for k,indices in enumerate(np.array_split(order,4)):diagnostics.append(dict(band='DC',group=f'fixed_original_channel_quartile_{k+1}',q=float(channel_q[indices].mean())))
                for y in range(24):
                    for x in range(24):
                        a=y*24+x
                        for yy,xx in [(y+1,x),(y,x+1)]:
                            if yy>=24 or xx>=24:continue
                            b=yy*24+xx;v=white[:,a].ravel();w=white[:,b].ravel();pair_rows.append(dict(mode_a=a,mode_b=b,normalized_product=float(v@w/np.sqrt((v@v)*(w@w)))))
        assert digest(before)==frozenhash
        for filename,data in [('dc-candidate-scores.csv',dc_rows),('joint-core-scores.csv',core_rows),('apertures.csv',apertures),('diagnostics.csv',diagnostics),('mode-scores.csv',mode_rows),('channel-scores.csv',channel_rows),('cross-mode-products.csv',pair_rows)]:write_csv(out/filename,data)
        summary=dict(status='DC_PREDICTIVE_VARIANCE_DEVELOPMENT',chosen=chosen,dc_candidate_summary=dc_summary,joint=results,selected_apertures=[r for r in apertures if r['model']=='selected'],dc_diagnostics=[r for r in diagnostics if r['band']=='DC'],mode_pass_count=sum(r['q_pass'] for r in mode_rows),channel_pass_count=sum(r['q_pass'] for r in channel_rows),crossmode_abs_above_point1=sum(abs(r['normalized_product'])>.1 for r in pair_rows),observed_gravity_scores=0,source_likelihood_admitted=False,elapsed_seconds=time.monotonic()-started);write_json(out/'summary.json',summary)
        size=sum(p.stat().st_size for p in P.rglob('*') if p.is_file());assert size<20*1024**2
        print(json.dumps(summary,indent=2))
    except Exception as exc:write_json(out/'failure.json',dict(error=repr(exc)));raise

if __name__=='__main__':run()
