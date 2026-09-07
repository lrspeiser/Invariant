"""No-selection recipe evaluation on frozen NGC3198 external cores."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,csv,time,json,subprocess
from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.linalg import solve_triangular
from mond_atlas_noise_transfer import ROOT,P,PRIVATE,mp,regularized_covariance,extract_background,read_json,write_json,write_csv,digest

def fit(west):
    mean=west.mean(axis=(0,1,2));coef=mp.transform(west,mean);bands,_,_=mp.geometry(*west.shape[1:3]);models={}
    for band,mask in bands.items():
        raw=np.mean(coef[:,mask]**2,axis=(0,2));positive=raw[raw>0]
        if not len(positive):raise ValueError('zero band')
        floor=max(1e-300,1e-12*np.median(positive));p=np.maximum(raw,floor);p/=p.mean()
        C=regularized_covariance((coef[:,mask]/np.sqrt(p)[None,:,None]).reshape(-1,west.shape[-1]),dict(kind='full',shrinkage={'DC':1.,'low':.6,'middle':.3,'high':.1}[band]))
        if band=='DC':
            diagonal=np.diag(C);C=np.diag((len(west)+1)/(len(west)-1)*(.5*diagonal+.5*diagonal.mean()))
        models[band]=dict(power=p,C=C,floor_hits=int((raw<floor).sum()))
    return mean,models

def controls():
    rng=np.random.default_rng(4029)
    for c in [7,72]:
        data=rng.normal(size=(8,4,4,c));mean,models=fit(data);bands,U,_=mp.geometry(4,4);coef=mp.transform(data,mean)
        assert np.allclose(np.sum(coef**2),np.sum((data-mean)**2),rtol=1e-13)
        m2,models2=fit(data*3)
        for band in bands:
            assert np.allclose(models2[band]['C'],9*models[band]['C'],rtol=1e-12,atol=1e-12)
            np.linalg.cholesky(models[band]['C'])
        if c==7:
            # Full dense covariance with spatial-major/channel-minor vectorization.
            dense=sum(np.kron((U[:,mask]*models[b]['power'])@U[:,mask].T,models[b]['C']) for b,mask in bands.items())
            v=(data[0]-mean).ravel();direct=v@np.linalg.solve(dense,v)
            modal=sum(mp.band_score(coef[:1,mask],models[b]['power'],models[b]['C'])[0].sum() for b,mask in bands.items())
            assert abs(direct-modal)<1e-9
            A=np.ones(16)/16;operator=np.kron(A[None,:],np.eye(c));ap=operator@dense@operator.T
            expected=sum(np.sum((A@U[:,mask])**2*models[b]['power'])*models[b]['C'] for b,mask in bands.items())
            assert np.allclose(ap,expected,atol=1e-12)
    return dict(status='PASS',channels=[7,72],dense_inverse=True,parseval=True,aperture=True,units=True,rank_deficient_regularization=True)

def run():
    t=time.monotonic();out=P/'run001';out.mkdir(exist_ok=False);PRIVATE.mkdir(parents=True,exist_ok=True)
    try:
        write_json(out/'controls.json',controls())
        record=read_json(P/'support001/summary.json')['checks'][-1];assert record['eligible']
        cube=ROOT/record['cube_path'];assert digest(cube)==record['cube_sha256_expected']
        with (P/f"support001/{record['name']}-cores.csv").open(newline='',encoding='utf-8') as f:rows=[{k:(v if k in ['block_id','region'] else int(v)) for k,v in r.items()} for r in csv.DictReader(f)]
        wr=[r for r in rows if r['region']=='training'];er=[r for r in rows if r['region']=='validation']
        deps=[Path(__file__),P/'PREFLIGHT.md',P/'support001/summary.json',P/f"support001/{record['name']}-cores.csv",ROOT/'scripts/mond_atlas_noise_transfer.py',ROOT/'scripts/mond_atlas_noise_mode_power.py',ROOT/'scripts/mond_atlas_noise_scale_channel.py',ROOT/'scripts/mond_atlas_native_covariance.py',ROOT/'docs/OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md']
        write_json(out/'bindings.json',dict(files={p.relative_to(ROOT).as_posix():digest(p) for p in deps},cube_sha256=record['cube_sha256_expected'],moment_sha256=record['moment_sha256'],paper='https://arxiv.org/abs/0810.2125'))
        # memmap maps the file; only explicitly sliced western cores are materialized.
        with fits.open(cube,memmap=True) as h:west=extract_background(h[0].data.squeeze(),wr)
        assert west.shape==(len(wr),24,24,record['channels'])
        mean,models=fit(west);before=out/'model-before-east.json'
        write_json(before,dict(mean=mean.tolist(),models={b:{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in m.items()} for b,m in models.items()},east_read=False,selection=False));freeze=digest(before)
        with fits.open(cube,memmap=True) as h:east=extract_background(h[0].data.squeeze(),er)
        np.savez_compressed(PRIVATE/'cores.npz',training=west,validation=east)
        assert (PRIVATE/'cores.npz').stat().st_size<100*1024**2
        n,_,_,c=east.shape;bands,U,r2=mp.geometry(24,24);coef=mp.transform(east,mean);qtotal=np.zeros(n);lptotal=np.zeros(n);modes=[];diagnostics=[];white=np.empty_like(coef)
        for b,mask in bands.items():
            m=models[b];q,lp=mp.band_score(coef[:,mask],m['power'],m['C']);qtotal+=q.sum(axis=1);lptotal+=lp.sum(axis=1)
            normalized=coef[:,mask]/np.sqrt(m['power'])[None,:,None];L=np.linalg.cholesky(m['C']);white[:,mask]=solve_triangular(L,normalized.reshape(-1,c).T,lower=True).T.reshape(normalized.shape)
            eig,V=np.linalg.eigh(m['C']);ep=(normalized@V)**2/eig
            diagnostics.append(dict(band=b,group='all',q=float(ep.mean())))
            for k,indices in enumerate(np.array_split(np.arange(c),4)):diagnostics.append(dict(band=b,group=f'eigen_quartile_{k+1}',q=float(ep[:,:,indices].mean())))
            if b!='DC':
                low=r2[mask]<=np.median(r2[mask])
                for label,keep in [('frequency_lower',low),('frequency_upper',~low)]:diagnostics.append(dict(band=b,group=label,q=float(ep[:,keep].mean())))
            for idx,v in zip(np.flatnonzero(mask),q.mean(axis=0)/c):modes.append(dict(mode=int(idx),band=b,q=float(v),descriptive_flag=bool(.8<=v<=1.2)))
        cores=[dict(core=r['block_id'],q=float(qtotal[j]/(576*c)),logpdf=float(lptotal[j]/(576*c))) for j,r in enumerate(er)]
        aps=[]
        for side in (1,2,4,8,12,24):
            q,tr=mp.aperture(east,mean,bands,U,models,side);aps.append(dict(side=side,q=float(q.mean()),trace_ratio=float(tr.mean())))
        channel=[dict(channel=i,q=float(v)) for i,v in enumerate(np.mean(coef[:,0]**2,axis=0)/np.diag(models['DC']['C']))]
        pairs=[]
        for a in range(576):
            for b in [a+24,a+1 if a%24<23 else 576]:
                if b>=576:continue
                v=white[:,a].ravel();w=white[:,b].ravel();pairs.append(dict(mode_a=a,mode_b=b,product=float(v@w/np.sqrt((v@v)*(w@w)))))
        for file,data in [('cores.csv',cores),('modes.csv',modes),('diagnostics.csv',diagnostics),('apertures.csv',aps),('channels.csv',channel),('crossmode.csv',pairs)]:write_csv(out/file,data)
        assert digest(before)==freeze
        result=dict(status='FIXED_RECIPE_BACKGROUND_TRANSFER',galaxy=record['name'],western_cores=len(wr),eastern_cores=n,channels=c,joint_q=float(np.mean([r['q'] for r in cores])),joint_logpdf=float(np.mean([r['logpdf'] for r in cores])),apertures=aps,diagnostics=diagnostics,mode_flag_count=sum(r['descriptive_flag'] for r in modes),channel_flag_count=sum(.8<=r['q']<=1.2 for r in channel),crossmode_abs_gt_point1=sum(abs(r['product'])>.1 for r in pairs),new_raw_bytes=0,private_sha256=digest(PRIVATE/'cores.npz'),source_likelihood_admitted=False,observed_gravity_scores=0,elapsed_seconds=time.monotonic()-t)
        write_json(out/'summary.json',result);print(json.dumps(result,indent=2));assert time.monotonic()-t<300
    except Exception as exc:write_json(out/'failure.json',dict(error=repr(exc)));raise
if __name__=='__main__':run()
