"""Manual cosine-basis/source/moment/inverse replay; no evaluator import."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,json,hashlib
from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree
P=Path(__file__).resolve().parent;R=P.parents[2];out=P/'run001'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
rows=lambda p:list(csv.DictReader(p.open(encoding='utf-8',newline='')))
summary=read(out/'summary.json');support=read(P/'support001/summary.json')['checks'][0]
raw=R/'work/private/mond-atlas-noise-transfer-001/cores.npz'
assert hashlib.sha256(raw.read_bytes()).hexdigest()==summary['private_sha256']
with np.load(raw) as z:west=z['training'];east=z['validation']
geometry=rows(P/'support001/NGC3198-cores.csv')
cube=R/support['cube_path'];moment=R/support['moment_path'];m=fits.getdata(moment).squeeze();h=fits.getheader(moment)
tree=cKDTree(np.argwhere((m>0)|~np.isfinite(m))*abs(h['CDELT1'])*3600)
covered=np.zeros(m.shape,bool)
with fits.open(cube,memmap=True) as hdus:
    counts={'training':0,'validation':0}
    for row in geometry:
        y0,y1,x0,x1=[int(row[k]) for k in ('y0','y1','x0','x1')];region=row['region'];stored=(west if region=='training' else east)[counts[region]];counts[region]+=1
        actual=hdus[0].data.squeeze()[:,y0:y1,x0:x1].astype(float).transpose(1,2,0)*1000
        assert np.array_equal(actual,stored)
        assert not covered[y0:y1,x0:x1].any();covered[y0:y1,x0:x1]=True
        yy,xx=np.mgrid[y0:y1,x0:x1];angular=abs(h['CDELT1'])*3600
        assert tree.query(np.column_stack((yy.ravel(),xx.ravel()))*angular)[0].min()>120
        e=(xx+1-h['CRPIX1'])*h['CDELT1']*3600;n=(yy+1-h['CRPIX2'])*h['CDELT2']*3600
        assert np.all((np.hypot(e,n)>=550)&(np.hypot(e,n)<=680))
        assert np.all(e < -90 if region=='training' else e >90)
        assert min(x0,y0)>=42 and x1<=m.shape[1]-42 and y1<=m.shape[0]-42
idx=np.arange(24);T=np.sqrt(2/24)*np.cos(np.pi*np.outer(idx,idx+.5)/24);T[0]/=np.sqrt(2);U=np.kron(T.T,T.T)
mean=west.mean(axis=(0,1,2));wc=np.einsum('ps,npc->nsc',U,(west-mean).reshape(len(west),576,72));ec=np.einsum('ps,npc->nsc',U,(east-mean).reshape(len(east),576,72))
r2=np.array([y*y+x*x for y in idx for x in idx]);bands={'DC':r2==0,'low':(r2>0)&(r2<=9),'middle':(r2>9)&(r2<=64),'high':r2>64}
saved=read(out/'model-before-east.json');errors=[float(np.max(abs(mean-np.array(saved['mean']))))];models={};q=np.zeros((len(east),576));lp=np.zeros_like(q);diag=[]
for b,mask in bands.items():
    power=np.mean(wc[:,mask]**2,axis=(0,2));floor=max(1e-300,1e-12*np.median(power[power>0]));power=np.maximum(power,floor);power/=power.mean()
    v=wc[:,mask]/np.sqrt(power)[None,:,None];v=v.reshape(-1,72);S=v.T@v/len(v);d=np.diag(S);S+=np.diag(np.maximum(d,max(1e-12,1e-8*np.median(d)))-d)
    a={'DC':1,'low':.6,'middle':.3,'high':.1}[b];C=(1-a)*S+a*np.diag(np.diag(S))
    if b=='DC':d=np.diag(C);C=np.diag(21/19*(d+d.mean())/2)
    old=saved['models'][b];errors.extend([float(np.max(abs(power-old['power']))),float(np.max(abs(C-old['C'])))])
    models[b]=(power,C);v=ec[:,mask]/np.sqrt(power)[None,:,None];inv=np.linalg.inv(C);q[:,mask]=np.einsum('nmc,cd,nmd->nm',v,inv,v)
    lp[:,mask]=-.5*(q[:,mask]+np.linalg.slogdet(C)[1]+72*np.log(2*np.pi)+72*np.log(power))
    eigen,V=np.linalg.eigh(C);ep=(v@V)**2/eigen;diag.append((b,'all',ep.mean()))
    for i,ci in enumerate(np.array_split(np.arange(72),4)):diag.append((b,f'eigen_quartile_{i+1}',ep[:,:,ci].mean()))
    if b!='DC':
        low=r2[mask]<=np.median(r2[mask]);diag.extend([(b,'frequency_lower',ep[:,low].mean()),(b,'frequency_upper',ep[:,~low].mean())])
for i,row in enumerate(rows(out/'cores.csv')):errors.extend([abs(q[i].mean()/72-float(row['q'])),abs(lp[i].mean()/72-float(row['logpdf']))])
for row in rows(out/'modes.csv'):errors.append(abs(q[:,int(row['mode'])].mean()/72-float(row['q'])))
for row in rows(out/'diagnostics.csv'):errors.append(abs(next(v for b,g,v in diag if b==row['band'] and g==row['group'])-float(row['q'])))
for row in rows(out/'channels.csv'):i=int(row['channel']);errors.append(abs(np.mean(ec[:,0,i]**2)/models['DC'][1][i,i]-float(row['q'])))
for row in rows(out/'apertures.csv'):
    side=int(row['side']);qs=[];ts=[]
    for y in range(0,24,side):
        for x in range(0,24,side):
            A=np.zeros((24,24));A[y:y+side,x:x+side]=1/side**2;w=(A.ravel()@U)**2
            C=sum(np.dot(w[mask],models[b][0])*models[b][1] for b,mask in bands.items());v=(east[:,y:y+side,x:x+side]-mean).mean(axis=(1,2))
            qs.append(np.einsum('nc,cd,nd->n',v,np.linalg.inv(C),v).mean()/72);ts.append(np.sum(v*v,axis=1).mean()/np.trace(C))
    errors.extend([abs(np.mean(qs)-float(row['q'])),abs(np.mean(ts)-float(row['trace_ratio']))])
assert max(errors)<1e-9
result=dict(status='PASS_INDEPENDENT_REPLAY',source_cores_readback=38,manual_cosine_basis=True,western_covariances_refit=4,core_scores=18,mode_scores=576,diagnostics=26,channels=72,apertures=6,maximum_absolute_error=max(errors),observed_gravity_arrays_read=0)
(P/'independent-review').mkdir(exist_ok=True);(P/'independent-review/receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
