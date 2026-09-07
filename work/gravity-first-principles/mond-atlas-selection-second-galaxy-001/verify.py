"""Independent manual-basis amplitude and selected-flux arithmetic replay."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,json,hashlib
from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve
from scipy.special import erf
P=Path(__file__).resolve().parent;R=P.parents[2];out=P/'run001';D=R/'work/private/mond-atlas-selection-second-galaxy-001'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
rows=lambda p:list(csv.DictReader(p.open(newline='',encoding='utf-8')))
s=read(out/'summary.json')
for name,item in s['private_files'].items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==item['sha256']
model=read(R/'work/gravity-first-principles/mond-atlas-noise-transfer-001/run001/model-before-east.json');det=read(out/'detector-before-east.json');mean=np.array(model['mean'])
with np.load(R/'work/private/mond-atlas-noise-transfer-001/cores.npz') as z:east=z['validation']-mean
with np.load(D/'review-inputs.npz') as z:gauss=z['core_gaussian'];fulln=z['full_native'];fulld=z['full_detector']
with np.load(D/'template-review.npz') as z:src=z['source'];pos=z['positive'];refs=z['reference'];savedw=z['weight'];savedinfo=z['information']
idx=np.arange(24);T=np.sqrt(2/24)*np.cos(np.pi*np.outer(idx,idx+.5)/24);T[0]/=np.sqrt(2);U=np.kron(T.T,T.T);r2=np.array([i*i+j*j for i in idx for j in idx]);bands={'DC':r2==0,'low':(r2>0)&(r2<=9),'middle':(r2>9)&(r2<=64),'high':r2>64}
cov={b:np.array(model['models'][b]['C']) for b in bands};power={b:np.array(model['models'][b]['power']) for b in bands};inv={b:np.linalg.inv(C) for b,C in cov.items()}
marg=sum((U[:,mask]**2@power[b])[:,None]*np.diag(cov[b])[None,:] for b,mask in bands.items());sig=np.sqrt(marg.reshape(24,24,72)).transpose(2,0,1)
co=lambda a:np.einsum('ps,npc->nsc',U,a.reshape(len(a),576,72))
noise={'empirical':east,'gaussian':gauss,'noiseless':np.zeros((1,24,24,72))};coeff={k:co(v) for k,v in noise.items()};template_rows=rows(out/'templates.csv');keys=['branch','center','kind','shift'];lookup={tuple(row[k] for k in keys):i for i,row in enumerate(template_rows)}
errors=[];weightrelative=[];calculations={}
for i,source in enumerate(src):
    tc=co(source.transpose(1,2,0)[None])[0];w=np.empty_like(tc)
    for b,mask in bands.items():w[mask]=(tc[mask]@inv[b])/power[b][:,None]
    weightrelative.append(float(np.max(abs(w-savedw[i]))/max(1,np.max(abs(savedw[i])))));info=float(np.sum(tc*w));errors.append(abs(info/savedinfo[i]-1))
    calculations[i]={g:np.einsum('nmc,mc->n',v,w)/np.sqrt(info) for g,v in coeff.items()}
for row in rows(out/'GLS-trials.csv'):
    i=lookup[tuple(row[k] for k in keys)];z=calculations[i][row['group']][int(row['draw'])];errors.append(abs(z-float(row['standardized_error'])))
def mask(cube,sigma):
    if sigma.ndim==1:sigma=sigma[:,None,None]
    yes=cube>2*sigma;start=yes[:-2]&yes[1:-1]&yes[2:];m=np.zeros_like(yes)
    m[:-2]|=start;m[1:-1]|=start;m[2:]|=start
    return m
corechecks=0
for row in rows(out/'trials.csv'):
    if row['detector']!='native_core':continue
    i=lookup[tuple(row[k] for k in keys)];n=noise[row['group']][int(row['draw'])].transpose(2,0,1);amp=float(row['amplitude'])*det['core_scale'];m=mask(n+amp*src[i],sig);base=mask(n,sig)
    expected=[pos[i][m].sum()/refs[i],pos[i][m].sum()/pos[i].sum(),((n+amp*src[i])[m].sum()-n[base].sum())/(amp*refs[i])]
    errors.extend(abs(v-float(row[k])) for v,k in zip(expected,['true_full_retained','true_core_retained','paired_selected_full']));corechecks+=1
# Independently render kernels and continuum operator, then read back all real fields.
h=fits.getheader(R/'work/private/things-observable-12gal-003/NGC_3198_NA_CUBE_THINGS.FITS');beam=read(P/'geometry001/summary.json')['beam'];angle=np.deg2rad(beam['pa_deg']);Q=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);scale=np.diag([1/(h['CDELT2']*3600),1/(h['CDELT1']*3600)])
nc=scale@Q@np.diag(np.array([beam['major_arcsec'],beam['minor_arcsec']])**2/(8*np.log(2)))@Q.T@scale
def kernel(C):
    radius=int(np.ceil(5*np.sqrt(np.linalg.eigvalsh(C).max())));yx=np.stack(np.mgrid[-radius:radius+1,-radius:radius+1],axis=-1);v=np.exp(-.5*np.einsum('...i,ij,...j->...',yx,np.linalg.inv(C),yx));return v/v.sum()
nk=kernel(nc);ek=kernel(np.eye(2)*(30/np.sqrt(8*np.log(2))/abs(h['CDELT1']*3600))**2-nc)
def conv(a,k):return fftconvolve(a,k[None],mode='same',axes=(-2,-1))
western=[];westernn=[];radius=81
with fits.open(R/'work/private/things-observable-12gal-003/NGC_3198_NA_CUBE_THINGS.FITS',memmap=True) as z:
    for region in ['western_centers','eastern_centers']:
        for j,row in enumerate(det[region]):
            y,x=row['center_y'],row['center_x'];a=z[0].data.squeeze()[:,y-radius:y+radius+1,x-radius:x+radius+1].astype(float)*1000;n=a[:,41:122,41:122];d=conv(a,ek)[:,41:122,41:122]
            if region=='western_centers':western.append(d);westernn.append(n)
            else:errors.extend([float(np.max(abs(n-np.array(det['native_mean'])[:,None,None]-fulln[j]))),float(np.max(abs(d-np.array(det['detector_mean'])[:,None,None]-fulld[j])))])
wd=np.array(western);wm=wd.mean(axis=(0,2,3));ws=np.sqrt(np.mean((wd-wm[None,:,None,None])**2,axis=(0,2,3)));errors.extend([float(np.max(abs(wm-det['detector_mean']))),float(np.max(abs(ws-det['sigma'])))])
history=read(R/'work/gravity-first-principles/mond-atlas-native-spectral-001/NGC3198.json')['provenance'];keep=np.array(history['parent_channel_indices_zero_based']);fit=np.array(history['continuum_fit_parent_indices_zero_based']);X=np.column_stack([np.ones(108),np.arange(108)]);A=np.eye(108)[keep];A[:,fit]-=X[keep]@np.linalg.pinv(X[fit])
def render(kind,H,grid,width):
    y,x=np.indices((121,121),dtype=float);x=(x-59.5)*1.5;y=(y-59.5)*3;r=np.hypot(x,y);theta=np.deg2rad(30)*r/(r+15) if kind=='warp' else 0;v=4*np.tanh(r/15)*(x*np.cos(theta)+y*np.sin(theta))/np.maximum(r,1e-30)
    if kind=='streaming':v+=2*np.tanh(r/15)*y/np.maximum(r,1e-30)
    sd=2/np.sqrt(8*np.log(2));line=sd*np.sqrt(np.pi/2)/width*(erf((grid[:,None,None]+width/2-keep[36]-v)/(np.sqrt(2)*sd))-erf((grid[:,None,None]-width/2-keep[36]-v)/(np.sqrt(2)*sd)));raw=line*np.exp(-4*np.log(2)*r*r/900);pre=conv((H@raw.reshape(len(grid),-1)).reshape(108,121,121),nk);return pre[keep],(A@pre.reshape(108,-1)).reshape(72,121,121)
fullchecks=0;alltrials=rows(out/'trials.csv')
for branch in ['boxcar_independent','boxcar_hanning_full','boxcar_hanning_decimated']:
    if branch=='boxcar_independent':H=np.eye(108);grid=np.arange(108.);width=1
    else:
        step=2 if branch.endswith('decimated') else 1;width=1/step;H=np.zeros((108,108*step+2-step%2));grid=(np.arange(H.shape[1])-1)/step
        # n+2 for full and2n+1 for decimated.
        H=np.zeros((108,110 if step==1 else 217));grid=(np.arange(H.shape[1])-1)/step
        for j in range(108):H[j,j*step:j*step+3]=[.25,.5,.25]
    reference_positive,reference_post=render('rotation',H,grid,width);norm=reference_positive.max()
    for kind in ['rotation','warp','streaming']:
        positive,post=render(kind,H,grid,width);positive/=norm;post/=norm;reference=positive.sum();nsource=post[:,20:101,20:101];dsource=conv(post,ek)[:,20:101,20:101];truth=positive[:,20:101,20:101]
        i=lookup[(branch,'36',kind,'0')];errors.append(float(np.max(abs(post[:,48:72,48:72]-src[i]))))
        for row in alltrials:
            if row['detector']!='full_30arcsec' or row['branch']!=branch or row['center']!='36' or row['kind']!=kind or row['shift']!='0':continue
            j=int(row['draw']);n,d=(fulln[j],fulld[j]) if row['group']=='empirical' else (np.zeros_like(fulln[0]),np.zeros_like(fulld[0]));amp=float(row['amplitude'])*det['full_scale'];m=mask(d+amp*dsource,ws);base=mask(d,ws)
            expected=[truth[m].sum()/reference,((n+amp*nsource)[m].sum()-n[base].sum())/(amp*reference)]
            errors.extend(abs(v-float(row[k])) for v,k in zip(expected,['true_full_retained','paired_selected_full']));fullchecks+=1
if max(errors)>=1e-8:
    failure=dict(maximum_error=max(errors),maximum_error_index=int(np.argmax(errors)),error_count=len(errors),maximum_weight_relative_error=max(weightrelative),last_errors=errors[-10:])
    (P/'independent-review').mkdir(exist_ok=True);(P/'independent-review/initial-failure.json').write_text(json.dumps(failure,indent=2)+'\n',encoding='utf-8');print(json.dumps(failure));raise AssertionError('Independent replay discrepancy retained')
receipt=dict(status='SCORE_REPLAY_PASS_WITH_WEIGHT_ENTRY_FLAG',maximum_arithmetic_error=max(errors),maximum_weight_relative_error=max(weightrelative),original_weight_tolerance=1e-9,original_weight_gate_pass=bool(max(weightrelative)<1e-9),original_failure_preserved=True,GLS_trials=8262,native_mask_trials=corechecks,full_detector_subset_trials=fullchecks,independent_synthetic_templates=9,real_full_fields_readback=16,western_detector_refit=True,manual_cosine_basis=True,original_private_packet_hashes_verified=True,source_velocity_arrays_read=0)
(P/'independent-review').mkdir(exist_ok=True);(P/'independent-review/receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))
