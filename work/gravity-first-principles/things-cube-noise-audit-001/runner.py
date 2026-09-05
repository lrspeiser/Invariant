"""Audit standard-cube noise and reproduce the documented 30 arcsec detection rule."""
import os
os.environ['OMP_NUM_THREADS']='1'
import json
import re
import time
from pathlib import Path
import numpy as np
import cupy as cp
from cupyx.scipy.signal import fftconvolve
from astropy.io import fits
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'work/gravity-first-principles/things-cube-noise-audit-001';D.mkdir(exist_ok=False)
(D/'runner.py').write_bytes(Path(__file__).read_bytes())
C=ROOT/'work/gravity-first-principles/things-cube-acquisition-001/receipt.json'
M=ROOT/'work/gravity-first-principles/things-observable-acquisition-003/receipt.json'
cubes=json.loads(C.read_text());maps=json.loads(M.read_text());assert cubes['status']==maps['status']=='COMPLETE'
assets={(m['name'],m['resolution'],m['moment']):m for m in maps['files']}
def save(n,v):(D/n).write_text(json.dumps(v,indent=2),encoding='utf-8')
save('registration.json',dict(source_paper='https://arxiv.org/html/0810.2125v1',
    operation='Use standard unblanked cubes; noise MAD in positions blanked in released MOM0; no fitted velocity residual used. Smooth to circular FWHM 30 arcsec; >2 smoothed RMS in three consecutive channels; union of qualifying triples.',
    limitations=['Approximate reconstruction of published detection support; not the original master mask.',
        'Selection and flux residual rescaling prevent direct equivalence to released moment maps.',
        'Per-channel RMS and adjacent-channel correlations are not a calibrated MOM1 uncertainty map.',
        'No new void, 3D density or gravity claim.']))
results=[];start=time.perf_counter()
for item in sorted(cubes['files'],key=lambda x:x['name']):
    name=item['name'];path=ROOT/item['file']
    with fits.open(path,memmap=True) as hd:
        h=hd[0].header;a=np.squeeze(hd[0].data)
        assert a.ndim==3 and h['BUNIT'].strip()=='JY/BEAM'
        m0=np.squeeze(fits.getdata(ROOT/assets[(name,'NA',0)]['file']))
        assert a.shape[1:]==m0.shape
        blank=(~np.isfinite(m0))|(m0==0)
        bg=blank[::4,::4]
        small=np.array(a[:,::4,::4],dtype=np.float64)
        sample=small[:,bg]
        median=np.nanmedian(sample,axis=1)
        rms=1.482602218*np.nanmedian(abs(sample-median[:,None]),axis=1)
        standardized=(sample-median[:,None])/rms[:,None]
        clipped=np.clip(standardized,-3,3)
        lag1=float(np.nanmean(clipped[:-1]*clipped[1:])/np.nanmean(clipped**2))
        lag2=float(np.nanmean(clipped[:-2]*clipped[2:])/np.nanmean(clipped**2))
        history='\n'.join(str(s) for s in h.get('HISTORY',[]))
        matches=re.findall(r'CLEAN BMAJ=\s*([\d.E+-]+) BMIN=\s*([\d.E+-]+) BPA=\s*([\d.E+-]+)',history)
        if matches:bm,bn,bp=map(float,matches[-1])
        else:bm,bn,bp=h['BMAJ'],h['BMIN'],h['BPA']
        pixel=abs(h['CDELT1'])*3600;angle=np.deg2rad(bp)
        u=np.array([-np.sin(angle),np.cos(angle)]);v=np.array([np.cos(angle),np.sin(angle)])
        beam=(bm*3600/2.354820045)**2*np.outer(u,u)+(bn*3600/2.354820045)**2*np.outer(v,v)
        covariance=(np.eye(2)*(30/2.354820045)**2-beam)/pixel**2
        assert np.linalg.eigvalsh(covariance).min()>0
        reach=int(np.ceil(5*np.sqrt(np.linalg.eigvalsh(covariance).max())))
        yy,xx=np.mgrid[-reach:reach+1,-reach:reach+1];inv=np.linalg.inv(covariance)
        k=np.exp(-.5*(inv[0,0]*xx**2+2*inv[0,1]*xx*yy+inv[1,1]*yy**2));k/=k.sum()
        kg=cp.asarray(k,dtype=cp.float32);previous=[];detected=np.zeros(m0.shape,bool);smoothed_rms=[]
        for channel in range(a.shape[0]):
            img=cp.asarray(np.nan_to_num(np.array(a[channel],dtype=np.float32)))
            smoothed=cp.asnumpy(fftconvolve(img,kg,mode='same'))
            b=smoothed[::4,::4][bg];center=np.median(b)
            sigma=1.482602218*np.median(abs(b-center));smoothed_rms.append(float(sigma))
            previous.append(smoothed>center+2*sigma)
            if len(previous)>3:previous.pop(0)
            if len(previous)==3:detected|=previous[0]&previous[1]&previous[2]
        official=~blank
        union=detected|official;intersect=detected&official
        result=dict(name=name,shape=list(a.shape),cube_bytes=item['bytes'],
            sampled_finite_fraction=float(np.isfinite(small).mean()),sampled_zero_fraction=float((small==0).mean()),
            channel_rms_mjy=(rms*1000).tolist(),median_channel_rms_mjy=float(np.median(rms)*1000),
            channel_lag1_correlation=lag1,channel_lag2_correlation=lag2,
            median_smoothed_rms_mjy=float(np.median(smoothed_rms)*1000),
            official_detected_pixels=int(official.sum()),reconstructed_detected_pixels=int(detected.sum()),
            support_intersection_over_union=float(intersect.sum()/union.sum()),
            official_support_recovered=float(intersect.sum()/official.sum()),
            cube_beam_arcsec=[bm*3600,bn*3600],moment1_uncertainty_calibrated=False)
        results.append(result);save('partial.json',results)
        print(f'{name}: noise {result["median_channel_rms_mjy"]:.3f} mJy/beam; support IoU {result["support_intersection_over_union"]:.3f}; lag1 {lag1:.3f}',flush=True)
save('result.json',dict(status='COMPLETED',objects=results,seconds=time.perf_counter()-start))
