"""Registered development pilot: conditional channel-cube prediction, CUDA.

Not a tilted-ring 3D density inversion. Brightness is conditioned on the same
observations; spatial holdouts are not independent observing epochs.
The frozen pilot-001 runner included a gas diagnostic using blanked zeros.
That diagnostic is removed here. Use audit_gravity_cube_gas_context.py for
the covered-emission comparison.
"""
import argparse, hashlib, json, shutil, traceback, warnings
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.linalg import toeplitz
from astropy.io import fits
from astropy.wcs import WCS
import torch
from gravity_cube_model import CubeModel, controls, tensor

ROOT=Path(__file__).resolve().parents[1]
def read(p): return json.loads((ROOT/p).read_text())
def save(p,d): p.write_text(json.dumps(d,indent=2,allow_nan=False))
def block(a,k=8):
    h,w=a.shape[-2:];return a.reshape(*a.shape[:-2],h//k,k,w//k,k).mean(axis=(-3,-1))
def beam_values(header):
    import re
    for line in header.get('HISTORY',[]):
        m=re.search(r'CLEAN BMAJ=\s*([\d.E+-]+) BMIN=\s*([\d.E+-]+) BPA=\s*([\d.E+-]+)',line)
        if m:return float(m[1])*3600,float(m[2])*3600,float(m[3])
    return float(header['BMAJ'])*3600,float(header['BMIN'])*3600,float(header.get('BPA',0))

def prepare(name,geom,cube_asset,moment_asset,cache):
    with fits.open(ROOT/cube_asset['file'],memmap=False) as hd:
        raw=np.squeeze(hd[0].data).astype(np.float32);h=hd[0].header.copy()
    if not np.all(np.isfinite(raw)):raise ValueError('Nonfinite standard cube requires separate coverage treatment')
    w=WCS(h);nc,ny,nx=raw.shape;k=8;pixel=abs(h['CDELT1'])*3600*k
    edges=w.spectral.all_pix2world((np.arange(nc+1)-.5)[:,None],0).ravel()
    ctype=w.spectral.wcs.ctype[0]
    if ctype=='FREQ':edges=299792.458*(h['RESTFREQ']/edges-1)
    elif ctype.startswith('VRAD'):
        radio=edges/1000;edges=radio/(1-radio/299792.458)
    else:edges=edges/1000
    if edges[0]>edges[-1]:edges=edges[::-1].copy();raw=raw[::-1].copy()
    assert np.all(np.diff(edges)>0)
    centers=(edges[1:]+edges[:-1])/2
    # Common additional smoothing. Native beam is retained in source brightness;
    # its velocity mixing is an explicitly unmodeled small-scale approximation.
    bmaj,bmin,bpa=beam_values(h);target=48.;sigma=np.sqrt(target**2-bmaj**2)/2.354820045
    native=abs(h['CDELT1'])*3600
    source=block(np.einsum('c,chw->hw',np.diff(edges),raw))
    data=block(gaussian_filter(raw,(0,sigma/native,sigma/native),mode='constant'))
    del raw
    n,m=source.shape;yy,xx=np.mgrid[:n,:m]
    ra,dec=w.celestial.all_pix2world(xx*k+(k-1)/2,yy*k+(k-1)/2,0)
    east=(ra-geom['ra'])*np.cos(np.deg2rad(geom['dec']))*3600
    north=(dec-geom['dec'])*3600
    pa,inc=np.deg2rad([geom['pa'],geom['inc']]);major=east*np.sin(pa)+north*np.cos(pa)
    minor=(east*np.cos(pa)-north*np.sin(pa))/np.cos(inc);radius=np.hypot(major,minor)
    # Geometry only: all spectral channels, no detection or velocity threshold.
    interior=((xx%16)>=5)&((xx%16)<11)&((yy%16)>=5)&((yy%16)<11)
    parity=(xx//16+yy//16)%2;aperture=(radius<450)&(radius>48)
    train=aperture&interior&(parity==0);test=aperture&interior&(parity==1)
    if min(train.sum(),test.sum())<30:raise ValueError('Insufficient geometric holdout area')
    sky=np.hypot(east,north)
    background=(sky>550)&(sky<680)&(xx>6)&(yy>6)&(xx<m-7)&(yy<n-7)
    bgtrain=background&(parity==0);bgtest=background&(parity==1)
    if min(bgtrain.sum(),bgtest.sum())<200:raise ValueError('Insufficient separate background')
    noise=data[:,bgtrain].astype(float);offset=np.mean(noise,axis=1)
    noise-=offset[:,None];std=np.sqrt(np.mean(noise**2,axis=1));z=noise/std[:,None]
    corr=np.array([1.]+[float(np.mean(z[lag:]*z[:-lag]))*(1-lag/7) for lag in range(1,7)]+[0.]*(nc-7))
    correlation=toeplitz(corr);mineig=np.linalg.eigvalsh(correlation).min()
    jitter=max(0.,.05-mineig);correlation=(correlation+np.eye(nc)*jitter)/(1+jitter)
    covariance=std[:,None]*correlation*std[None,:]
    whitener=np.linalg.inv(np.linalg.cholesky(covariance))
    data=data-offset[:,None,None]
    independent=whitener@data[:,bgtest]
    diag=np.mean(independent**2,axis=1)
    lag1=float(np.mean(independent[1:]*independent[:-1]))
    noise_ok=bool(.5<np.median(diag)<2 and abs(lag1)<.15)
    # Brightness is a conditioned observable, not a held-out velocity measurement.
    amplitude=np.maximum(source-np.sum(offset*np.diff(edges)),0)
    fy=np.fft.fftfreq(2*n);fx=np.fft.rfftfreq(2*m)
    transfer=np.exp(-2*np.pi**2*(sigma/pixel)**2*(fy[:,None]**2+fx[None,:]**2))
    # Initial motion uses only training spectra (positive weights only for seed).
    positive=np.maximum(data[:,train],0);flux=positive.sum(axis=0)
    centroid=(centers@positive)/np.maximum(flux,1e-12)
    ct=major[train]/np.maximum(radius[train],1)
    design=np.column_stack([np.ones(len(ct)),np.sin(inc)*ct])
    coef=np.linalg.lstsq(design*np.sqrt(flux[:,None]),centroid*np.sqrt(flux),rcond=None)[0]
    scale=max(abs(coef[1]),40.);rot=np.array([0,.7,1,1,1])*coef[1]
    # HI context from published residual-rescaled MOM0, never interpreted as all matter.
    mom=np.squeeze(fits.getdata(ROOT/moment_asset['file'])).astype(float)*.001
    hi=block(np.maximum(np.nan_to_num(mom),0))
    local=gaussian_filter(hi,48/2.35482/pixel)
    broad=gaussian_filter(hi,96/2.35482/pixel)
    context=np.clip(np.log((broad+1e-3)/(local+1e-3)),-1,1)
    packet=dict(east=east,north=north,amplitude=amplitude,velocity_edges=edges,
        radius=np.minimum(radius,600),rmax=600,beam_transfer=transfer,pa=pa,inc=inc,
        gas_context=context,speed_scale=scale,vsys_initial=float(coef[0]),cube=data,
        whitener=whitener,train_mask=train,test_mask=test,rotation_initial=rot)
    np.savez_compressed(cache/(name+'.npz'),**packet)
    audit=dict(name=name,channels=nc,shape=list(data.shape),train_pixels=int(train.sum()),test_pixels=int(test.sum()),
        spectral_input=ctype,velocity_edges_kms=[float(edges[0]),float(edges[-1])],
        train_blocks=int(len(set(zip((xx[train]//16).tolist(),(yy[train]//16).tolist())))),
        test_blocks=int(len(set(zip((xx[test]//16).tolist(),(yy[test]//16).tolist())))),
        mask_uses_response=False,mask_overlap=int(np.sum(train&test)),guard_gap_arcsec=float(10*pixel),
        native_beam_arcsec=[bmaj,bmin,bpa],extra_smoothing_sigma_arcsec=float(sigma),
        native_major_variance_fraction=float((bmaj/target)**2),
        background_train_pixels=int(bgtrain.sum()),background_test_pixels=int(bgtest.sum()),
        whitened_validation_median_variance=float(np.median(diag)),whitened_validation_lag1=lag1,
        covariance_pass=noise_ok,covariance_diagonal_jitter=float(jitter),
        covariance_lag_coefficients=corr[:7].tolist(),vsys_initial=float(coef[0]),rotation_initial_kms=rot.tolist(),
        geometry=geom,cube_sha256=cube_asset['sha256'],hi_context_sha256=moment_asset['sha256'])
    return packet,audit

def main():
    arg=argparse.ArgumentParser();arg.add_argument('--run-id',default='conditional-cube-pilot-001');args=arg.parse_args()
    dest=ROOT/'work/gravity-first-principles'/args.run_id;dest.mkdir(exist_ok=False)
    cache=ROOT/'work/private'/args.run_id;cache.mkdir(exist_ok=False)
    reg=read('work/gravity-first-principles/things-direct-patterns-003/registration.json')
    registration=dict(names=reg['names'],geometry=reg['geometry'],status='DEVELOPMENT_PILOT',
        modes=['rotation','warp','stream','asymmetric','full'],maxiter=180,
        mask='Fixed deprojected 48<r<450 arcsec; 192 arcsec checkerboard blocks, interiors pixels 5..10, all channels. No intensity or velocity selection.',
        noise='Separate sky annulus 550..680 arcsec, checkerboard train/test; channel variances plus tapered six-lag Toeplitz correlation, positive-definite floor. Gate: held-out median whitened variance .5..2 and absolute lag1<.15.',
        gas_beta_grid=np.linspace(-.3,.3,13).tolist(),
        gas_test='Freeze full-model nuisance parameters; select one gas beta from OTHER galaxies training spectra, equal galaxy weights; score spatial test spectra. No target test fitting.',
        limits=['Conditional on same-observation brightness; not independent observing data or pristine confirmation.',
            'Projected coarse cube approximation; native-beam velocity mixing, disk thickness and exact instrumental spectral response not validated.',
            'Warp, radial streaming and lagging profiles need not be physically identifiable from line-of-sight data.',
            'Free rotation curve absorbs smooth radial gravity changes; this is a kinematic nuisance test, not a gravity-law validation.',
            'HI context is not total density. Stellar/CO assets are audited separately; their mass conversion and matched-resolution modeling remain necessary.',
            'Spectral whitening does not model spatial covariance; no pixel-count significance or calibrated chi-square.'])
    save(dest/'registration.json',registration)
    for file in [Path(__file__),Path(__file__).with_name('gravity_cube_model.py')]:shutil.copy2(file,dest/file.name)
    save(dest/'numerical-controls.json',controls())
    cubes={a['name']:a for a in read('work/gravity-first-principles/things-cube-acquisition-001/receipt.json')['files']}
    moments={a['name']:a for a in read('work/gravity-first-principles/things-observable-acquisition-003/receipt.json')['files'] if a['resolution']=='NA' and a['moment']==0}
    rows=[];failures=[];audits=[]
    for name in registration['names']:
        try:
            print('PREPARE',name,flush=True)
            packet,audit=prepare(name,reg['geometry'][name],cubes[name],moments[name],cache)
            audits.append(audit);save(dest/'data-audit.json',audits)
            if not audit['covariance_pass']:
                failures.append(dict(name=name,stage='noise_validation',reason='Independent background covariance gate failed'));continue
            model=CubeModel(packet);fitsout=[]
            base=model.fit('rotation',maxiter=180);fitsout.append(base)
            for mode in ('warp','stream','asymmetric'):
                fitsout.append(model.fit(mode,base['params'],maxiter=180))
            initial=min(fitsout,key=lambda a:a['objective'])['params']
            fitsout.append(model.fit('full',initial,maxiter=250))
            row=dict(name=name,fits=fitsout)
            rows.append(row);save(dest/(name+'.json'),row)
            print('FIT',name,[(f['mode'],round(f['test_loss'],3),f['optimizer_success']) for f in fitsout],flush=True)
            del model;torch.cuda.empty_cache()
        except Exception as e:
            failures.append(dict(name=name,stage='exception',error=repr(e),traceback=traceback.format_exc()))
            print('FAILED',name,repr(e),flush=True)
        save(dest/'failures.json',failures)
    result=dict(status='COMPLETED_DEVELOPMENT_PILOT',objects=rows,failures=failures,
        legacy_gas_diagnostic_admissible=False,gas_replacement='audit_gravity_cube_gas_context.py',
        torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0),interpretation=registration['limits'])
    save(dest/'result.json',result)
    print('DONE',len(rows),'fits',len(failures),'failures; gas scoring requires separate coverage audit',flush=True)

if __name__=='__main__':
    warnings.filterwarnings('ignore',category=Warning,module='astropy')
    main()
