"""Constrained kinematic repair and independently rendered mismatch tests."""
import argparse,json,shutil,traceback
from pathlib import Path
import numpy as np
from scipy.special import erf
from scipy.ndimage import gaussian_filter
import torch
from gravity_cube_model import tensor
from gravity_cube_constrained import ConstrainedCube
ROOT=Path(__file__).resolve().parents[1]
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False))

def synthetic(mode,seed=42):
    # Independent NumPy generator at twice the fitted spatial resolution.
    # Analytic tanh rotation, radial dispersion, vertical layers and lag; no use
    # of the fitting renderer or its piecewise-linear radial coefficients.
    rng=np.random.default_rng(seed);n=96;pixel=3.;y,x=np.mgrid[:n,:n]
    east=(x-(n-1)/2)*pixel;north=(y-(n-1)/2)*pixel
    pa=.35;inc=.9;edges=np.linspace(-180,180,49);lower=edges[:-1,None,None];upper=edges[1:,None,None]
    cube=np.zeros((48,n,n));amplitude=np.zeros((n,n));moment=np.zeros((n,n))
    layers=np.array([0.]) if mode!='thick_lag_hanning' else np.array([-16,-8,0,8,16.])
    weights=np.exp(-.5*(layers/8)**2);weights/=weights.sum()
    for z,weight in zip(layers,weights):
        e=east-z*np.sin(inc)*np.cos(pa);nn=north+z*np.sin(inc)*np.sin(pa)
        reference=np.hypot(e*np.sin(pa)+nn*np.cos(pa),(e*np.cos(pa)-nn*np.sin(pa))/np.cos(inc))
        angle=pa+(np.deg2rad(8)*(reference/160)**2 if mode=='warp' else 0)
        major=e*np.sin(angle)+nn*np.cos(angle);minor=(e*np.cos(angle)-nn*np.sin(angle))/np.cos(inc)
        r=np.hypot(major,minor);ct=major/np.maximum(r,1e-9);st=minor/np.maximum(r,1e-9)
        brightness=12*np.exp(-r/65)*(1+.15*np.cos(e/22)*np.sin(nn/31))
        speed=125*np.tanh(r/45)*max(.5,1-.0125*abs(z))
        vr=18*np.sin(r/80) if mode=='stream' else 0
        velocity=np.sin(inc)*(speed*ct+vr*st)
        sigma=7+5*np.exp(-r/90)
        profile=.5*(erf((upper-velocity)/(np.sqrt(2)*sigma))-erf((lower-velocity)/(np.sqrt(2)*sigma)))/(upper-lower)
        cube+=weight*brightness*profile;amplitude+=weight*brightness;moment+=weight*brightness*velocity
    if mode=='thick_lag_hanning':cube=.5*cube+.25*np.concatenate([np.zeros_like(cube[:1]),cube[:-1]])+.25*np.concatenate([cube[1:],np.zeros_like(cube[:1])])
    cube=gaussian_filter(cube,(0,2,2),mode='constant')
    def down(a):return a.reshape(*a.shape[:-2],48,2,48,2).mean(axis=(-3,-1))
    clean=down(cube);amp=down(amplitude);e=down(east);nn=down(north)
    radius=np.hypot(e*np.sin(pa)+nn*np.cos(pa),(e*np.cos(pa)-nn*np.sin(pa))/np.cos(inc))
    nc=48;cov=.25**np.abs(np.arange(nc)[:,None]-np.arange(nc)[None,:])*.004**2
    noise=(np.linalg.cholesky(cov)@rng.normal(size=(nc,48*48))).reshape(nc,48,48)
    fy=np.fft.fftfreq(96);fx=np.fft.rfftfreq(96);beam=np.exp(-2*np.pi**2*(fy[:,None]**2+fx[None,:]**2))
    yy,xx=np.mgrid[:48,:48]
    return dict(east=e,north=nn,amplitude=amp,velocity_edges=edges,radius=radius,rmax=600,
        beam_transfer=beam,pa=pa,inc=inc,gas_context=np.zeros_like(e),speed_scale=120,vsys_initial=0,
        cube=clean+noise,whitener=np.linalg.inv(np.linalg.cholesky(cov)),train_mask=(xx<18),test_mask=(xx>30),
        rotation_initial=np.array([0,120,125,125,125])),clean

def fit_object(packet):
    m=ConstrainedCube(packet)
    seeds=[m.seed(f) for f in (.7,1,1.3)]
    seeds[0][15]=-.7;seeds[2][15]=.7
    rotation=m.fit_multistart('rotation',seeds)
    fits=[rotation]
    for mode in ('warp','stream','asymmetric'):
        fits.append(m.fit_multistart(mode,[rotation['params']]))
    best=min(fits,key=lambda r:r['objective'])
    null=np.array(rotation['params']);null[17:22]=0;null[22]=.2
    mixed=np.array(rotation['params']);mixed[17:21]=[.25,-.25,.15,-.15];mixed[21:23]=[.1,.2]
    fits.append(m.fit_multistart('full',[null,best['params'],mixed],maxiter=380))
    return fits

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['synthetic','real'],required=True);args=parser.parse_args()
    dest=ROOT/'work/gravity-first-principles'/('constrained-cube-'+args.kind+'-001');dest.mkdir(exist_ok=False)
    registration=dict(kind=args.kind,rotation='One training-selected spin; nonnegative speeds <=600 km/s at 40,80,140,220,320,450,600 arcsec; zero at center.',
        dispersion='Three radial coefficients 3..40 km/s.',geometry='Global PA +/-20 and inclination +/-10 degrees; outer warp bounds unchanged at 15/8 degrees. Six damped implicit radius iterations.',
        starts='Three rotation starts at 0.7/1/1.3 seed speed and -7/0/+7 PA; full starts from nested rotation, best simpler training fit and fixed mixed seed.',
        selection='Minimum penalized training objective among converged starts; held-out score evaluated only afterwards.',
        no_gas_formula_selection=True,original_noise_and_masks_reused=True,
        caution='Conditional same-observation source brightness; projected model, native beam velocity mixing and complete spectral response remain unvalidated.',
        synthetic_generator='Independent NumPy analytic tanh rotation + varying dispersion at twice resolution; one case includes multiple vertical layers, lag and Hanning spectral smoothing.')
    save(dest/'registration.json',registration)
    for f in [Path(__file__),Path(__file__).with_name('gravity_cube_constrained.py')]:shutil.copy2(f,dest/f.name)
    names=['rotation','warp','stream','thick_lag_hanning'] if args.kind=='synthetic' else json.loads((ROOT/'work/gravity-first-principles/conditional-cube-pilot-001/registration.json').read_text())['names']
    results=[];failures=[]
    for name in names:
        try:
            print('START',name,flush=True)
            if args.kind=='synthetic':packet,clean=synthetic(name)
            else:packet=dict(np.load(ROOT/'work/private/conditional-cube-pilot-001'/(name+'.npz')))
            if args.kind=='synthetic' and name=='rotation':
                model=ConstrainedCube(packet);p=model.seed();p[17:22]=0
                with torch.no_grad():nested=float(torch.max(torch.abs(model.render(tensor(p),'rotation')-model.render(tensor(p),'full'))))
                assert nested<1e-7
                q=tensor(p).requires_grad_(True);loss=model.loss(q,'rotation');loss.backward()
                lo=p.copy();hi=p.copy();lo[2]-=.001;hi[2]+=.001
                difference=(float(model.loss(tensor(hi),'rotation'))-float(model.loss(tensor(lo),'rotation')))/.002
                error=abs(float(q.grad[2])-difference)/max(abs(difference),1)
                assert error<.02
                save(dest/'numerical-controls.json',dict(nested_model_max_error=nested,gradient_relative_error=error))
            fits=fit_object(packet)
            assert all(min(f['params'][:7])>=0 for f in fits)
            result=dict(name=name,fits=fits)
            if args.kind=='synthetic':
                white=packet['whitener'];mask=packet['test_mask']
                result['known_noise_loss']=float(np.mean((white@(packet['cube']-clean)[:,mask])**2))
            results.append(result);save(dest/(name+'.json'),result)
            print('DONE',name,[(f['mode'],round(f['test_loss'],3),f['optimizer_success']) for f in fits],flush=True)
            torch.cuda.empty_cache()
        except Exception as e:
            failures.append(dict(name=name,error=repr(e),traceback=traceback.format_exc()));print('FAIL',name,repr(e),flush=True)
        save(dest/'failures.json',failures)
    save(dest/'result.json',dict(status='COMPLETE_DIAGNOSTIC' if not failures else 'INCOMPLETE',objects=results,failures=failures,registration=registration))

if __name__=='__main__':main()
