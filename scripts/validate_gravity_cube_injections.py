"""Known-source injection controls; reports ambiguity rather than assigning causes."""
import json,shutil
from pathlib import Path
import numpy as np
import torch
from gravity_cube_model import CubeModel,controls,tensor
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'work/gravity-first-principles/cube-numerical-validation-001';D.mkdir(exist_ok=False)
shutil.copy2(__file__,D/'runner.py')
shutil.copy2(Path(__file__).with_name('gravity_cube_model.py'),D/'gravity_cube_model.py')
registration={'truth_modes':['rotation','warp','stream','asymmetric'],
 'candidate_modes':['rotation','warp','stream','asymmetric','full'],
 'scope':'Same-model synthetic injections validate computation and expose parameter ambiguity. They do not validate the model against independent physics.',
 'seed':20260905,'test':'Known brightness; separate left/right spatial regions; correlated spectral noise; assess every candidate, without selecting real-data models.'}
(D/'registration.json').write_text(json.dumps(registration,indent=2))
rng=np.random.default_rng(registration['seed']);n=32;yy,xx=np.mgrid[:n,:n]
x=(xx-15.5)*3;y=(yy-15.5)*3;pa=.3;inc=.8
major=x*np.sin(pa)+y*np.cos(pa);minor=(x*np.cos(pa)-y*np.sin(pa))/np.cos(inc);rad=np.hypot(major,minor)
fy=np.fft.fftfreq(2*n);fx=np.fft.rfftfreq(2*n)
beam=np.exp(-2*np.pi**2*1.5**2*(fy[:,None]**2+fx[None,:]**2))
cov=.25**np.abs(np.arange(32)[:,None]-np.arange(32)[None,:])*.002**2
packet=dict(east=x,north=y,amplitude=np.exp(-rad/35)*10,velocity_edges=np.linspace(-120,120,33),radius=rad,
 rmax=100,beam_transfer=beam,pa=pa,inc=inc,gas_context=np.sin(x/30),speed_scale=100,vsys_initial=0,
 cube=np.zeros((32,n,n)),whitener=np.linalg.inv(np.linalg.cholesky(cov)),train_mask=xx<12,test_mask=xx>20,
 rotation_initial=np.array([0,.5,.8,1.,1.])*100)
results=[]
for truthmode in registration['truth_modes']:
 model=CubeModel(packet);p=np.zeros(18);p[:5]=[0,.5,.8,1,1];p[6]=.2
 if truthmode=='warp':p[12:14]=[.7,.5]
 if truthmode=='stream':p[14:16]=[.5,-.4]
 if truthmode=='asymmetric':p[16:18]=[.25,.3]
 with torch.no_grad():truth=model.render(tensor(p),truthmode).cpu().numpy()
 noise=(np.linalg.cholesky(cov)@rng.normal(size=(32,n*n))).reshape(32,n,n)
 model.data=tensor(truth+noise)
 base=model.fit('rotation',maxiter=250);fitsout=[base]
 for mode in registration['candidate_modes'][1:]:fitsout.append(model.fit(mode,base['params'],maxiter=250))
 results.append({'truth':truthmode,'fits':fitsout})
 print(truthmode,[(f['mode'],round(f['test_loss'],3),f['optimizer_success']) for f in fitsout],flush=True)
result={'numerical_controls':controls(),'injections':results,'registration':registration,
 'physical_identification_certified':False,'independent_real_data_validation':False}
(D/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
