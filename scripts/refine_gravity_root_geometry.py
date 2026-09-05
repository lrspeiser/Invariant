"""Correct finite-iteration geometry without changing the response-blind split."""
import json,shutil,traceback
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import torch
from gravity_cube_root_geometry import RootGeometryCube
from gravity_cube_model import tensor
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'work/gravity-first-principles/constrained-cube-root-001';D.mkdir(exist_ok=False)
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False))
registration=dict(predecessor='constrained-cube-balanced-001',geometry='36 bracketed bisection steps with implicit gradient; same warp functions and physical bounds.',
 reason='Six damped radius iterations failed the .05 whitened-RMS accuracy check in four objects. Twenty iterations remained inadequate in two.',
 starts='Warp refits from previous warp and rotation; full refits from previous full, nested rotation and asymmetric model. Same training pixels only.',
 unchanged='Source brightness, covariance, physical parameter limits and balanced radial selection. Nonwarp models are algebraically unchanged.',
 limitations='A precise radius root does not certify unique 3D structure. A nonmonotonic ring-mapping scan flags possible folds.',
 selection='Converged minimum training objective; no test-spectrum selection.')
save(D/'registration.json',registration)
for name in ['refine_gravity_root_geometry.py','gravity_cube_root_geometry.py','gravity_cube_constrained.py']:shutil.copy2(Path(__file__).with_name(name),D/name)
source=json.loads((ROOT/'work/gravity-first-principles/constrained-cube-balanced-001/result.json').read_text())

# Independent scalar root solver plus finite-difference gradients before fitting.
packet=dict(np.load(ROOT/'work/private/constrained-cube-balanced-001/NGC3198.npz'))
m=RootGeometryCube(packet);p=m.seed();p[17:19]=[.3,.3];t=tensor(p).requires_grad_(True)
loss=m.loss(t,'full');loss.backward();checks=[]
for index in [2,15,17,18]:
    lo=p.copy();hi=p.copy();lo[index]-=.001;hi[index]+=.001
    difference=(float(m.loss(tensor(hi),'full'))-float(m.loss(tensor(lo),'full')))/.002
    relative=abs(float(t.grad[index])-difference)/max(abs(difference),1)
    assert relative<.03,(index,relative)
    checks.append(dict(parameter=index,relative_gradient_error=relative))
with torch.no_grad():root=m.geometry(tensor(p),'full')[0].cpu().numpy()
scalar_errors=[]
for y,x in np.argwhere(packet['test_mask'])[::10]:
    e=float(packet['east'][y,x]);n=float(packet['north'][y,x])
    def fn(r):
        rr=np.clip(r/600,0,1);pa=float(packet['pa'])+p[15]*np.deg2rad(10)+p[17]*np.deg2rad(15)*rr**2
        inc=np.clip(float(packet['inc'])+p[16]*np.deg2rad(5)+p[18]*np.deg2rad(8)*rr**2,np.deg2rad(10),np.deg2rad(85))
        a=e*np.sin(pa)+n*np.cos(pa);b=(e*np.cos(pa)-n*np.sin(pa))/np.cos(inc)
        return np.sqrt(a*a+b*b+1e-8)-r
    r=brentq(fn,0,np.hypot(e,n)/np.cos(np.deg2rad(85))+1)
    scalar_errors.append(abs(float(root[y,x])-r))
assert max(scalar_errors)<.01
save(D/'numerical-controls.json',dict(gradient_checks=checks,scalar_brent_max_error_arcsec=max(scalar_errors)))
results=[];failures=[]
for obj in source['objects']:
    try:
        name=obj['name'];packet=dict(np.load(ROOT/'work/private/constrained-cube-balanced-001'/(name+'.npz')));m=RootGeometryCube(packet)
        old={f['mode']:f for f in obj['fits']};fits=[old['rotation']]
        fits.append(m.fit_multistart('warp',[old['warp']['params'],old['rotation']['params']],maxiter=240))
        fits.extend([old['stream'],old['asymmetric']])
        null=np.array(old['rotation']['params']);null[17:22]=0;null[22]=.2
        full=m.fit_multistart('full',[old['full']['params'],null,old['asymmetric']['params']],maxiter=280);fits.append(full)
        diag=m.geometry_diagnostic(tensor(full['params']))
        assert diag['max_root_residual_arcsec']<.02
        with torch.no_grad():
            ordinary=m.render(tensor(full['params']),'full');m.root_iterations=52
            reference=m.render(tensor(full['params']),'full')
            precision=float(torch.mean((m.white@(ordinary-reference)[:,m.test])**2).sqrt())
        diag['root_36_vs_52_whitened_rms']=precision
        assert precision<.005
        row=dict(name=name,fits=fits,geometry_diagnostic=diag);results.append(row);save(D/(name+'.json'),row)
        print(name,'full',round(full['test_loss'],3),'success',full['optimizer_success'],'fold_fraction',round(diag['possible_fold_fraction'],3),flush=True)
        del m;torch.cuda.empty_cache()
    except Exception as e:
        failures.append(dict(name=obj['name'],error=repr(e),traceback=traceback.format_exc()));print('FAIL',obj['name'],repr(e),flush=True)
    save(D/'failures.json',failures)
save(D/'result.json',dict(status='COMPLETE_NUMERICAL_REPAIR' if not failures else 'INCOMPLETE',objects=results,failures=failures,registration=registration))
