"""Self-adjoint Helmholtz curvature filter: discrete action-variation checks."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
root=Path(__file__).parent/'Invariant'
sys.path.insert(0,str(root/'src'))
from invariant_gravity_extensions.length_screening import LengthScreening
dest=root/'work/gravity-first-principles/filtered-action-variation-001'
dest.mkdir(exist_ok=False)
module=root/'src/invariant_gravity_extensions/length_screening.py'
registration=dict(module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
    filter='S=(1-L²*Laplacian)^(-1), self-adjoint translation-invariant periodic realization',
    action='P(x,h), x=|grad psi|²/a0², h=ell²|Hessian S psi|²/a0²',
    flux='J=Px grad psi - ell² S div(Ph Hessian S psi); nonlinear product remains inside outer S.',
    shapes=[.5,1.,2.],filter_lengths=[0.,.1,1.],lengths=[.1,1.],nodes=512,
    scope='Static nonlocal ansatz and one-dimensional periodic variation controls; no relativistic completion, full stability or observational admission.')
(dest/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
n=512; z=np.arange(n)*2*np.pi/n; wave=np.fft.fftfreq(n,1/n)
def derivative(v): return np.fft.ifft(1j*wave*np.fft.fft(v)).real
psi=.03*np.sin(z)+.007*np.cos(3*z)+.002*np.sin(7*z)
direction=np.cos(z)+.3*np.sin(3*z)+.1*np.cos(7*z)
rows=[]
for shape in registration['shapes']:
    spec=LengthScreening(shape)
    for L in registration['filter_lengths']:
        multiplier=1/(1+L*L*wave*wave)
        def S(v): return np.fft.ifft(multiplier*np.fft.fft(v)).real
        for ell in registration['lengths']:
            def action(v):
                p=1+derivative(v); H=derivative(derivative(S(v)))
                return float(np.mean(spec.value(p*p,ell*ell*H*H)))
            p=1+derivative(psi); H=derivative(derivative(S(psi)))
            px,ph,*_=spec.partials(p*p,ell*ell*H*H)
            J=(1+px)*p-ell*ell*S(derivative(ph*H))
            gradient=-2*derivative(J)
            expected=float(np.mean(gradient*direction))
            incorrect=-2*derivative((1+px)*p-ell*ell*derivative(ph*H))
            values=[]
            for eps in [1e-4,1e-5,1e-6]:
                finite=(action(psi+eps*direction)-action(psi-eps*direction))/(2*eps)
                values.append(dict(epsilon=eps,finite_difference=finite,
                    absolute_error=abs(finite-expected)))
            rows.append(dict(shape=shape,L=L,ell=ell,predicted_variation=expected,checks=values,
                smallest_step_absolute_error=values[-1]['absolute_error'],
                omitted_outer_filter_error=abs(float(np.mean(incorrect*direction))-expected),
                constant_shift_residual=abs(float(np.mean(gradient))),
                translation_variation_residual=abs(float(np.mean(gradient*derivative(psi))))))
# Principal-symbol amplification: k²/(1+L² k²)² <= 1/(4L²).
bounds=[]
for L in [.1,1.,10.]:
    k=np.geomspace(1e-6/L,1e6/L,12001)
    factor=k*k/(1+L*L*k*k)**2
    assert max(factor)<=1/(4*L*L)*(1+1e-14)
    bounds.append(dict(L=L,sampled_maximum=float(max(factor)),exact_bound=1/(4*L*L)))
out=dict(registration=registration,rows=rows,principal_symbol_bounds=bounds,
    worst_variation_error=max(r['smallest_step_absolute_error'] for r in rows),
    observational_scores=0,physical_exclusions=0)
(dest/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('Worst action variation error',out['worst_variation_error'])
print('Largest omitted-filter error',max(r['omitted_outer_filter_error'] for r in rows))
print('Largest translation residual',max(r['translation_variation_residual'] for r in rows))
