"""Transverse linear response of the retained static action; not a stability proof."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

root=Path(__file__).parent/'Invariant'
sys.path.insert(0,str(root/'src'))
from invariant_gravity_extensions.length_screening import LengthScreening, anomalous_flux
dest=root/'work/gravity-first-principles/shortwave-transfer-002'
dest.mkdir(exist_ok=False)
module=root/'src/invariant_gravity_extensions/length_screening.py'
registration=dict(module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
    shapes=[.5,1.,2.],x_values=np.logspace(-8,8,9).tolist(),
    wave_number_ratios=[.5,2.],combined_perturbation_ratios=[1e-5,1e-6],
    background='Constant auxiliary gradient in z; perturbation delta psi=A cos(k x), transverse to background. a0=ell=1.',
    transfer='T(k)=1+Eprime(x)+ell²*x*Kprime(x)*k². P_h=x*Kprime(x)<0 for the concave excess E.',
    scope='Formal static linear response about a uniform external field. Density perturbation is signed; not a full nonnegative-source counterexample, dynamical stability proof, or physical exclusion.')
(dest/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
(dest/'length_screening.py').write_bytes(module.read_bytes())
rows=[]
for shape in registration['shapes']:
    spec=LengthScreening(shape)
    for x in registration['x_values']:
        k0,ph,_,ep=spec.kernel(x)
        long=float(1+ep)
        assert ph<0
        critical=float(np.sqrt(long/-ph))
        checks=[]
        for ratio in registration['wave_number_ratios']:
            k=critical*ratio
            target=float(long+ph*k*k)
            for small in registration['combined_perturbation_ratios']:
                amplitude=small*np.sqrt(x)/(k*max(1.,k))
                phase=np.pi/4
                p=np.array([-amplitude*k*np.sin(phase),0.,np.sqrt(x)])
                H=np.zeros((3,3)); H[0,0]=-amplitude*k*k*np.cos(phase)
                third=amplitude*k**3*np.sin(phase)
                dH2=np.array([2*H[0,0]*third,0.,0.])
                dlap=np.array([third,0.,0.])
                J=p+anomalous_flux(spec,p,H,dH2,dlap,1.,1.)
                measured=float(J[0]/p[0])
                error=abs(measured-target)/max(long,abs(target))
                checks.append(dict(k_over_critical=ratio,combined_amplitude_bound=small,
                    predicted_transfer=target,measured_transfer=measured,scaled_error=error))
        rows.append(dict(shape=shape,x=x,ph=float(ph),critical_k_times_length=critical,
            critical_wavelength_over_length=2*np.pi/critical,checks=checks))
out=dict(registration=registration,rows=rows,
    maximum_smallest_amplitude_error=max(c['scaled_error'] for r in rows for c in r['checks'] if c['combined_amplitude_bound']==1e-6),
    observational_scores=0,physical_exclusions=0)
(dest/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('Maximum small-amplitude error',out['maximum_smallest_amplitude_error'])
print('x=1 thresholds',[(r['shape'],r['critical_wavelength_over_length']) for r in rows if r['x']==1])
