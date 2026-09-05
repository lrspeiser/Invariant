"""Finite positive spherical sources and full length-action force response."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import quad
root=Path(__file__).parent/'Invariant'
sys.path.insert(0,str(root/'src'))
from invariant_gravity_extensions.length_screening import LengthScreening
from invariant_gravity_extensions.smooth_spherical_source import spherical_acceleration_derivatives, spherical_length_anomaly
dest=root/'work/gravity-first-principles/finite-positive-response-002'
dest.mkdir(exist_ok=False)
paths=[root/'src/invariant_gravity_extensions'/name for name in ['length_screening.py','smooth_spherical_source.py']]
registration=dict(input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    shapes=[.5,1.,2.],plummer_scales=[100.,300.],contrasts=[.01,.1],k_ratios=[.5,2.,8.],sample_counts=[256,512],
    source='rho=3M/(4*pi*a³)*(1+r²/a²)^(-5/2)*(1+c*cos(k*r)); M=2*sqrt(2)*a², G=a0=ell=1.',
    positivity='0<c<1 implies positive density at every finite radius; mass bounded between (1-c)M and (1+c)M.',
    force='Full spherical action variation with regular origin, eliminating the extra C/r² integration constant. Both background and perturbed sources use the same origin condition.',
    reference='At r=a the unperturbed Newtonian acceleration is 1. Use local zero-Hessian parallel threshold sqrt((1+Eprime+2*x*Esecond)/(-x*Kprime)) to choose k; finite-source comparison is measured directly.',
    scope='Synthetic finite positive sources; static perturbation transfer, not observational exclusion or dynamical-stability proof.')
(dest/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
for p in paths: (dest/p.name).write_bytes(p.read_bytes())
rows=[]
for shape in registration['shapes']:
    spec=LengthScreening(shape)
    ep,epp,_=spec.excess_derivatives(1.)
    _,ph,_,_=spec.kernel(1.)
    critical=float(np.sqrt((1+ep+2*epp)/-ph))
    for a in registration['plummer_scales']:
        M=2*np.sqrt(2)*a*a
        def rho0(r): return 3*M/(4*np.pi*a**3)*(1+(r/a)**2)**-2.5
        for c in registration['contrasts']:
            for ratio in registration['k_ratios']:
                k=ratio*critical
                coeff=[]; minima=[]; integration_errors=[]
                for n in registration['sample_counts']:
                    r=a+np.linspace(-4*np.pi,4*np.pi,n,endpoint=False)/k
                    assert r.min()>0
                    density=rho0(r); derivative=-5*r/(a*a+r*r)*density
                    mass=M*r**3/(a*a+r*r)**1.5
                    integrals=[quad(lambda t:t*t*rho0(t),0,float(R),weight='cos',wvar=k,epsabs=1e-11,epsrel=1e-10) for R in r]
                    dm=4*np.pi*c*np.array([z[0] for z in integrals])
                    rho=density*(1+c*np.cos(k*r))
                    drho=derivative*(1+c*np.cos(k*r))-c*k*density*np.sin(k*r)
                    g0,dg0,ddg0=spherical_acceleration_derivatives(r,mass,density,derivative,1.)
                    g,dg,ddg=spherical_acceleration_derivatives(r,mass+dm,rho,drho,1.)
                    physical0=g0+spherical_length_anomaly(spec,r,g0,dg0,ddg0,1.,1.)
                    physical=g+spherical_length_anomaly(spec,r,g,dg,ddg,1.,1.)
                    delta=g-g0
                    coeff.append(float(np.dot(physical-physical0,delta)/np.dot(delta,delta)))
                    minima.append(float(physical.min()))
                    integration_errors.append(float(max(z[1] for z in integrals)*4*np.pi*c))
                rows.append(dict(shape=shape,scale=a,contrast=c,k_ratio=ratio,k=k,
                    transfer_coefficients=coeff,minimum_total_inward_force=minima,
                    mass_quadrature_error_estimates=integration_errors,
                    sampling_change=abs(coeff[-1]-coeff[0])/max(1,abs(coeff[-1])),
                    expected_high_frequency_sign=bool(coeff[-1]<0) if ratio>1 else bool(coeff[-1]>0)))
        print('shape',shape,'scale',a,'complete',flush=True)
out=dict(registration=registration,rows=rows,all_expected_signs=all(r['expected_high_frequency_sign'] for r in rows),
    max_sampling_change=max(r['sampling_change'] for r in rows),observational_scores=0,physical_exclusions=0)
(dest/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('All expected signs',out['all_expected_signs'],'max sampling change',out['max_sampling_change'])
