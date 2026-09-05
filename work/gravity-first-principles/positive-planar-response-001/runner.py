"""Positive-density planar response of the full static length action."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
root=Path(__file__).parent/'Invariant'
sys.path.insert(0,str(root/'src'))
from invariant_gravity_extensions.length_screening import LengthScreening, anomalous_flux
dest=root/'work/gravity-first-principles/positive-planar-response-001'
dest.mkdir(exist_ok=False)
module=root/'src/invariant_gravity_extensions/length_screening.py'
registration=dict(module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
    shapes=[.5,1.,2.],background_curvatures=[.01,.001,.0001],density_contrasts=[.1,.01],
    k_ratios=[.5,2.,8.],phase_nodes=[1024,2048],
    potential='psi=z+b*x²/2-A*cos(k*x)/k², A=c*b, a0=ell=G=1',
    positivity='rho=(b+A*cos(k*x))/(4*pi)>0 for b>0 and 0<c<1, everywhere.',
    force='Translation invariance in y and z gives dPhi/dx=Jx+C. Perturbed and unperturbed solutions use the same C; the sinusoidal coefficient also removes any constant.',
    scope='Exact planar static field response with positive infinite matter distribution; not a finite isolated source, time-dependent stability analysis, or observational exclusion.')
(dest/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
(dest/'length_screening.py').write_bytes(module.read_bytes())
rows=[]
for shape in registration['shapes']:
    spec=LengthScreening(shape)
    _,ph,_,ep=spec.kernel(1.)
    critical=float(np.sqrt((1+ep)/-ph))
    for b in registration['background_curvatures']:
        for contrast in registration['density_contrasts']:
            A=contrast*b
            for ratio in registration['k_ratios']:
                k=ratio*critical
                coefficients=[]
                for n in registration['phase_nodes']:
                    theta=np.linspace(-np.pi,np.pi,n,endpoint=False)
                    x=theta/k
                    p0=np.array([b*x,np.zeros(n),np.ones(n)])
                    H0=np.zeros((3,3,n)); H0[0,0]=b
                    zeros=np.zeros_like(p0)
                    J0=p0+anomalous_flux(spec,p0,H0,zeros,zeros,1.,1.)
                    p=p0.copy(); p[0]+=A/k*np.sin(theta)
                    H=H0.copy(); H[0,0]+=A*np.cos(theta)
                    T=-A*k*np.sin(theta)
                    dH2=zeros.copy(); dH2[0]=2*H[0,0]*T
                    dlap=zeros.copy(); dlap[0]=T
                    J=p+anomalous_flux(spec,p,H,dH2,dlap,1.,1.)
                    # Ratio of physical gradient perturbation to Newtonian
                    # gradient perturbation; acceleration signs cancel in ratio.
                    coefficients.append(float(2*np.mean((J[0]-J0[0])*np.sin(theta))/(A/k)))
                expected=float(1+ep+ph*k*k)
                rows.append(dict(shape=shape,background_curvature=b,density_contrast=contrast,
                    k_over_zero_curvature_threshold=ratio,minimum_density=(b-A)/(4*np.pi),
                    transfer_coefficients=coefficients,zero_curvature_prediction=expected,
                    quadrature_change=abs(coefficients[-1]-coefficients[0])/max(1,abs(coefficients[-1])),
                    response_sign_agrees=bool(np.sign(coefficients[-1])==np.sign(expected))))
assert len(rows)==54 and all(r['minimum_density']>0 for r in rows)
out=dict(registration=registration,rows=rows,all_signs_agree=all(r['response_sign_agrees'] for r in rows),
    maximum_quadrature_change=max(r['quadrature_change'] for r in rows),
    observational_scores=0,physical_exclusions=0)
(dest/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('All signs agree:',out['all_signs_agree'],'quadrature:',out['maximum_quadrature_change'])
print('Positive transfers:',sum(r['transfer_coefficients'][-1]>0 for r in rows),'negative:',sum(r['transfer_coefficients'][-1]<0 for r in rows))
