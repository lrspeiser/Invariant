"""Smooth shell source controls; no cluster fit or observational prediction."""
import json
from pathlib import Path
import numpy as np
import sympy as sp
import mpmath as mp
from scipy.integrate import quad

root=Path(__file__).parent/'Invariant'
dest=root/'work/gravity-first-principles/smooth-shell-controls-001'
dest.mkdir(exist_ok=False)
r,s,a=sp.symbols('r s a',positive=True)
potential=-2/(sp.sqrt((r-s)**2+a*a)+sp.sqrt((r+s)**2+a*a))
expressions=[sp.diff(potential,r,n) for n in range(4)]
evaluate=sp.lambdify((r,s,a),expressions,'numpy',cse=True)
registration=dict(potential='-2 GM/[sqrt((r-s)^2+a^2)+sqrt((r+s)^2+a^2)]',
    interpretation='Uniform spherical shell of Plummer centers, each with scale a; positive convolution, finite mass, smooth for a>0.',
    shell_radii=[1.,10.],width_ratios=[.02,.2,1.],radius_ratios=[.001,.1,.5,1.,2.,10.,100.],
    derivative_relative_target=1e-8,poisson_scaled_target=1e-8,
    scope='Exact generative source prototype; no source width inferred, no gravity family admission.')
(dest/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
mp.mp.dps=70
rows=[]
for shell in registration['shell_radii']:
    for ratio in registration['width_ratios']:
        width=shell*ratio
        for rr in registration['radius_ratios']:
            radius=shell*rr
            values=np.array(evaluate(radius,shell,width),float)
            sm,am,rm=map(mp.mpf,map(str,[shell,width,radius]))
            def phi(x):
                return -2/(mp.sqrt((x-sm)**2+am*am)+mp.sqrt((x+sm)**2+am*am))
            reference=np.array([float(mp.diff(phi,rm,n)) for n in range(4)])
            errors=abs(values-reference)/np.maximum(abs(reference),1e-300)
            # Independent angular average of positive Plummer density.
            rho=quad(lambda mu:3*width**2/(8*np.pi)*(radius**2+shell**2+width**2-2*radius*shell*mu)**-2.5,
                     -1,1,epsabs=1e-14,epsrel=1e-11)[0]
            poisson=values[2]+2*values[1]/radius
            residual=abs(poisson-4*np.pi*rho)/max(abs(values[2]),abs(2*values[1]/radius),4*np.pi*rho)
            rows.append(dict(shell_radius=shell,width=width,radius=radius,
                potential_gradient_first_second=values.tolist(),derivative_relative_errors=errors.tolist(),
                density_from_positive_convolution=rho,poisson_scaled_residual=float(residual),
                all_finite=bool(np.all(np.isfinite(values))),
                derivative_pass=bool(np.max(errors)<1e-8),poisson_pass=bool(residual<1e-8)))
out=dict(registration=registration,rows=rows,passed=sum(x['derivative_pass'] and x['poisson_pass'] for x in rows),
    cases=len(rows),observational_scores=0,physical_exclusions=0)
(dest/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('Passed',out['passed'],'/',out['cases'])
print('Worst derivative relative error',max(max(x['derivative_relative_errors']) for x in rows))
print('Worst Poisson scaled residual',max(x['poisson_scaled_residual'] for x in rows))
