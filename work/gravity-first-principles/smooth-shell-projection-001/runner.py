"""Two independent smooth-shell projected-mass integrals; no source fit."""
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad

def offset_fraction(R,a,d):
    R,a,d=np.broadcast_arrays(R,a,d)
    D=np.sqrt(((R-d)**2+a*a)*((R+d)**2+a*a))
    X=R*R-a*a-d*d
    return np.where(X<0,2*a*a*R*R/(D*(D-X)),.5*(1+X/D))

def projected(R,s,a,n):
    q,w=leggauss(n)
    # Both hemispheres have the same projected center offset.
    mu=(q+1)/2
    return float(np.sum(w/2*offset_fraction(R,a,s*np.sqrt(1-mu*mu))))

def rho(r,s,a):
    if r==0:
        return 3*a*a/(4*np.pi*(s*s+a*a)**2.5)
    A=(r-s)**2+a*a
    # Stable analytic angular average of Plummer density.
    difference=-np.expm1(-1.5*np.log1p(4*r*s/A))
    return a*a/(8*np.pi*r*s)*A**-1.5*difference

def density_projection(R,s,a):
    inside=quad(lambda r:4*np.pi*r*r*rho(r,s,a),0,R,
                points=[s] if s<R else None,epsabs=1e-13,epsrel=1e-11)
    outside=quad(lambda u:4*np.pi*R**3*rho(R/u,s,a)/(u*u*(1+np.sqrt(1-u*u))),0,1,
                 points=[R/s] if R<s else None,epsabs=1e-13,epsrel=1e-11)
    return inside[0]+outside[0],inside[1]+outside[1]

root=Path(__file__).parent/'Invariant'
dest=root/'work/gravity-first-principles/smooth-shell-projection-001'
dest.mkdir(exist_ok=False)
registration=dict(shell_radius=1.,widths=[.02,.1,.2],
    radii=[.001,.01,.1,.5,.9,1.,1.1,2.,10.,100.],nodes=[64,128,256],
    relative_target=1e-7,scope='Compare orientation-averaged offset Plummer apertures with direct cylinder integration of the spherical convolution density. No cluster source or gravity admission.')
(dest/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
rows=[]
for a in registration['widths']:
    for R in registration['radii']:
        reference,error=density_projection(R,1.,a)
        values=[projected(R,1.,a,n) for n in registration['nodes']]
        discrepancies=[abs(v/reference-1) for v in values]
        rows.append(dict(width=a,radius=R,density_projection=reference,
            adaptive_error_estimate=error,orientation_projection=values,
            relative_discrepancies=discrepancies,passes=discrepancies[-1]<1e-7))
out=dict(registration=registration,rows=rows,passed=sum(r['passes'] for r in rows),
         cases=len(rows),new_gravity_scores=0,physical_exclusions=0)
(dest/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('Passed',out['passed'],'/',out['cases'])
print('Worst',max(rows,key=lambda r:r['relative_discrepancies'][-1]))
