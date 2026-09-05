"""Grid-independent necessary positivity condition for spherical projection."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad

root=Path(__file__).parent/'Invariant'
base=root/'work/gravity-first-principles'
p=json.loads((base/'projected-stellar-packet-001/result.json').read_bytes())
rows=[]
for x in p['stellar_packets']:
    c=x['columns']; r,m,lo,hi=[np.array(c[k]) for k in ['RADIUS','MSTAR','MSTAR_LO','MSTAR_HI']]
    witnesses=[]
    for j in range(1,len(r)):
        q=r[:j]/r[j]
        fraction=q*q/(1+np.sqrt(1-q*q))
        bound=(fraction*lo[j]-hi[:j])/(m[:j]+fraction*m[j])
        i=int(np.argmax(bound))
        witnesses.append((float(bound[i]),i,j,float(fraction[i])))
    bound,i,j,f=max(witnesses)
    rows.append(dict(cluster=x['cluster'],inner_radius_kpc=float(r[i]),outer_radius_kpc=float(r[j]),
        necessary_inner_fraction=f,inner_mass_upper=float(hi[i]),outer_mass_lower=float(lo[j]),
        minimum_fractional_relaxation_lower_bound=max(0,bound),
        positive_spherical_projection_bracket_conflict=bound>1e-8))
# Integrate uniform-volume shells to recover a uniform sphere's cylinder mass.
errors=[]
for R in [.01,.1,.5,.99,1.,2.]:
    def integrand(s):
        u=min(R/s,1.)
        return 3*s*s*u*u/(1+np.sqrt(1-u*u))
    numeric=quad(integrand,0,1,points=[R] if R<1 else None,epsabs=1e-12,epsrel=1e-12)[0]
    exact=1-(1-R*R)**1.5 if R<1 else 1.
    errors.append(abs(numeric-exact))
assert max(errors)<1e-10
out=dict(rows=rows,uniform_sphere_absolute_errors=errors,
    condition='For any positive spherical shell distribution and R1<R2, Mproj(R1)>=B(R1/R2)*Mproj(R2), B(q)=1-sqrt(1-q²). Each shell obeys the inequality; nonnegative summation preserves it.',
    scope='Necessary geometric condition within simultaneous quoted brackets, not a statistical rejection or gravity-law exclusion.')
(base/'stellar-shell-feasibility-001/pair_certificates.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
(base/'stellar-shell-feasibility-001/certify.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(rows,indent=2))
