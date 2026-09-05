"""Direct isolated flux response for the strongest negative compact control."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import roots_legendre,spherical_in
import sympy as sp
base=Path(__file__).parent; root=base/'Invariant'
sys.path.insert(0,str(root/'src'))
from invariant_gravity_extensions.length_screening import LengthScreening
source=root/'work/gravity-first-principles/positive-filtered-source-001'
parent=root/'work/gravity-first-principles/positive-compact-perturbations-001'
dest=root/'work/gravity-first-principles/compact-direct-force-001'
dest.mkdir(exist_ok=False)
tree=ast.parse((source/'runner.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['fields','quadrature']],type_ignores=[]),'source_functions','exec'))
d=json.loads((parent/'result.json').read_text())
card=min((r for r in d['rows'] if r['n']==1024),key=lambda r:r['response'])
L,ell,k,shape=[card[key] for key in ['L','ell','k','shape']]
spec=LengthScreening(shape)
registration=dict(parent_sha256=hashlib.sha256((parent/'result.json').read_bytes()).hexdigest(),
    selection='Minimum projected response in previous 72-case diagnostic; development follow-up, not independent held-out validation.',
    card=card,source_quadrature=[32,64],projection_quadrature=[128,256],
    probes=[.5,.9,1.,1.5,2.,2.5,3.,3.1,3.5,5.],relative_tolerance=1e-5,
    scope='Direct nonlinear flux central differences; no dynamical evolution or physical admission.')
(dest/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
z=sp.symbols('z',real=True)
expr=(1-z*z)**6*sp.cos(k*z)
df=[sp.lambdify(z,sp.diff(expr,z,j),'numpy') for j in range(5)]

def perturb(r):
    r=np.asarray(r); inside=(r>1)&(r<3)
    dd=[np.where(inside,f(r-2),0.) for f in df]
    v=dd[1]-L*L*(dd[3]+2*dd[2]/r-2*dd[1]/r**2)
    dv=dd[2]-L*L*(dd[4]+2*dd[3]/r-4*dd[2]/r**2+4*dd[1]/r**3)
    return v,dv,dd[2],dd[1]/r,dd[3],dd[2]/r-dd[1]/r**2

def flux_terms(r,alpha):
    bg=fields(r,1.,L); delta=perturb(r)
    p,dp,hr,ht,dhr,dht=[a+alpha*b for a,b in zip(bg,delta)]
    x=p*p; h=ell*ell*(hr*hr+2*ht*ht); u=x+h
    px,ph,k1,k2,f=spec.partials(x,h)
    dx=2*p*dp; dh=2*ell*ell*(hr*dhr+2*ht*dht)
    dph=((k1+f*k2)*dx+f*k2*dh)/u
    return (1+px)*p,dph*hr+ph*(dhr+2*dht)

def response_at(r,n,alpha):
    s,w=quadrature(r,L,n)
    first_plus=flux_terms(np.array([r]),alpha)[0][0]
    first_minus=flux_terms(np.array([r]),-alpha)[0][0]
    reaction_difference=flux_terms(s,alpha)[1]-flux_terms(s,-alpha)[1]
    return (first_plus-first_minus-ell*ell*np.dot(w,reaction_difference))/(2*alpha)

rows=[]
for nq in registration['projection_quadrature']:
    zq,wq=roots_legendre(nq); r=zq+2; v=perturb(r)[0]; measure=wq*r*r
    norm=float(np.dot(measure,v*v))
    for ns in registration['source_quadrature']:
        for check in card['checks']:
            alpha=check['alpha']
            response=np.array([response_at(x,ns,alpha) for x in r])
            projected=float(np.dot(measure,v*response)/norm)
            probes=[dict(r=x,newtonian_perturbation=float(perturb(np.array([x]))[0][0]),
                         gravity_perturbation=float(response_at(x,ns,alpha))) for x in registration['probes']]
            rows.append(dict(projection_nodes=nq,source_nodes=ns,c=check['c'],alpha=alpha,
                projected_response=projected,relative_error=abs(projected-card['response'])/max(1,abs(card['response'])),
                probes=probes))
    print(f'Completed projection order {nq}',flush=True)
worst=max(r['relative_error'] for r in rows)
out=dict(registration=registration,rows=rows,worst_relative_error=worst,
    passed=bool(worst<1e-5),observational_scores=0,admitted_candidates=0)
(dest/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(dict(worst_relative_error=worst,passed=out['passed'],last=rows[-1])))
