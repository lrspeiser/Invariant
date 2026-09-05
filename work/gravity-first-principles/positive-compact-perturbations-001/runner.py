"""Mass-conserving positive-source perturbations: exact action Hessian."""
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy.special import roots_legendre
import sympy as sp
base=Path(__file__).parent; root=base/'Invariant'
sys.path.insert(0,str(root/'src'))
from invariant_gravity_extensions.length_screening import LengthScreening
parent=root/'work/gravity-first-principles/positive-filtered-source-001'
dest=root/'work/gravity-first-principles/positive-compact-perturbations-001'
dest.mkdir(exist_ok=False)
tree=ast.parse((parent/'runner.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fields'],type_ignores=[]),'positive_fields','exec'))
registration=dict(L=[.1,.3,.5],ell=[1.,10.],shapes=[.5,1.,2.],k=[2.,8.,32.,128.],
    M=1.,a=1.,G=1.,a0=1.,quadrature_nodes=[512,1024],
    perturbation='delta_chi=(1-(r-2)^2)^6 cos(k(r-2)) for 1<r<3, zero outside; delta_psi=(1-L^2 Delta)delta_chi.',
    amplitudes='alpha = c rho_lower / rigorous_abs_delta_rho_bound, c=0.1 and 0.01',
    positivity='For both signs rho_total >= (1-c)rho_background everywhere.',
    quantity='Integral delta_grad_psi dot delta_J / integral |delta_grad_psi|^2, using exact full-space action Hessian.',
    quadrature_relative_tolerance=1e-6,finite_derivative_relative_tolerance=1e-5,
    scope='Static directional response of compact positive-source perturbations. Not a dynamical eigenvalue, equilibrium test, or proof of stability.',
    parent_sha256=hashlib.sha256((parent/'result.json').read_bytes()).hexdigest())
(dest/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
z,kk=sp.symbols('z kk',real=True)
poly=(1-z*z)**6
derivatives=[sp.lambdify((z,kk),sp.diff(poly*sp.cos(kk*z),z,j),'numpy') for j in range(5)]
polynomial_bounds=[float(sum(abs(c) for c in sp.Poly(sp.diff(poly,z,j),z).all_coeffs())) for j in range(5)]
rows=[]
for n in registration['quadrature_nodes']:
    zq,w=roots_legendre(n); r=zq+2; measure=w*r*r
    for k in registration['k']:
        d=[f(zq,k) for f in derivatives]
        # |z|<=1, r>=1, and trigonometric factors bounded by one.
        B=[sum(math.comb(j,m)*polynomial_bounds[m]*k**(j-m) for m in range(j+1)) for j in range(5)]
        for L in registration['L']:
            p,_,hr,ht,_,_=fields(r,1.,L)
            v=d[1]-L*L*(d[3]+2*d[2]/r-2*d[1]/r**2)
            vr=d[2]; vt=d[1]/r
            density_delta=(d[2]+2*d[1]/r-L*L*(d[4]+4*d[3]/r))/(4*np.pi)
            density_bound=(B[2]+2*B[1]+L*L*(B[4]+4*B[3]))/(4*np.pi)
            lower=3/(4*np.pi*10**2.5)*(1-20*L*L/7)
            rho=3/(4*np.pi*(1+r*r)**2.5)*(1-5*L*L*(4*r*r-3)/(1+r*r)**2)
            norm=float(np.dot(measure,v*v))
            for shape in registration['shapes']:
                spec=LengthScreening(shape)
                for ell in registration['ell']:
                    x=p*p; h=ell*ell*(hr*hr+2*ht*ht); u=x+h
                    px,ph,k1,k2,f=spec.partials(x,h)
                    pxx=(2*k1+f*k2)/u; pxh=(k1+f*k2)/u; phh=f*k2/u
                    contraction=hr*vr+2*ht*vt
                    first=(1+px+2*pxx*p*p)*v*v
                    cross=4*ell*ell*pxh*p*v*contraction
                    curvature=ell*ell*ph*(vr*vr+2*vt*vt)+2*ell**4*phh*contraction**2
                    response=float(np.dot(measure,first+cross+curvature)/norm)
                    checks=[]
                    for c in [.1,.01]:
                        alpha=c*lower/density_bound
                        terms=[]
                        for sign in [1,-1]:
                            pp=p+sign*alpha*v; rr=hr+sign*alpha*vr; tt=ht+sign*alpha*vt
                            ex,hp,*_=spec.partials(pp*pp,ell*ell*(rr*rr+2*tt*tt))
                            terms.append((1+ex)*pp*v+ell*ell*hp*(rr*vr+2*tt*vt))
                        finite=float(np.dot(measure,terms[0]-terms[1])/(2*alpha*norm))
                        checks.append(dict(c=c,alpha=alpha,finite_response=finite,
                            relative_error=abs(finite-response)/max(1,abs(response)),
                            sampled_minimum_density_ratio=float(np.min(1-alpha*abs(density_delta)/rho)),
                            guaranteed_minimum_density_ratio=1-c))
                    rows.append(dict(n=n,k=k,L=L,ell=ell,shape=shape,response=response,
                        first_gradient_projection=float(np.dot(measure,first)/norm),
                        mixed_projection=float(np.dot(measure,cross)/norm),
                        curvature_projection=float(np.dot(measure,curvature)/norm),checks=checks))
    print(f'Completed {n} nodes',flush=True)
coarse={(r['k'],r['L'],r['ell'],r['shape']):r for r in rows if r['n']==512}
fine=[r for r in rows if r['n']==1024]
refine=max(abs(r['response']-coarse[(r['k'],r['L'],r['ell'],r['shape'])]['response'])/max(1,abs(r['response'])) for r in fine)
fd=max(c['relative_error'] for r in fine for c in r['checks'])
out=dict(registration=registration,rows=rows,quadrature_error=refine,finite_derivative_error=fd,
    passed=bool(refine<1e-6 and fd<1e-5),minimum_response=min(r['response'] for r in fine),
    maximum_response=max(r['response'] for r in fine),negative_responses=sum(r['response']<0 for r in fine),
    observational_scores=0,admitted_candidates=0)
(dest/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ['registration','rows']}))
