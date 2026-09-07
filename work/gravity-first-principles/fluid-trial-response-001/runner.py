"""Conditional barotropic-fluid Rayleigh quotients for compact perturbations."""
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
dest=root/'work/gravity-first-principles/fluid-trial-response-001'
dest.mkdir(exist_ok=False)
tree=ast.parse((source/'runner.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'source_functions','exec'))
data=json.loads((parent/'result.json').read_text())
registration=dict(parent_sha256=hashlib.sha256((parent/'result.json').read_bytes()).hexdigest(),
    matter='Inviscid barotropic Euler fluid, Newtonian inertial coupling, instantaneous registered static gravitational field.',
    equilibrium='Diagnostic source-specific P(r)=integral_r^infinity rho g dr; cs^2=Pprime/rhoprime=-rho g/rhoprime.',
    displacement='xi=-delta_psi_prime/(4 pi G rho), compact on 1<r<3.',
    numerator='-integral r^2 cs^2 delta_rho^2/rho dr + integral r^2 delta_psi_prime delta_g dr/(4 pi G).',
    denominator='integral r^2 rho xi^2 dr; angular factor 4pi cancels.',
    nodes=[256,512],source_nodes=64,comparison_tolerance=1e-5,
    scope='Conditional Rayleigh quotients for selected trial displacements. Source-specific EOS, not a universal law; global equilibrium and full spectrum not certified.')
(dest/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
z,kk=sp.symbols('z kk',real=True)
expr=(1-z*z)**6*sp.cos(kk*z)
df=[sp.lambdify((z,kk),sp.diff(expr,z,j),'numpy') for j in range(5)]
rows=[]
for n in registration['nodes']:
    zq,w=roots_legendre(n); r=zq+2; measure=w*r*r
    for L in [.1,.3,.5]:
        grids=[quadrature(x,L,64) for x in r]
        D=1+r*r; rhoP=3/(4*np.pi*D**2.5)
        rho=rhoP*(1-5*L*L*(4*r*r-3)/D**2)
        drho=-5*r*rhoP/D**3*(D*D+35*L*L-28*L*L*r*r)
        for shape in [.5,1.,2.]:
            spec=LengthScreening(shape)
            for ell in [1.,10.]:
                _,first,_=terms(r,1.,L,ell,spec)
                reaction=np.array([np.dot(weights,terms(s,1.,L,ell,spec)[2]) for s,weights in grids])
                g=first-ell*ell*reaction
                cs2=-rho*g/drho
                for k in [2.,8.,32.,128.]:
                    dd=[f(zq,k) for f in df]
                    v=dd[1]-L*L*(dd[3]+2*dd[2]/r-2*dd[1]/r**2)
                    delta_rho=(dd[2]+2*dd[1]/r-L*L*(dd[4]+4*dd[3]/r))/(4*np.pi)
                    xi=-v/(4*np.pi*rho)
                    norm=float(np.dot(measure,v*v))
                    static=next(row['response'] for row in data['rows'] if row['n']==1024 and row['L']==L and row['shape']==shape and row['ell']==ell and row['k']==k)
                    gravity=static*norm/(4*np.pi)
                    pressure=float(np.dot(measure,cs2*delta_rho**2/rho))
                    inertia=float(np.dot(measure,rho*xi*xi))
                    rows.append(dict(n=n,L=L,shape=shape,ell=ell,k=k,
                        gravity_term=gravity,pressure_restoring_term=pressure,inertia=inertia,
                        sigma_squared_trial=(gravity-pressure)/inertia,
                        minimum_sampled_cs_squared=float(cs2.min()),minimum_sampled_force=float(g.min())))
        print(f'Completed n={n}, L={L}',flush=True)
fine=[r for r in rows if r['n']==512]
coarse={(r['L'],r['shape'],r['ell'],r['k']):r for r in rows if r['n']==256}
err=max(abs(r['sigma_squared_trial']-coarse[(r['L'],r['shape'],r['ell'],r['k'])]['sigma_squared_trial'])/max(1,abs(r['sigma_squared_trial'])) for r in fine)
out=dict(registration=registration,rows=rows,refinement_error=err,passed=bool(err<1e-5),
    positive_trial_count=sum(r['sigma_squared_trial']>0 for r in fine),
    minimum_trial=min(r['sigma_squared_trial'] for r in fine),maximum_trial=max(r['sigma_squared_trial'] for r in fine),
    minimum_sampled_cs_squared=min(r['minimum_sampled_cs_squared'] for r in fine),
    observational_scores=0,admitted_candidates=0)
(dest/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ['registration','rows']}))
