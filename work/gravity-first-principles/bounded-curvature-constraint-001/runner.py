"""Symbolic examples and registered-action checks for a bounded-curvature constraint."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import sympy as sp
from scipy.optimize import brentq
root=Path(__file__).parent/'Invariant'
sys.path.insert(0,str(root/'src'))
from invariant_gravity_extensions.length_screening import LengthScreening
dest=root/'work/gravity-first-principles/bounded-curvature-constraint-001'
dest.mkdir(exist_ok=False)
h=sp.symbols('h',nonnegative=True)
p=sp.symbols('p',positive=True)
examples=[]
for name,f in [('exponential',sp.exp(-h)),('inverse_power',(1+h)**(-p))]:
    B=sp.simplify(sp.diff(f,h)+2*h*sp.diff(f,h,2))
    expected=sp.exp(-h)*(2*h-1) if name=='exponential' else p*(1+h)**(-p-2)*((2*p+1)*h-1)
    assert sp.simplify(B-expected)==0
    examples.append(dict(name=name,f=str(f),B=str(expected),positive_when='h>1/2' if name=='exponential' else 'h>1/(2p+1)'))
rows=[]
for shape in [.5,1.,2.]:
    spec=LengthScreening(shape)
    def B(v):
        _,ph,_,k2,fraction=spec.partials(1.,v)
        return float(ph+2*v*fraction*k2/(1+v))
    grid=np.logspace(-8,8,513)
    pairs=[(a,b) for a,b in zip(grid[:-1],grid[1:]) if B(a)<=0<B(b)]
    assert pairs
    crossing=brentq(B,*pairs[0],xtol=1e-12)
    rows.append(dict(shape=shape,x=1.,first_sampled_negative_to_positive_root_h=crossing,
        B_below=B(crossing*.9),B_above=B(crossing*1.1),B_large=B(1e6)))
out=dict(examples=examples,registered_action=rows,
    theorem='A twice-differentiable function on the entire real line that is bounded below and concave everywhere must be constant. Applied to P at fixed gradient along H=H0+t*n*n, any bounded-below nonconstant dependence has positive second derivative somewhere.',
    implication='For the local Hessian action, that second derivative is proportional to the leading B coefficient. Global B<=0 for pressureless shortwave safety conflicts with nontrivial bounded screening on such a ray.',
    assumptions=['local instantaneous Hessian action with the derived matter coupling','twice differentiable constitutive function on the full ray','bounded below and nonconstant along that ray','pressureless continuum shortwave criterion imposed on every admissible background'],
    limitation='Conditional structural obstruction, not an exclusion of nonlocal, dynamical, restricted-domain or pressure-supported completions.',
    module_sha256=hashlib.sha256((root/'src/invariant_gravity_extensions/length_screening.py').read_bytes()).hexdigest(),
    physical_exclusions=0,observational_scores=0)
(dest/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
print(rows)
