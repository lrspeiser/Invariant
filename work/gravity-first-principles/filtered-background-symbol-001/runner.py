"""Locate force weakening and check frozen shortwave coefficients."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
base=Path(__file__).parent; root=base/'Invariant'
sys.path.insert(0,str(root/'src'))
from invariant_gravity_extensions.length_screening import LengthScreening
parent=root/'work/gravity-first-principles/positive-filtered-source-001'
dest=root/'work/gravity-first-principles/filtered-background-symbol-001'
dest.mkdir(exist_ok=False)
data=json.loads((parent/'result.json').read_text())
tree=ast.parse((parent/'runner.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fields'],type_ignores=[]),'frozen_fields','exec'))
registration=dict(parent_sha256=hashlib.sha256((parent/'result.json').read_bytes()).hexdigest(),
    symbol='T(k,n)=A(n)+ell^2 B(n) k^2/(1+L^2 k^2)^2 in the frozen-coefficient approximation.',
    A='Px+2 Pxx (p dot n)^2; Pxx=2 Kprime+x Ksecond',
    B='Ph+2 ell^2 Phh (n dot Hchi dot n)^2; Phh=x Ksecond',
    directions='radial and tangential',
    limit='T tends to A as k tends to infinity. This is a static response, not a complete dynamical dispersion relation.',
    derivative_control_relative_tolerance=1e-6,
    scope='Frozen coefficients only; finite-k predictions need a locally uniform background across filter length. No general stability admission.')
(dest/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
r=np.array(data['registration']['radii'])
rows=[]; weak=[]; derivative_errors=[]
for card in data['rows']:
    L,M,ell,shape=[card[k] for k in ['L','M','ell','shape']]
    spec=LengthScreening(shape)
    p,_,hr,ht,_,_=fields(r,M,L)
    x=p*p; h=ell*ell*(hr*hr+2*ht*ht); u=x+h
    k,k1,k2,_=spec.kernel(u)
    px,ph,_,_,fraction=spec.partials(x,h)
    A_rad=1+px+2*fraction*(2*k1+fraction*k2)
    A_tan=1+px
    step=1e-5*p
    def first_flux(v):
        return (1+spec.partials(v*v,h)[0])*v
    numeric=(first_flux(p+step)-first_flux(p-step))/(2*step)
    derivative_errors.extend((abs(numeric-A_rad)/np.maximum(abs(A_rad),1)).tolist())
    force=np.array(card['force64'])
    first=(1+px)*p
    for j,radius in enumerate(r):
        ratio=float(force[j]/p[j])
        if ratio<1:
            weak.append(dict(L=L,M=M,ell=ell,shape=shape,r=float(radius),force_ratio=ratio,
                first_term_ratio=float(first[j]/p[j]),reaction_term_ratio=float((force[j]-first[j])/p[j])))
        for direction,A,H in [('radial',A_rad,hr),('tangential',A_tan,ht)]:
            B=ph[j]+2*ell*ell*fraction[j]*k2[j]*H[j]**2/u[j]
            correction=ell*ell*B/(4*L*L)
            rows.append(dict(L=L,M=M,ell=ell,shape=shape,r=float(radius),direction=direction,
                A=float(A[j]),B=float(B),frozen_minimum_T=float(A[j]+min(0,correction)),
                frozen_maximum_T=float(A[j]+max(0,correction))))
weak.sort(key=lambda row:row['force_ratio'])
out=dict(registration=registration,rows=rows,weakening_samples=weak,
    derivative_control_worst_relative_error=max(derivative_errors),
    derivative_control_passed=bool(max(derivative_errors)<1e-6),
    minimum_A=min(row['A'] for row in rows),maximum_A=max(row['A'] for row in rows),
    negative_A_count=sum(row['A']<0 for row in rows),
    negative_frozen_T_count=sum(row['frozen_minimum_T']<0 for row in rows),
    observational_scores=0,admitted_candidates=0)
(dest/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({key:val for key,val in out.items() if key not in ['registration','rows','weakening_samples']}))
print('Strongest weakening',json.dumps(weak[0] if weak else None))
