"""Isolated one-dimensional Helmholtz Green operator and boundary checks."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.signal import fftconvolve

root=Path(__file__).parent/'Invariant'
dest=root/'work/gravity-first-principles/isolated-filter-001'
dest.mkdir(exist_ok=False)
registration=dict(kernel='exp(-abs(x-y)/L)/(2L)',lengths=[.3,1.,3.],
    spacings=[.04,.02,.01],half_box='max(12,8L) and twice that',
    quadrature='Exact kernel integrals per source cell, piecewise-constant source, zero extension and linear (not circular) FFT convolution.',
    scope='One-dimensional decaying test functions and isolated filter only; not a three-dimensional gravity solution or complete action-boundary admission.')
(dest/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
rows=[]
for L in registration['lengths']:
    probes=np.array([0.,1.,3.])
    ref=np.array([quad(lambda y:np.exp(-abs(x-y)/L)/(2*L)*np.exp(-y*y/2),-np.inf,x,epsabs=1e-12)[0]
        +quad(lambda y:np.exp(-abs(x-y)/L)/(2*L)*np.exp(-y*y/2),x,np.inf,epsabs=1e-12)[0] for x in probes])
    for box_factor in [1,2]:
        extent=max(12,8*L)*box_factor
        for dx in registration['spacings']:
            n=int(round(2*extent/dx))+1; x=np.linspace(-extent,extent,n)
            j=np.arange(-(n-1),n)
            distance=abs(j)*dx
            weights=.5*np.exp(-(distance-dx/2)/L)*(-np.expm1(-dx/L))
            weights[j==0]=-np.expm1(-dx/(2*L))
            def S(v):return fftconvolve(v,weights,mode='full')[n-1:2*n-1]
            f=np.exp(-x*x/2); g=np.exp(-(x-1.3)**2/3)*(1+.1*x)
            Sf,Sg=S(f),S(g)
            indices=np.rint((probes+extent)/dx).astype(int)
            error=float(np.max(abs(Sf[indices]-ref)/ref))
            left=float(np.dot(f,Sg)*dx); right=float(np.dot(Sf,g)*dx)
            # Analytic source derivative versus derivative of its filtered field.
            dSf=np.gradient(Sf,dx,edge_order=2)
            interior=abs(x)<3
            commutator=float(np.max(abs(dSf[interior]-S(-x*f)[interior])))
            rows.append(dict(L=L,half_box=extent,dx=dx,reference=ref.tolist(),
                predicted=Sf[indices].tolist(),maximum_relative_error=error,
                adjoint_relative_residual=abs(left-right)/max(abs(left),abs(right)),
                interior_derivative_commutator=commutator,
                minimum_filtered_positive_source=float(Sf.min())))
out=dict(registration=registration,rows=rows,observational_scores=0,physical_exclusions=0)
(dest/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('Fine-grid errors',[(r['L'],r['half_box'],r['maximum_relative_error']) for r in rows if r['dx']==.01])
print('Worst adjoint residual',max(r['adjoint_relative_residual'] for r in rows))
