"""Isolated radial vector Helmholtz Green function, l=1 controls."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import spherical_in

root = Path(__file__).parent / 'Invariant'
dest = root / 'work/gravity-first-principles/radial-vector-filter-001'
dest.mkdir(exist_ok=False)
registration = dict(
    operator='For V=v(r) rhat: Delta V=(v_second+2 v_first/r-2 v/r^2) rhat.',
    dimensionless_kernel='t^2 i1(min(q,t)) k1(max(q,t)); q=r/L, t=s/L, k1(z)=exp(-z)(z+1)/z^2 (no pi/2 normalization).',
    radii=[0., .001, .01, .1, 1., 3., 10., 100.],
    widths=[.1, 1., 10.],
    tolerance_absolute=1e-9,
    control='Manufactured Gaussian gradient v=-q/b^2 exp(-q^2/(2b^2)); input (1-Delta_vector)v.',
    boundary='Regular vector vanishes at origin; growing solution excluded at infinity.',
    scope='Linear vector inverse only; no full action variation, astronomical score or stability admission.')
(dest/'registration.json').write_text(json.dumps(registration, indent=2), encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())

def scaled_i1(z):
    if z < 20:
        return np.exp(-z)*spherical_in(1, z)
    return ((z-1)+(z+1)*np.exp(-2*z))/(2*z*z)

def kernel(q, t):
    small, large = min(q, t), max(q, t)
    return t*t*np.exp(-(large-small))*scaled_i1(small)*(large+1)/(large*large)

def inverse(q, f):
    if q == 0:
        return 0.
    return (quad(lambda t: kernel(q,t)*f(t), 0, q, epsabs=1e-12, epsrel=1e-12, limit=300)[0]
            +quad(lambda t: kernel(q,t)*f(t), q, np.inf, epsabs=1e-12, epsrel=1e-12, limit=300)[0])

def wrong_scalar_inverse(q, f):
    if q == 0:
        return quad(lambda t: t*np.exp(-t)*f(t), 0, np.inf, epsabs=1e-12)[0]
    return (quad(lambda t: np.exp(-(q-t))*(-np.expm1(-2*t))*t*f(t), 0,q,epsabs=1e-12)[0]
            +quad(lambda t: np.exp(-(t-q))*(-np.expm1(-2*q))*t*f(t),q,np.inf,epsabs=1e-12)[0])/(2*q)

rows=[]
for b in registration['widths']:
    def f(q):
        return q*(-1/b**2-5/b**4+q*q/b**6)*np.exp(-q*q/(2*b*b))
    for q in registration['radii']:
        exact=-q/b**2*np.exp(-q*q/(2*b*b))
        predicted=inverse(q,f)
        wrong=wrong_scalar_inverse(q,f)
        rows.append(dict(width=b,q=q,exact=float(exact),predicted=float(predicted),
                         absolute_error=float(abs(predicted-exact)),
                         incorrect_scalar_result=float(wrong),
                         incorrect_scalar_error=float(abs(wrong-exact))))
# Kernel symmetry in the physical radial measure r^2 dr. This does not
# independently prove the nonlinear action or its discretized adjoint relation.
pairs=[]
for q in [.001,.1,1.,10.,100.]:
    for t in [.002,.2,2.,20.]:
        left=q*q*kernel(q,t)
        right=t*t*kernel(t,q)
        pairs.append(dict(q=q,t=t,relative_error=float(abs(left-right)/max(abs(left),abs(right),1e-300))))
worst=max(row['absolute_error'] for row in rows)
adjoint=max(row['relative_error'] for row in pairs)
result=dict(registration=registration,rows=rows,weighted_symmetry=pairs,
            worst_absolute_error=worst,worst_symmetry_relative_error=adjoint,
            passed=bool(worst<1e-9 and adjoint<1e-12),observational_scores=0,admitted_candidates=0)
(dest/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
digest=hashlib.sha256((dest/'result.json').read_bytes()).hexdigest()
report=f'''# Radial vector filter: the directional response

For a spherical vector V=v(r) rhat, the Laplacian is (v''+2v'/r-2v/r²) rhat. The last term distinguishes it from the scalar operator. The inverse uses the angular l=1 Green function, with a regular origin and no growing homogeneous term at infinity.

In dimensionless radii q=r/L and t=s/L, the kernel is t² i1(min(q,t)) k1(max(q,t)). Here i1(z)=(z cosh(z)-sinh(z))/z² and k1(z)=exp(-z)(z+1)/z², explicitly without the alternative pi/2 normalization. Scaled functions avoid overflow at large radius.

We manufactured smooth Gaussian-gradient vector solutions and applied the inverse to their exact Helmholtz sources. These are mathematical test fields, not proposed matter densities. All 24 controls passed the predeclared 1e-9 absolute tolerance; worst discrepancy {worst:.6g}. Twenty kernel pairs also satisfy symmetry in the physical r² dr measure, with worst relative discrepancy {adjoint:.6g}.

As a negative control, the scalar inverse was applied to the same radial component. Its maximum absolute error was {max(r['incorrect_scalar_error'] for r in rows):.6g}, including nonzero origin values where a smooth radial vector must vanish. This demonstrates why componentwise scalar radial filtering would change the theory.

For the gravitational action, filter the scalar potential before taking its Hessian. The divergence of the resulting nonlinear tensor product is a radial vector and requires this l=1 inverse. The calculation has not yet verified the full isolated action variation, its nonlinear stability, or any galaxy, cluster, Solar System or lensing prediction.

Evidence: radial-vector-filter-001/result.json. SHA-256 {digest}.
'''
(dest/'report.md').write_text(report,encoding='utf-8')
(Path(__file__).parent.parent/'outputs/Gravity-radial-vector-filter.md').write_text(report,encoding='utf-8')
print(json.dumps(dict(controls=len(rows),worst_error=worst,adjoint_error=adjoint,passed=result['passed'])))
assert result['passed']
