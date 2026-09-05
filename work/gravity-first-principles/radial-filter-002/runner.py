"""Independent exact controls for the isolated scalar 3D Helmholtz inverse."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad

root = Path(__file__).parent / 'Invariant'
dest = root / 'work/gravity-first-principles/radial-filter-002'
dest.mkdir(exist_ok=False)
registration = {
    'operator': 'S=(1-L^2 Laplacian)^(-1), L>0, isolated 3D scalar field',
    'radial_kernel': '[exp(-abs(r-s)/L)-exp(-(r+s)/L)] s/(2 L r)',
    'boundary': 'u=r Sf vanishes at r=0; no growing homogeneous solution at infinity',
    'dimensionless_radii': [0., .001, .01, .1, 1., 3., 10., 100.],
    'gaussian_width_over_L': [.1, 1., 10.],
    'absolute_error_threshold_normalized_fields': 1e-9,
    'scope': 'Scalar radial inverse only. No vector/tensor componentwise filtering, full action solution, observations, or stability admission.',
}
(dest / 'registration.json').write_text(json.dumps(registration, indent=2), encoding='utf-8')
(dest / 'runner.py').write_bytes(Path(__file__).read_bytes())

def inverse(q, f):
    # q=r/L, t=s/L. The r=0 limit is integrated explicitly.
    if q == 0:
        return quad(lambda t: t*np.exp(-t)*f(t), 0, np.inf,
                    epsabs=1e-12, epsrel=1e-12, limit=300)[0]
    left = quad(lambda t: np.exp(-(q-t))*(-np.expm1(-2*t))*t*f(t),
                0, q, epsabs=1e-12, epsrel=1e-12, limit=300)[0]
    right = quad(lambda t: np.exp(-(t-q))*(-np.expm1(-2*q))*t*f(t),
                 q, np.inf, epsabs=1e-12, epsrel=1e-12, limit=300)[0]
    return (left+right)/(2*q)

rows = []
for q in registration['dimensionless_radii']:
    # Potential normalized by GM/L. Singular input, finite filtered origin.
    predicted = inverse(q, lambda t: -1/t)
    exact = -1. if q == 0 else np.expm1(-q)/q
    rows.append(dict(control='Newtonian point potential', q=q,
                     predicted=predicted, exact=exact, absolute_error=abs(predicted-exact)))
for b in registration['gaussian_width_over_L']:
    # Manufactured smooth solution: f=(1-Delta_q) exp(-q^2/(2 b^2)).
    # Input is signed, a mathematical operator control rather than a mass density.
    def f(t):
        return (1+3/b**2-t*t/b**4)*np.exp(-t*t/(2*b*b))
    for q in registration['dimensionless_radii']:
        predicted = inverse(q, f)
        exact = np.exp(-q*q/(2*b*b))
        rows.append(dict(control='manufactured smooth Gaussian', width_over_L=b,
                         q=q, predicted=predicted, exact=exact, absolute_error=abs(predicted-exact)))
worst = max(r['absolute_error'] for r in rows)
result = dict(registration=registration, rows=rows, worst_absolute_error=worst,
              passed=bool(worst < registration['absolute_error_threshold_normalized_fields']),
              observational_scores=0, admitted_candidates=0)
payload = json.dumps(result, indent=2)
(dest / 'result.json').write_text(payload, encoding='utf-8')
digest = hashlib.sha256((dest / 'result.json').read_bytes()).hexdigest()
report = f'''# Isolated three-dimensional scalar filter

The radial Green function of S=(1-L² Laplacian)^(-1) contains an image subtraction:

    Sf(r) = integral_0^infinity [exp(-|r-s|/L)-exp(-(r+s)/L)] s f(s) ds / (2 L r).

This follows by setting u=r Sf, imposing u(0)=0 and excluding the exponentially growing solution at infinity. At the origin the limiting kernel is s exp(-s/L)/L². The underlying three-dimensional Green function is exp(-r/L)/(4 pi L² r).

For f=-GM/r the exact result is -GM(1-exp(-r/L))/r, with finite origin value -GM/L. Its radial cusp at the origin reflects the singular input; this is not a smooth-source gravity admission.

An independent smooth manufactured solution uses Sf=exp(-r²/(2 a²)), with f=(1-L² Laplacian)Sf. That input is signed and is an operator control, not a proposed mass distribution. Widths a/L=0.1, 1, 10 test narrow and broad structure. Quadrature over the full half-line tests radii r/L=0 through 100 without a finite outer box.

All {len(rows)} controls passed the predeclared 1e-9 absolute tolerance for normalized fields. Worst absolute discrepancy: {worst:.6g}. These are scalar inverse checks, not a verification of the full nonlinear gravitational action.

Implementation constraint: a radial vector has an additional -2/r² term in its radial Laplacian. Applying this scalar kernel separately to radial vector or tensor components would implement the wrong operator. For the action's filtered Hessian, first filter the scalar potential and then differentiate it; the outer vector response needs its appropriate angular sector or an equivalent Cartesian treatment.

Next: verify that outer vector filter and the isolated action variation before making spherical-source or astronomical predictions. No observations were scored and no candidate was admitted.

Evidence: radial-filter-002/result.json; SHA-256 {digest}. The predecessor radial-filter-001 stopped during JSON serialization of a NumPy boolean; its registration and runner are preserved. The successor changes only that serialization and output directory.
'''
(dest / 'report.md').write_text(report, encoding='utf-8')
(Path(__file__).parent.parent / 'outputs/Gravity-radial-filter-checks.md').write_text(report, encoding='utf-8')
print(json.dumps(dict(controls=len(rows), worst_absolute_error=worst, passed=result['passed'], sha256=digest)))
assert result['passed']
