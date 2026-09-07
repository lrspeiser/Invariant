"""Independent theory arithmetic, without importing the implementation."""
import hashlib
import json
import math
from pathlib import Path
from scipy.integrate import quad

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
saved = json.loads((PACKAGE / 'run001/summary.json').read_text(encoding='utf-8'))
errors = []
for row in saved['constant_environment_cases']:
    # Algebraically independent rational expression: epsilon=(1+5u)/(5+5u).
    lab = (1 + 5*row['u_lab'])/(5 + 5*row['u_lab'])
    env = (1 + 5*row['u_environment'])/(5 + 5*row['u_environment'])
    errors.append(abs(lab/env - row['relative_to_measured_newton']))
    # Phi=-G_bare*M/(epsilon*r), g_r=-dPhi/dr.
    r, step = 2., 1.e-4
    phi = lambda x: -lab/(env*x)
    derivative = (phi(r-2*step)-8*phi(r-step)+8*phi(r+step)-phi(r+2*step))/(12*step)
    assert abs(derivative*r*r-lab/env) < 1.e-10
    if row['u_lab'] == row['u_environment']:
        assert abs(lab/env-1) < 1.e-15
assert len(errors) == 36 and max(errors) < 1.e-14

# Separable three-dimensional normalization, not radial integration.
one_d, estimate = quad(lambda x: math.exp(-x*x/2)/math.sqrt(2*math.pi),
                       -math.inf, math.inf, epsabs=1.e-13, epsrel=1.e-13)
assert abs(one_d**3-1) < 1.e-12 and estimate < 1.e-12
bounds = []
for row in saved['isolated_source_bounds']:
    ell = row['ell_kpc']
    peak = (1/(math.sqrt(2*math.pi)*ell))**3/1.e7
    maximum = 4*peak/(5+5*peak)
    assert math.isclose(peak, row['peak_u'], rel_tol=1.e-14)
    assert math.isclose(maximum, row['maximum_epsilon_increment'], rel_tol=1.e-14)
    # Exact perturbation for nonnegative background b is bounded by b=0.
    for background in [0, .01, .1, 1, 10, 100]:
        delta = .8*peak/((1+background)*(1+background+peak))
        assert 0 < delta <= maximum*(1+1.e-14)
    bounds.append({'ell_kpc':ell,'peak_u':peak,'maximum_epsilon_increment':maximum})
assert math.isclose(bounds[0]['peak_u']/bounds[1]['peak_u'], 8., rel_tol=1.e-14)
result = {'status':'PASS_INDEPENDENT_THEORY_REVIEW', 'observational_tests':0,
          'constant_environment_cases':len(errors),'maximum_ratio_absolute_error':max(errors),
          'separable_gaussian_integral':one_d**3,'one_dimensional_quadrature_error_estimate':estimate,
          'bounds':bounds,'physical_G_calibration_resolved':False,
          'reviewed_summary_sha256':hashlib.sha256((PACKAGE/'run001/summary.json').read_bytes()).hexdigest()}
(HERE/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
