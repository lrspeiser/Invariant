"""Audit derivative information in the already exposed stellar source packet."""
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).parent / 'Invariant'
path = root / 'work/gravity-first-principles/xcop-pressure-002/source_preflight.json'
packet = json.loads(path.read_bytes())
rows = []
for p in packet['packets']:
    if p['stellar'] is None:
        continue
    s = p['stellar']
    r, m = np.array(s['radius_kpc']), np.array(s['mass_msun'])
    monotone = np.maximum.accumulate(m)
    rows.append(dict(cluster=p['cluster'], knots=len(r), intervals=len(r)-1,
        raw_flat_intervals=int(sum(np.diff(m)==0)), raw_decreasing_intervals=int(sum(np.diff(m)<0)),
        corrected_flat_intervals=int(sum(np.diff(monotone)==0)),
        corrected_mass_values=int(sum(monotone!=m)),
        largest_monotonic_correction=float(max(monotone/m-1)),
        available_stellar_columns=list(s),
        median_log_radius_spacing=float(np.median(np.diff(np.log(r)))),
        source_access=[a for a in p['access'] if a.get('role')=='stellar_mass']))
# Dimensionless witness, not a fitted cluster model. The smooth perturbation
# vanishes outside an interval containing no mass measurement.
# b(t)=256*t^4*(1-t)^4 has max b=1, |b'|<=16, b''(1/2)=-32.
# M=r+epsilon*w*b(t)/16 therefore has M'>=1-epsilon>0,
# while its second derivative at the midpoint is -2*epsilon/w.
witness = []
epsilon = .5
for width in [.1, .01, .001, .0001]:
    witness.append(dict(width=width, maximum_mass_perturbation=epsilon*width/16,
        lower_bound_mass_derivative=1-epsilon,
        midpoint_second_derivative=-2*epsilon/width,
        mass_at_measurement_radii_1_and_2=[1.,2.]))
out = dict(input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=rows,
    analytic_witness=witness,
    witness_scope='C3 compact perturbation inside (1,2), centered at 1.5, positive monotone mass; derivative non-identifiability without a smoothness bound. Not a cluster fit or a force-sign counterexample.',
    conclusion='The retained stellar packet contains enclosed mass and radius, but no stellar derivative uncertainty or covariance. Numerical source convergence does not establish physical derivative accuracy.',
    family_exclusions=0, new_observational_scores=0)
dest = root / 'work/gravity-first-principles/stellar-gradient-information-001'
dest.mkdir(exist_ok=False)
(dest/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
(dest/'audit.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(rows,indent=2))
