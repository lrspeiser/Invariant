"""Separate interpolation arithmetic error from rounded source samples."""
import argparse
import json
import runpy
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp

repo = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
t = runpy.run_path(str(repo / 'tests/test_gravity_tensor_potential.py'))
R, Z = t['R'], t['Z']
expr = 17+R**2/5+3*Z**2/10+7*R**2*Z**2/10-13*R**4*Z**2/100+R**6*Z**4/20+R**4*Z**6/100
r, z = np.array([0., .4, 1., 2.3]), np.array([0., .3, .9, 1.7])
data = t['table'](expr, r, z)
mp.mp.dps = 70
left = [[1,0,0,0,-35,84,-70,20], [0,1,0,0,-20,45,-36,10],
        [0,0,mp.mpf(1)/2,0,-5,10,-mp.mpf(15)/2,2],
        [0,0,0,mp.mpf(1)/6,-mp.mpf(2)/3,1,-mp.mpf(2)/3,mp.mpf(1)/6]]
dr, dz, q = mp.mpf(float(r[1])), mp.mpf(float(z[1])), mp.mpf(.23)/mp.mpf(float(z[1]))
def axis_limit(a):
    total = mp.mpf(0)
    for e in range(2):
        for i in range(4):
            c4 = left[i][4] if e == 0 else sum((-1)**(i+4)*mp.binomial(n,4)*left[i][n] for n in range(4,8))
            for f in range(2):
                for j in range(4):
                    qq = q if f == 0 else 1-q
                    basis = sum(mp.mpf(c)*qq**n for n,c in enumerate(left[j])) * ((-1)**j if f else 1)
                    total += c4*dr**i*basis*dz**j*mp.mpf(float(a[i,j,e,f]))
    return 8*total/dr**4
expected = sp.diff(sp.cancel(sp.diff(expr,R)/R),R,2).subs({R:0,Z:sp.Rational(float(.23))})
exact = mp.mpf(str(expected.evalf(70)))
actual = t['packed'](t['C3TensorPotential'](r,z,data).fields([1e-14],[.23]))[11,0]/1e-14
accurate = np.array([[[[float(sp.diff(expr,R,i,Z,j).subs({R:sp.Rational(float(rr)), Z:sp.Rational(float(zz))}).evalf(70)) for zz in z] for rr in r] for j in range(4)] for i in range(4)])
record = {'scope':'near-axis TRpp/R for fixed symbolic control; exact arithmetic on stored float samples',
          'symbolic':float(exact), 'implementation':float(actual), 'exact_interpolant_of_numpy_samples':float(axis_limit(data)),
          'exact_interpolant_of_correctly_rounded_samples':float(axis_limit(accurate)),
          'implementation_error_vs_interpolant':float(mp.mpf(float(actual))-axis_limit(data)),
          'sample_error_vs_symbolic':float(axis_limit(data)-exact)}
path = args.output
if path.exists():
    raise FileExistsError(path)
path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
print(json.dumps(record,indent=2))
