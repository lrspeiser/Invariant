import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import roots_legendre,spherical_in
base=Path(__file__).parent; root=base/'Invariant'
sys.path.insert(0,str(root/'src'))
from invariant_gravity_extensions.length_screening import LengthScreening
dest=root/'work/gravity-first-principles/positive-filtered-source-001'
data=json.loads((dest/'result.json').read_text())
# Load only function definitions, without rerunning the frozen campaign.
tree=ast.parse((dest/'runner.py').read_text(encoding='utf-8'))
definitions=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[])
exec(compile(definitions,'frozen-positive-source-functions','exec'))
rows=[]
radii=np.array(data['registration']['radii'])
for L in data['registration']['L']:
    grids=[quadrature(r,L,64,tail=80) for r in radii]
    for parent in [r for r in data['rows'] if r['L']==L]:
        M=parent['M']; ell=parent['ell']; spec=LengthScreening(parent['shape'])
        p,first,_=terms(radii,M,L,ell,spec)
        reaction=np.array([np.dot(w,terms(s,M,L,ell,spec)[2]) for s,w in grids])
        force=first-ell*ell*reaction
        error=float(np.max(abs(force-np.array(parent['force64']))/np.maximum(abs(force),p)))
        rows.append(dict(L=L,M=M,ell=ell,shape=parent['shape'],tail_doubling_error=error))
tail=dict(rows=rows,worst_error=max(r['tail_doubling_error'] for r in rows),
    scope='Upper integration distance changed from r+40L to r+80L at all registered samples. Sensitivity control, not rigorous tail bound.')
(dest/'tail_check.json').write_text(json.dumps(tail,indent=2),encoding='utf-8')
(dest/'tail_runner.py').write_bytes(Path(__file__).read_bytes())
minimum=min(r['minimum_force_over_newtonian'] for r in data['rows'])
hashes={name:hashlib.sha256((dest/name).read_bytes()).hexdigest() for name in ['result.json','tail_check.json']}
report=f'''# First finite positive-source test of the filtered action

All 81 registered source/parameter combinations produced inward force at all 41 sampled radii. This is a limited positive result: the force calculation behaves sensibly on this source family, but it has not passed a stability test or astronomical comparison.

We choose an exactly filtered Plummer potential chi=-GM/sqrt(r²+a²), and construct the actual Newtonian potential psi=chi-L² Laplacian chi. Its source density is

    rho = rho_Plummer * [1 - 5 L² (4r²-3a²)/(r²+a²)²].

The bracket is bounded below by 1-20L²/(7a²), so the source is positive everywhere when L/a<sqrt(7/20). It is smooth, has finite total mass M, and has an isolated potential. For L/a=0.1, 0.3, 0.5 the lower bounds are respectively 0.9714, 0.7429, and 0.2857. These are exact positivity guarantees, unlike the sampled force check.

The scan uses G=a=a0=1, masses 0.01, 1, 100, three kernel shapes, curvature lengths ell=0.1, 1, 10, and the three filter lengths above. Source construction depends on L. Consequently this is a diagnostic family, not a comparison of different L values against the same observed object. The exact construction avoids numerical uncertainty in the inner scalar filter; the nonlinear outer vector filter is integrated with the previously checked l=1 kernel.

Force was sampled at 41 logarithmically spaced radii from 0.01a to 100a. The smallest force/Newtonian-force ratio was {minimum:.8g}. No sampled force pointed outward. Doubling the quadrature order gave a worst difference of {data['worst_refinement_error']:.8g}, scaled by the larger of the final force magnitude and Newtonian force. Doubling the upper integration distance from r+40L to r+80L changed the result by at most {tail['worst_error']:.8g} on the same scale. These convergence tests do not establish a rigorous error bound or behavior at unsampled radii.

Next: test perturbations and the high-frequency response around these positive sources. An inward background force alone does not establish stable dynamics. The global-parameter galaxy/cluster/Solar System and light-bending requirements remain open; no observations were scored and no candidate was admitted.

Evidence: positive-filtered-source-001. Result SHA-256 {hashes['result.json']}; tail-check SHA-256 {hashes['tail_check.json']}.
'''
(dest/'report.md').write_text(report,encoding='utf-8')
(base.parent/'outputs/Gravity-positive-filtered-sources.md').write_text(report,encoding='utf-8')
print(json.dumps(dict(minimum_force_ratio=minimum,worst_tail_change=tail['worst_error'],hashes=hashes)))
