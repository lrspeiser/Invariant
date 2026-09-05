"""Conditional total-mass range for the completed A2142 narrow-source case."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import linprog

root=Path(__file__).parent/'Invariant'
base=root/'work/gravity-first-principles'
run=base/'smooth-stellar-feasibility-001'
paths=[run/'runner.py',run/'case_A2142_0.02.json',base/'projected-stellar-packet-001/result.json']
# Load function definitions only; never execute the running campaign's entry point.
tree=ast.parse(paths[0].read_text(encoding='utf-8'))
for node in tree.body:
    if isinstance(node,ast.FunctionDef):
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(paths[0]),'exec'))
parent=json.loads(paths[1].read_bytes())
packet=json.loads(paths[2].read_bytes())
p=next(p for p in packet['stellar_packets'] if p['cluster']=='A2142')
c=p['columns']; r,m,lo,hi=[np.array(c[k]) for k in ['RADIUS','MSTAR','MSTAR_LO','MSTAR_HI']]
shells=np.array(parent['shell_radii_kpc']); scales=np.array(parent['offset_scales_kpc'])
fraction=np.c_[shell_matrix(r,shells,.02,512),offset_fraction(r[:,None],scales[None,:],parent['offset_kpc'])]
design=fraction*m[-1]/m[:,None]
A=np.r_[design,-design]; b=np.r_[hi/m,-lo/m]
rows=[]
for direction in [1,-1]:
    fit=linprog(np.full(design.shape[1],direction),A_ub=A,b_ub=b,bounds=(0,None),method='highs',
                options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    assert fit.success
    masses=fit.x*m[-1]; predicted=fraction@masses
    violation=float(max(0,np.max((lo-predicted)/m),np.max((predicted-hi)/m)))
    assert violation<1e-8 and np.all(masses>=0)
    rows.append(dict(objective='minimum_total_mass' if direction==1 else 'maximum_total_mass',
        total_mass_msun=float(sum(masses)),total_over_last_measured_mass=float(sum(masses)/m[-1]),
        bracket_violation=violation,component_mass_msun=masses.tolist(),projected_mass_msun=predicted.tolist()))
out=dict(input_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    cluster='A2142',width=.02,shell_radii_kpc=shells.tolist(),offset_scales_kpc=scales.tolist(),
    offset_kpc=parent['offset_kpc'],rows=rows,
    scope='Finite-dictionary deterministic mass range under all original projected brackets, fixed observed center and widths. Not a confidence interval, no outer-mass prior and no preferred reconstruction.',
    gravity_scores=0,physical_exclusions=0)
dest=base/'smooth-source-tail-range-001'
dest.mkdir(exist_ok=False)
(dest/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
print([(r['objective'],r['total_over_last_measured_mass'],r['bracket_violation']) for r in rows])
