"""Replay the two completed A2319 constraint-check failures without clipping."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import linprog

root=Path(__file__).parent/'Invariant'
base=root/'work/gravity-first-principles'
runner=base/'smooth-stellar-feasibility-002/runner.py'
for node in ast.parse(runner.read_text(encoding='utf-8')).body:
    if isinstance(node,ast.FunctionDef):
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(runner),'exec'))
packet_path=base/'projected-stellar-packet-001/result.json'
p=next(p for p in json.loads(packet_path.read_bytes())['stellar_packets'] if p['cluster']=='A2319')
c=p['columns']; r,m,lo,hi=[np.array(c[k]) for k in ['RADIUS','MSTAR','MSTAR_LO','MSTAR_HI']]
dest=base/'smooth-source-solver-replay-001'
dest.mkdir(exist_ok=False)
paths=[runner,packet_path]+[base/f'smooth-stellar-feasibility-002/case_A2319_{w}.json' for w in [.02,.1]]
registration=dict(input_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    methods=['highs-ds','highs-ipm'],presolve=False,direct_constraint_target=2e-8,
    scope='Same matrix, source basis, projected bounds and minimax objective; replay numerical optimization only. No parameter-range expansion or physical exclusion.')
(dest/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
rows=[]
for width in [.02,.1]:
    original=json.loads((base/f'smooth-stellar-feasibility-002/case_A2319_{width}.json').read_bytes())
    shells=np.array(original['shell_radii_kpc']); scales=np.array(original['offset_scales_kpc'])
    fraction=np.c_[shell_matrix(r,shells,width,512),offset_fraction(r[:,None],scales[None,:],original['offset_kpc'])]
    design=fraction*m[-1]/m[:,None]
    A=np.r_[np.c_[design,-np.ones(len(r))],np.c_[-design,-np.ones(len(r))]]
    b=np.r_[hi/m,-lo/m]
    for method in registration['methods']:
        answer=linprog(np.r_[np.zeros(design.shape[1]),1.],A_ub=A,b_ub=b,bounds=(0,None),method=method,
            options={'presolve':False,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'ipm_optimality_tolerance':1e-10})
        row=dict(width=width,method=method,status=int(answer.status),message=answer.message)
        if answer.success:
            masses=answer.x[:-1]*m[-1]; pred=fraction@masses
            violation=float(max(0,np.max((lo-pred)/m),np.max((pred-hi)/m)))
            discrepancy=abs(violation-answer.x[-1])
            row.update(relaxation=float(answer.x[-1]),verified_violation=violation,
                relaxation_discrepancy=float(discrepancy),minimum_mass=float(masses.min()),
                direct_constraint_pass=bool(discrepancy<2e-8 and np.all(masses>=0)),
                component_mass_msun=masses.tolist())
        rows.append(row)
        print(width,method,row.get('relaxation_discrepancy'),row.get('direct_constraint_pass'),flush=True)
(dest/'result.json').write_text(json.dumps(dict(registration=registration,rows=rows,gravity_scores=0,
    physical_exclusions=0),indent=2),encoding='utf-8')
