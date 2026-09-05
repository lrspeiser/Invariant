"""Replay, radial-support and geometry-iteration checks on frozen results."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from gravity_cube_constrained import ConstrainedCube
from gravity_cube_model import tensor
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'work/gravity-first-principles/constrained-cube-balanced-001'
source=Path(__file__).with_name('gravity_cube_constrained.py').read_text()
reference={};exec(compile(source.replace('range(6)','range(20)'),'<geometry-20-iteration-reference>','exec'),reference)
LongGeometry=reference['ConstrainedCube']
result=json.loads((D/'result.json').read_text());audit=json.loads((D/'mask-audit.json').read_text())
assert len(result['objects'])==12 and not result['failures']
for a in audit:
    assert a['minimum_center_separation_arcsec']>=120 and a['response_mutation_check_pass']
    assert all(not good or min(x,y)>=8 for good,x,y in zip(a['admitted_radial_bins'],a['radial_train_counts'],a['radial_test_counts']))
rows=[]
with torch.no_grad():
    for obj in result['objects']:
        name=obj['name'];path=ROOT/'work/private/constrained-cube-balanced-001'/(name+'.npz');p=dict(np.load(path))
        m=ConstrainedCube(p);long=LongGeometry(p);f=obj['fits'][-1];params=tensor(f['params'])
        replay=float(m.loss(params,'full','test',penalize=False));assert abs(replay-f['test_loss'])<1e-4
        assert all(min(x['params'][:7])>=0 for x in obj['fits'])
        prediction=m.render(params,'full');reference_prediction=long.render(params,'full')
        discrepancy=m.white@(prediction-reference_prediction)[:,m.test]
        difference=float(torch.sqrt(torch.mean(discrepancy**2)))
        rows.append(dict(name=name,replayed_full_test_loss=replay,geometry_6_vs_20_whitened_rms=difference,
            geometry_difference_below_005=difference<.05,packet_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        del m,long
(D/'verification.json').write_text(json.dumps(dict(status='NUMERICAL_REPLAY_AND_MASK_CHECKS_PASS',objects=rows,
    geometry_reference='Same parameter vector, 20 rather than 6 damped radius iterations; diagnostic threshold .05 whitened RMS.',
    geometry_accuracy_all_pass=all(r['geometry_difference_below_005'] for r in rows)),indent=2,allow_nan=False))
print(json.dumps(rows,indent=2))
