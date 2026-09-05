"""Linear mixed-input attribution at the exposed cutoff interpolation failure."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from audit_gravity_matched_tensor import serial

from invariant_gravity_extensions.exterior_moments import ExteriorMomentField
from invariant_gravity_extensions.matched_tensor import MatchedTensorPotential
from invariant_gravity_extensions.potential_join import pack_cartesian


class ZeroExterior:
    minimum_radius = 60.

    def fields(self,R,z):
        R,z = np.broadcast_arrays(R,z)
        return pack_cartesian(np.zeros(R.shape),np.zeros((3,)+R.shape),np.zeros((3,3)+R.shape),
            np.zeros((3,3,3)+R.shape),R,z)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    base = ROOT/'work/gravity-first-principles'
    low_path = base/'tensor-quadrature-cutoff-001/mixed_height_half.json'
    ref_path = base/'tensor-source-004/mixed_table.json'
    moment_path = base/'exterior-moment-002/moments_height_half_reference.json'
    units_path = base/'map-source-003/result.json'
    paths = [Path(__file__),ROOT/'scripts/audit_gravity_matched_tensor.py',low_path,ref_path,moment_path,units_path,
        *sorted((ROOT/'src/invariant_gravity_extensions').glob('*.py'))]
    hashes = {p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in paths}
    for p in paths:
        target = args.output/'input-snapshots'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.read_bytes())

    def write(name,value):
        with (args.output/name).open('x',encoding='utf8',newline='\n') as handle:
            json.dump(serial(value),handle,indent=2,allow_nan=False)
            handle.write('\n')

    write('started.json',{'input_hashes':hashes,'started_utc':datetime.now(UTC).isoformat(),
        'scope':'One exposed point; linear decomposition of cutoff-200 minus cutoff-400 sampled mixed jets, not a repaired source'})
    raw = json.loads(low_path.read_bytes())
    r,z,low = [np.array(raw[key]) for key in ['radius_kpc','height_kpc','mixed']]
    raw = json.loads(ref_path.read_bytes())
    tr,tz,ref = [np.array(raw[key]) for key in ['radius_kpc','height_kpc','mixed']]
    ri,zi = np.searchsorted(tr,r),np.searchsorted(tz,z)
    assert np.array_equal(tr[ri],r) and np.array_equal(tz[zi],z)
    ref = ref[:,:,ri][:,:,:,zi]
    G = json.loads(units_path.read_bytes())['config']['units']['G_kpc_kms2_msun']
    exterior = ExteriorMomentField(json.loads(moment_path.read_bytes()),G,minimum_radius=60.)
    R,Z = np.array([65.]),np.array([.025])
    f0 = MatchedTensorPotential(r,z,ref,exterior).fields(R,Z)
    f1 = MatchedTensorPotential(r,z,low,exterior).fields(R,Z)
    scale = float(np.sqrt(f0['hessian_norm'][0])/(np.hypot(R,Z)[0]+.1))
    measured = (f1['gradient_laplacian_R_z']-f0['gradient_laplacian_R_z'])[:,0]
    rows = []
    for i in range(4):
        for j in range(4):
            delta = np.zeros_like(ref)
            delta[i,j] = low[i,j]-ref[i,j]
            field = MatchedTensorPotential(r,z,delta,ZeroExterior()).fields(R,Z)
            contribution = field['gradient_laplacian_R_z'][:,0]
            rows.append({'radial_order':i,'vertical_order':j,'gradient_trace_contribution':contribution,
                'scaled_vector':contribution/scale,'scaled_norm':float(np.linalg.norm(contribution)/scale)})
    total = sum(row['gradient_trace_contribution'] for row in rows)
    residual = measured-total
    result = {'R_kpc':65.,'z_kpc':.025,'reference_scale':scale,'rows':rows,
        'measured_difference':measured,'summed_linear_contributions':total,
        'rounding_residual':residual,'scaled_residual_norm':float(np.linalg.norm(residual)/scale),
        'largest_contributions':sorted(rows,key=lambda x:x['scaled_norm'],reverse=True)[:5],
        'physical_source_changed':False,'repaired_source':False,'new_observational_scores':0}
    assert all(sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
    write('result.json',result)
    print(json.dumps(serial({k:result[k] for k in ['measured_difference','scaled_residual_norm','largest_contributions']}),indent=2))


if __name__=='__main__':
    main()
