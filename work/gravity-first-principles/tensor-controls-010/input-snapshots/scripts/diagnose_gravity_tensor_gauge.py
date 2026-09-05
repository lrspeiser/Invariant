"""Compare global and cell-local gauge arithmetic at the exposed cutoff failure."""
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
from audit_gravity_matched_tensor import serial, source_errors

from invariant_gravity_extensions.exterior_moments import ExteriorMomentField
from invariant_gravity_extensions.length_galaxy_development import regular_disks
from invariant_gravity_extensions.matched_tensor import MatchedTensorPotential
from invariant_gravity_extensions.tensor_potential import C3TensorPotential


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    base = ROOT/'work/gravity-first-principles'
    table_path = base/'tensor-quadrature-cutoff-001/mixed_height_half.json'
    profile_path,units_path = base/'map-source-003/source_profiles.json',base/'map-source-003/result.json'
    moment_path = base/'exterior-moment-002/moments_height_half_reference.json'
    paths = [Path(__file__),ROOT/'scripts/audit_gravity_matched_tensor.py',table_path,profile_path,units_path,
        moment_path,*sorted((ROOT/'src/invariant_gravity_extensions').glob('*.py'))]
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
        'scope':'Exposed R=65,z=0.025 point only; compare mathematically equivalent gauge arithmetic without changing source samples'})
    table = json.loads(table_path.read_bytes())
    r,z,mixed = [np.array(table[key]) for key in ['radius_kpc','height_kpc','mixed']]
    G = json.loads(units_path.read_bytes())['config']['units']['G_kpc_kms2_msun']
    profile = json.loads(profile_path.read_bytes())['profiles'][-1]
    _,disks = regular_disks(profile,{'id':'height_half','height_factor':.5})
    exterior = ExteriorMomentField(json.loads(moment_path.read_bytes()),G,minimum_radius=60.)
    original = MatchedTensorPotential(r,z,mixed,exterior)
    cell_local = MatchedTensorPotential(r,z,mixed,exterior)
    # Controlled prototype: retain raw samples and let C3TensorPotential's
    # existing cell-local anchor handle subtraction/restoration internally.
    cell_local.near = C3TensorPotential(r,z,mixed)
    cell_local.gauge = 0.
    R,Z = np.array([65.]),np.array([.025])
    records = []
    for name,provider in [('global_shift',original),('cell_local_only',cell_local)]:
        fields = provider.fields(R,Z)
        records.append({'method':name,'fields':fields,'source_errors':source_errors(fields,disks,R,Z,G,.1)})
    i,j = np.searchsorted(r,65.)-1,np.searchsorted(z,.025)-1
    values = mixed[0,0,i:i+2,j:j+2]
    round_trip = (values-original.gauge)+original.gauge
    assert all(sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
    write('result.json',{'records':records,'raw_cell_potentials':values,'global_shift_round_trip_changes':round_trip-values,
        'raw_cell_ulp':abs(np.spacing(values)),'globally_shifted_cell_ulp':abs(np.spacing(values-original.gauge)),
        'new_observational_scores':0,'completed_utc':datetime.now(UTC).isoformat()})
    print(json.dumps(serial({'source_errors':[{ 'method':r['method'],'errors':r['source_errors']} for r in records],
        'round_trip_changes':round_trip-values}),indent=2))


if __name__=='__main__':
    main()
