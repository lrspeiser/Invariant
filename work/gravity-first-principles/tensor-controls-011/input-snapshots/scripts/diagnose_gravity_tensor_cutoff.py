"""Exposed-point diagnosis of the failed half-thickness cutoff source identity."""
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
from invariant_gravity_extensions.hankel_axisymmetric import cylindrical_jet
from invariant_gravity_extensions.hankel_tail import complete_leading_tail
from invariant_gravity_extensions.length_galaxy_development import regular_disks
from invariant_gravity_extensions.matched_axisymmetric import matched_grid
from invariant_gravity_extensions.vertical_green import Sech2VerticalGreen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    base = ROOT/'work/gravity-first-principles'
    transform_path = base/'potential-join-001/transform_r128_k64.json'
    profile_path,units_path = base/'map-source-003/source_profiles.json',base/'map-source-003/result.json'
    moment_path = base/'exterior-moment-002/moments_height_half_reference.json'
    failure_path = base/'tensor-quadrature-cutoff-001/result.json'
    interpolation_path = base/'tensor-quadrature-cutoff-001/fields_height_half.json'
    paths = [Path(__file__),ROOT/'scripts/audit_gravity_matched_tensor.py',transform_path,profile_path,units_path,
        moment_path,failure_path,interpolation_path,*sorted((ROOT/'src/invariant_gravity_extensions').glob('*.py'))]
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
        'scope':'Previously exposed numerical failure point R=65,z=0.025 kpc, half thickness; no broad admission or observational score',
        'cutoffs':[200,400]})
    try:
        failure = json.loads(failure_path.read_bytes())
        row = next(r for r in failure['rows'] if r['variant']=='height_half')
        point = row['source_identities']['density_gradient']
        assert point['value'] > .01 and point['R_kpc']==65. and point['z_kpc']==.025
        profile = json.loads(profile_path.read_bytes())['profiles'][-1]
        G = json.loads(units_path.read_bytes())['config']['units']['G_kpc_kms2_msun']
        _,disks = regular_disks(profile,{'id':'height_half','height_factor':.5})
        exterior = ExteriorMomentField(json.loads(moment_path.read_bytes()),G,minimum_radius=60.)
        raw = json.loads(transform_path.read_bytes())
        k0,w0,S0 = [np.array(raw[key]) for key in ['k','wavenumber_weights','surface_hankel']]
        R,z = np.array([65.]),np.array([.025])
        vertical_source = Sech2VerticalGreen(intervals=2400,extent=24.)
        records = []
        for cutoff in [200,400]:
            keep = k0 < cutoff
            k,w,S = k0[keep],w0[keep],S0[:,keep]
            vertical = []
            for name in raw['components']:
                h = disks[name].height
                vertical.append(vertical_source.jet(k*h,z/h)/h**np.arange(4)[:,None,None])
            near = cylindrical_jet(k,w,S,np.array(vertical),R,z,G)
            corrected,_ = complete_leading_tail(near,disks,raw['components'],k,w,S,R,z,G,cutoff,
                precision={'digits':50,'low_k_limit':8.})
            matched = matched_grid(corrected,exterior,R,z,inner=60.,outer=80.)
            errors = source_errors(matched,disks,R[:,None],z[None,:],G,min(d.height for d in disks.values()))
            records.append({'cutoff':cutoff,'direct_source_errors':errors,'fields':matched})
        interpolation = json.loads(interpolation_path.read_bytes())
        selected = (np.array(interpolation['R_kpc'])==65.) & (np.array(interpolation['z_kpc'])==.025)
        assert np.any(selected)
        stored = {key:np.array(v)[...,selected] for key,v in interpolation['fields'].items()}
        assert all(sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
        write('result.json',{'point':point,'direct_records':records,'stored_cutoff_200_interpolated_fields':stored,
            'stored_interpolated_source_errors':{key:np.array(v)[selected] for key,v in interpolation['source_errors'].items()},
            'claim_ceiling':'One exposed-point diagnostic; integration and interpolation error must still be separated across the full registered grid',
            'new_observational_scores':0,'completed_utc':datetime.now(UTC).isoformat()})
        print(json.dumps(serial({'direct':[{ 'cutoff':r['cutoff'],'source_errors':r['direct_source_errors']} for r in records],
            'interpolated_cutoff_200_density_gradient':point['value']}),indent=2))
    except Exception as exc:
        write('failure.json',{'type':type(exc).__name__,'message':str(exc)})
        raise


if __name__=='__main__':
    main()
