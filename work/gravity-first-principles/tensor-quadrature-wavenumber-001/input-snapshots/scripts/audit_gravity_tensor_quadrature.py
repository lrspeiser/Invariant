"""Vary one source quadrature with a fixed joined fine tensor representation."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from audit_gravity_matched_tensor import maxima, norm, scales, serial, source_errors

from invariant_gravity_extensions.exterior_moments import ExteriorMomentField
from invariant_gravity_extensions.length_galaxy_development import regular_disks
from invariant_gravity_extensions.matched_tensor import MatchedTensorPotential
from invariant_gravity_extensions.mixed_source import hankel_mixed_jet, leading_tail_mixed_jet
from invariant_gravity_extensions.vertical_green import Sech2VerticalGreen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = ROOT/'work/gravity-first-principles'
    config_path = ROOT/'configs/gravity_tensor_quadrature_v1.json'
    config = json.loads(config_path.read_bytes())
    case = next(c for c in config['cases'] if c['id']==args.case)
    tail_path = ROOT/'configs/gravity_source_tail_audit_v2.json'
    tail_config = json.loads(tail_path.read_bytes())
    audit_result = base/'matched-tensor-002/result.json'
    if sha256(audit_result.read_bytes()).hexdigest() != 'a8dbc3d8a86914d63912fa7841b159aebc54158603abad2ed36c4a5543e9d760':
        raise ValueError('Joined reference audit changed')
    probes = json.loads(audit_result.read_bytes())['config']
    profile_path, units_path = base/'map-source-003/source_profiles.json', base/'map-source-003/result.json'
    variants = [('primary','tensor-source-003','41f956064f275472550a03a3a663b792a7171b9d25f68aa08b8e331f6a17d9f0'),
        ('height_half','tensor-source-004','44965a3659efa9dc8f2129452736a2be7b63c00f1804d080e7a076608ac72e62')]
    paths = [Path(__file__),ROOT/'scripts/audit_gravity_matched_tensor.py',config_path,tail_path,audit_result,
        profile_path,units_path,ROOT/case['transform'],*sorted((ROOT/'src/invariant_gravity_extensions').glob('*.py'))]
    for variant,folder,digest in variants:
        if sha256((base/folder/'result.json').read_bytes()).hexdigest()!=digest:
            raise ValueError('Reference source pilot changed')
        paths.extend([base/folder/name for name in ['result.json','mixed_table.json','direct_reference.json']])
        paths.append(base/f'exterior-moment-002/moments_{variant}_reference.json')
    hashes = {p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in paths}
    for p in [profile_path,units_path,ROOT/case['transform'],*[base/f'exterior-moment-002/moments_{v}_reference.json' for v,_,_ in variants]]:
        if hashes[p.relative_to(ROOT).as_posix()]!=tail_config['input_files'][p.relative_to(ROOT).as_posix()]:
            raise ValueError('Registered source integration input changed')
    for p in paths:
        target = args.output/'input-snapshots'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.read_bytes())

    def write(name,value):
        with (args.output/name).open('x',encoding='utf8',newline='\n') as handle:
            json.dump(serial(value),handle,indent=2,allow_nan=False)
            handle.write('\n')

    write('started.json',{'config':config,'case':case,'input_hashes':hashes,'started_utc':datetime.now(UTC).isoformat(),
        'new_observational_scores':0,'full_source_admitted':False})
    try:
        raw = json.loads((ROOT/case['transform']).read_bytes())
        k,w,S = [np.array(raw[key]) for key in ['k','wavenumber_weights','surface_hankel']]
        keep = k < case['cutoff']
        k,w,S = k[keep],w[keep],S[:,keep]
        profile = json.loads(profile_path.read_bytes())['profiles'][-1]
        G = json.loads(units_path.read_bytes())['config']['units']['G_kpc_kms2_msun']
        vertical_source = Sech2VerticalGreen(intervals=case['vertical_intervals'],extent=case['vertical_extent'])
        rows = []
        for variant,folder,_ in variants:
            pilot = json.loads((base/folder/'result.json').read_bytes())['config']
            gr,gz = [np.array(pilot[key]) for key in ['coarse_radii_kpc','coarse_heights_kpc']]
            gr,gz = [np.sort(np.r_[a,a[:-1]+np.diff(a)/2]) for a in [gr,gz]]
            _,disks = regular_disks(profile,pilot['variant'])
            table = json.loads((base/folder/'mixed_table.json').read_bytes())
            tr,tz,tm = [np.array(table[key]) for key in ['radius_kpc','height_kpc','mixed']]
            ri,zi = np.searchsorted(tr,gr),np.searchsorted(tz,gz)
            assert np.array_equal(tr[ri],gr) and np.array_equal(tz[zi],gz)
            canonical = tm[:,:,ri][:,:,:,zi]
            vertical = []
            for name in raw['components']:
                print(f'{args.case} {variant}: vertical {name}',flush=True)
                h = disks[name].height
                vertical.append(vertical_source.jet(k*h,gz/h)/h**np.arange(4)[:,None,None])
            print(f'{args.case} {variant}: sixteen mixed partials',flush=True)
            mixed = hankel_mixed_jet(k,w,S,np.array(vertical),gr,gz,G)
            tail,_ = leading_tail_mixed_jet(disks,raw['components'],k,w,S,gr,gz,G,case['cutoff'],precision=tail_config['precision'])
            mixed += tail
            write(f'mixed_{variant}.json',{'radius_kpc':gr,'height_kpc':gz,'mixed':mixed})
            moments = json.loads((base/f'exterior-moment-002/moments_{variant}_reference.json').read_bytes())
            exterior = ExteriorMomentField(moments,G,minimum_radius=60.)
            reference = MatchedTensorPotential(gr,gz,canonical,exterior)
            value = MatchedTensorPotential(gr,gz,mixed,exterior)
            old = json.loads((base/folder/'direct_reference.json').read_bytes())
            R,Z = np.meshgrid(old['radius'],old['height'],indexing='ij')
            DR,DZ = np.meshgrid(probes['derivative_radii_kpc'],probes['derivative_heights_kpc'],indexing='ij')
            R,Z = np.r_[R.ravel(),DR.ravel()],np.r_[Z.ravel(),DZ.ravel()]
            ref,got = reference.fields(R,Z),value.fields(R,Z)
            h = min(d.height for d in disks.values())
            half = profile['stellar_half_mass_radius_kpc']
            scale = scales(ref,R,Z,h,G*moments['compact_source_mass']/half**2,half)
            errors = {'force':np.linalg.norm(got['gradient_R_z']-ref['gradient_R_z'],axis=0)/scale['force'],
                'hessian':norm(got['hessian_RR_Rz_zz_pp']-ref['hessian_RR_Rz_zz_pp'],[1,2,1,1])/scale['hessian'],
                'third':norm(got['third_RRR_RRz_Rzz_zzz_Rpp_zpp']-ref['third_RRR_RRz_Rzz_zzz_Rpp_zpp'],[1,3,3,1,3,3])/scale['third']}
            identities = source_errors(got,disks,R,Z,G,h)
            row = {'variant':variant,'probe_entries':R.size,'quadrature_change':maxima(errors,R,Z),
                'source_identities':maxima(identities,R,Z),
                'maximum_absolute_mixed_changes':np.max(abs(mixed-canonical),axis=(2,3)),
                'mixed_units':'(km/s)^2 / kpc^(radial_order+vertical_order); absolute entries have no standalone admission threshold'}
            row['within_registered_targets'] = all(v['value'] < config['targets'][key]
                for group in [row['quadrature_change'],row['source_identities']] for key,v in group.items())
            rows.append(row)
            write(f'fields_{variant}.json',{'R_kpc':R,'z_kpc':Z,'fields':got,'quadrature_errors':errors,'source_errors':identities})
            print(json.dumps(serial(row)),flush=True)
        assert all(sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
        write('result.json',{'case':case,'rows':rows,'completed_utc':datetime.now(UTC).isoformat(),
            'all_registered_cases_complete':False,'full_source_admitted':False,'new_observational_scores':0})
    except Exception as exc:
        write('failure.json',{'type':type(exc).__name__,'message':str(exc)})
        raise


if __name__ == '__main__':
    main()
