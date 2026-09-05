"""Expose the fixed-quadrature cancellation prototype to actual source points."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
from diagnose_gravity_length_cancellation import ROOT, precise, prototype

from invariant_gravity_extensions.exterior_moments import ExteriorMomentField
from invariant_gravity_extensions.galaxy_development import SI_ACCELERATION_TO_KMS2_KPC
from invariant_gravity_extensions.length_flux_difference import length_flux_difference
from invariant_gravity_extensions.length_screening import LengthScreening, anomalous_flux
from invariant_gravity_extensions.matched_tensor import MatchedTensorPotential
from invariant_gravity_extensions.potential_join import cartesian_tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    methods = parser.add_mutually_exclusive_group()
    methods.add_argument('--hybrid', action='store_true')
    methods.add_argument('--logarithmic', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = ROOT/'work/gravity-first-principles'
    source = base/'radial-tensor-replay-001'
    cards_path, units_path = base/'length-screening-local-001/result.json', base/'map-source-003/result.json'
    variants = ['primary', 'height_half']
    config = {'points_kpc':[[0.,0.],[0.,.1],[.0001,0.],[.3,.025],[2.,0.],[17.75,.025],[36.0625,.025],[65.,.025],[90.,.1]],
        'variants':variants, 'cards':'All 54 existing cards at nominal distance',
        'quadrature_orders':[16,32], 'relative_vector_target':1e-9,
        'scope':'Sampled actual-source diagnostic of the unmodified prototype; no full-domain or galaxy-force admission.',
        'precision_digits':80, 'production_changed':False,
        'hybrid':args.hybrid, 'logarithmic':args.logarithmic,
        'direct_subtraction_if_h_over_x_plus_epsilon_squared_above':.01,
        'hybrid_reason':'Integrate small shifts to avoid subtraction; use direct difference for large shifts to avoid unresolved quadrature endpoint structure.'}
    paths = {Path(__file__), ROOT/'scripts/diagnose_gravity_length_cancellation.py', cards_path, units_path,
        source/'result.json', *list((ROOT/'src/invariant_gravity_extensions').glob('*.py'))}
    for variant in variants:
        paths.add(source/f'mixed_canonical_{variant}.json')
        paths.add(base/f'exterior-moment-002/moments_{variant}_reference.json')
    hashes = {p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in paths}
    assert hashes[(source/'result.json').relative_to(ROOT).as_posix()] == '9136c90030a114b89b350f816beffa41ce24ad7958d6eeafaad107483e5c74c2'
    for path in paths:
        target = args.output/'input-snapshots'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())

    def write(name,data):
        with (args.output/name).open('x',encoding='utf8',newline='\n') as handle:
            json.dump(data,handle,indent=2,allow_nan=False)
            handle.write('\n')

    write('started.json',{'config':config,'input_hashes':hashes,'started_utc':datetime.now(UTC).isoformat()})
    cards = [r['card'] for r in json.loads(cards_path.read_bytes())['rows']]
    assert len(cards) == 54
    G = json.loads(units_path.read_bytes())['config']['units']['G_kpc_kms2_msun']
    rows = []
    for variant in variants:
        raw = json.loads((source/f'mixed_canonical_{variant}.json').read_bytes())
        moments = json.loads((base/f'exterior-moment-002/moments_{variant}_reference.json').read_bytes())
        provider = MatchedTensorPotential(np.array(raw['radius_kpc']),np.array(raw['height_kpc']),np.array(raw['mixed']),ExteriorMomentField(moments,G,minimum_radius=60.))
        for R,z in config['points_kpc']:
            fields = provider.fields(R,z)
            _, physical_p, physical_H, _ = cartesian_tensors(fields)
            for card in cards:
                a0 = card['a0_m_s2']*SI_ACCELERATION_TO_KMS2_KPC
                p,H = physical_p/a0,physical_H/a0
                dH2,dlap = np.r_[fields['gradient_hessian_norm_R_z'],0.]/a0**2,np.r_[fields['gradient_laplacian_R_z'],0.]/a0
                length = card['length_pc']/1000
                spec = LengthScreening(card['shape'],card['epsilon'])
                # Symmetry forces exact zero at the central stationary point;
                # ell=0 is the exact identical-law control, not an MP 0/0.
                zero = length == 0 or np.all(p == 0)
                ref = np.zeros(3) if zero else precise(card['shape'],card['epsilon'],p,H,dH2,dlap,length)
                direct = anomalous_flux(spec,p,H,dH2,dlap,length)-anomalous_flux(spec,p,H,dH2,dlap,0.)
                ratio = float(length**2*np.sum(H*H)/(np.dot(p,p)+spec.epsilon**2))
                use_direct = args.hybrid and ratio > .01
                values = {str(order):direct if use_direct else np.zeros(3) if length == 0 else prototype(spec,p,H,dH2,dlap,length,order) for order in [16,32]}
                if args.logarithmic:
                    values = {'logarithmic':length_flux_difference(spec,p,H,dH2,dlap,length)}
                norm = np.linalg.norm(ref)
                errors = {k:float(np.linalg.norm(v-ref)/norm) if norm else float(np.linalg.norm(v-ref)) for k,v in values.items()}
                rows.append({'variant':variant,'R_kpc':R,'z_kpc':z,'card':card['id'],
                    'h_over_x_plus_epsilon_squared':ratio, 'used_direct_branch':use_direct,
                    'zero_reference':bool(zero),'normalized_reference':ref.tolist(),
                    'prototype_errors':errors,'direct_subtraction_error':float(np.linalg.norm(direct-ref)/norm) if norm else float(np.linalg.norm(direct-ref)),
                    'error_scale':'relative vector' if norm else 'absolute normalized flux',
                    'within_target':all(e<1e-9 for e in errors.values())})
            print(f'{variant}: all 54 cards at R={R:g}, z={z:g}',flush=True)
    assert len(rows) == 972
    assert all(sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
    write('result.json',{'config':config,'rows':rows,'all_cases_complete':True,
        'passes':sum(r['within_target'] for r in rows),'production_changed':False,'new_observational_scores':0,'physical_exclusions':0})
    print(json.dumps({'cases':len(rows),'passes':sum(r['within_target'] for r in rows),
        'worst_prototype_error':max(e for r in rows for e in r['prototype_errors'].values())},indent=2))


if __name__ == '__main__':
    main()
