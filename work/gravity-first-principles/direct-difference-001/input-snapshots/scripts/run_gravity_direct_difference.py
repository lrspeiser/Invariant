"""Solve the length-dependent flux difference directly and refine its grid."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
from run_gravity_tensor_poisson import (
    ROOT,
    ExteriorMomentField,
    FluxPoissonSolver,
    MatchedTensorPotential,
    MultipoleGrid,
    serial,
)

from invariant_gravity_extensions.galaxy_development import SI_ACCELERATION_TO_KMS2_KPC
from invariant_gravity_extensions.length_flux_difference import length_flux_difference
from invariant_gravity_extensions.length_screening import LengthScreening
from invariant_gravity_extensions.potential_join import cartesian_tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = ROOT/'work/gravity-first-principles'
    source, old = base/'radial-tensor-replay-001', base/'tensor-poisson-001'
    previous = json.loads((old/'result.json').read_bytes())
    baseline = previous['config']['grids']['fine']
    grids = {'baseline':baseline,
        'angles':{**baseline,'angular_nodes':640},
        'multipoles':{**baseline,'angular_nodes':640,'l_max':160},
        'radial':{**baseline,'angular_nodes':640,'l_max':160,'radial_nodes':4097},
        'boundary':{**baseline,'angular_nodes':640,'l_max':160,'radial_nodes':4609,'r_min':.00005,'r_max':6000.}}
    variants, distances = previous['config']['variants'], previous['config']['distances']
    config = {'registration':'Freeze direct-difference grids before evaluating this campaign; retain all 54 cards, three distances and both thicknesses.',
        'grids':grids,'variants':variants,'distances':distances,'relative_peak_signal_target':.01,
        'comparison_pairs':[['angles','baseline'],['multipoles','angles'],['radial','multipoles'],['boundary','radial']],
        'metric':'max(norm(response_a-response_b)/fixed_full_force_scale) divided by max(norm(response_a)/fixed_full_force_scale); exact-zero cards checked separately.',
        'scope':'Sampled small-signal numerical convergence, not pointwise relative or uniform error bounds, observational detectability or physical source uncertainty.',
        'baseline_comparison':'Preserve old full-field subtraction as a diagnostic; do not require the inaccurate old tiny differences to match within a relative-signal tolerance.',
        'new_observational_scores':0}
    cards_path, units_path = base/'length-screening-local-001/result.json',base/'map-source-003/result.json'
    paths = {Path(__file__),ROOT/'scripts/run_gravity_tensor_poisson.py',ROOT/'scripts/audit_gravity_tensor_flux.py',
        cards_path,units_path,old/'result.json',source/'result.json',
        base/'source-cancellation-003/result.json',*list((ROOT/'src/invariant_gravity_extensions').glob('*.py'))}
    for variant in variants:
        paths.add(source/f'mixed_canonical_{variant}.json')
        paths.add(base/f'exterior-moment-002/moments_{variant}_reference.json')
        paths.add(old/f'family_{variant}_fine.json')
    hashes = {p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    expected = {old/'result.json':'5c5d19ea954df993f7e4e2104d6495257eb7b7995c0bb9cf56fb12f0ad656306',
        source/'result.json':'9136c90030a114b89b350f816beffa41ce24ad7958d6eeafaad107483e5c74c2',
        cards_path:'66ff601b1012da7cbc555a27d8836723a2c6e7b23f393ead530da64e6e938a77',
        base/'source-cancellation-003/result.json':'49d7cd269833880f4694ae831c153f952a8c467cb0b21ebf2c766d2387e49d2f'}
    for path,digest in expected.items():
        assert hashes[path.relative_to(ROOT).as_posix()] == digest
    for path in paths:
        target = args.output/'input-snapshots'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())

    def write(name,data):
        with (args.output/name).open('x',encoding='utf8',newline='\n') as handle:
            json.dump(data,handle,indent=2,default=serial,allow_nan=False)
            handle.write('\n')

    write('started.json',{'config':config,'input_hashes':hashes,'started_utc':datetime.now(UTC).isoformat()})
    cards = [r['card'] for r in json.loads(cards_path.read_bytes())['rows']]
    assert len(cards)==len({c['id'] for c in cards})==54
    G = json.loads(units_path.read_bytes())['config']['units']['G_kpc_kms2_msun']
    families,comparisons,subtractions = {},[],[]
    try:
        for variant in variants:
            raw = json.loads((source/f'mixed_canonical_{variant}.json').read_bytes())
            moments = json.loads((base/f'exterior-moment-002/moments_{variant}_reference.json').read_bytes())
            provider = MatchedTensorPotential(np.array(raw['radius_kpc']),np.array(raw['height_kpc']),np.array(raw['mixed']),ExteriorMomentField(moments,G,minimum_radius=60.))
            reference = json.loads((old/f'family_{variant}_fine.json').read_bytes())
            radii = np.array(reference['radii_kpc'])
            lookup = {(r['card'],r['distance']):r for r in reference['predictions']}
            cached_key,fields = None,None
            for name,definition in grids.items():
                solver = FluxPoissonSolver(MultipoleGrid(**definition))
                key = (definition['r_min'],definition['r_max'],definition['radial_nodes'],definition['angular_nodes'])
                if key != cached_key:
                    fields = None
                    chunks = []
                    for start in range(0,len(solver.radius),32):
                        r = solver.radius[start:start+32,None]
                        f = provider.fields(r*solver.sine,r*solver.mu)
                        _,p,H,_ = cartesian_tensors(f)
                        dH2,dlap = np.zeros_like(p),np.zeros_like(p)
                        dH2[:2],dlap[:2] = f['gradient_hessian_norm_R_z'],f['gradient_laplacian_R_z']
                        chunks.append((p,H,dH2,dlap))
                    fields = [np.concatenate([c[i] for c in chunks],axis=-2) for i in range(4)]
                    del chunks
                    cached_key = key
                print(f'{variant}/{name}: source ready ({definition["radial_nodes"]} x {definition["angular_nodes"]}, lmax={definition["l_max"]})',flush=True)
                predictions = []
                for distance in distances:
                    for card in cards:
                        difference = length_flux_difference(LengthScreening(card['shape'],card['epsilon']),*fields,
                            card['length_pc']/1000/distance,card['a0_m_s2']*SI_ACCELERATION_TO_KMS2_KPC)
                        flux = np.array([solver.sine*difference[0]+solver.mu*difference[1],solver.mu*difference[0]-solver.sine*difference[1]])
                        response = -solver.solve(flux).evaluate(radii,np.zeros_like(radii))['acceleration']
                        ref = lookup[(card['id'],distance)]
                        scale = np.maximum(abs(np.array(ref['gradient_R_z'])[0]),abs(np.array(reference['newton_gradient_R_z'])[0]))
                        prediction = {'card':card['id'],'length_pc':card['length_pc'],'distance':distance,'response_gradient_R_z':response,
                            'fixed_full_force_scale':scale,'zero_control_pass':bool(np.all(response==0)) if card['length_pc']==0 else None}
                        predictions.append(prediction)
                        if name == 'baseline':
                            old_signal = np.array(ref['length_minus_zero_gradient_R_z'])
                            peak = float(max(np.linalg.norm(response,axis=0)/scale))
                            error = float(max(np.linalg.norm(response-old_signal,axis=0)/scale))
                            subtractions.append({'variant':variant,'card':card['id'],'distance':distance,'length_pc':card['length_pc'],
                                'maximum_scaled_signal':peak,'maximum_scaled_old_subtraction_change':error,
                                'change_over_peak_signal':error/peak if peak else None})
                    print(f'{variant}/{name}: all 54 cards at distance {distance:.6g}',flush=True)
                families[variant+'/'+name] = {(r['card'],r['distance']):r for r in predictions}
                write(f'family_{variant}_{name}.json',{'variant':variant,'grid':definition,'radii_kpc':radii,'predictions':predictions})
            for newer,older in config['comparison_pairs']:
                for key,a in families[variant+'/'+newer].items():
                    b = families[variant+'/'+older][key]
                    scale = a['fixed_full_force_scale']
                    peak = float(max(np.linalg.norm(a['response_gradient_R_z'],axis=0)/scale))
                    change = float(max(np.linalg.norm(a['response_gradient_R_z']-b['response_gradient_R_z'],axis=0)/scale))
                    zero = a['length_pc']==0
                    passed = (a['zero_control_pass'] and b['zero_control_pass']) if zero else (peak>0 and change/peak<.01)
                    comparisons.append({'variant':variant,'newer':newer,'older':older,'card':key[0],'distance':key[1],
                        'length_pc':a['length_pc'],'maximum_scaled_signal':peak,'maximum_scaled_change':change,
                        'change_over_peak_signal':change/peak if peak else None,'zero_control':zero,'within_target':bool(passed)})
        assert len(comparisons)==1296 and len(subtractions)==324
        assert all(sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
        write('result.json',{'config':config,'comparisons':comparisons,'old_subtraction_comparisons':subtractions,
            'all_cases_complete':True,'all_small_signal_comparisons_pass':all(c['within_target'] for c in comparisons),
            'new_observational_scores':0,'physical_exclusions':0,'completed_utc':datetime.now(UTC).isoformat()})
    except Exception as exc:
        write('failure.json',{'type':type(exc).__name__,'message':str(exc)})
        raise


if __name__ == '__main__':
    main()
