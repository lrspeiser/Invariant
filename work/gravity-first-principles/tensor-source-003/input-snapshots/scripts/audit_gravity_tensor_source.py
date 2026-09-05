"""Registered interpolation pilots; no observational scoring or admission."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from invariant_gravity_extensions.hankel_axisymmetric import cylindrical_jet
from invariant_gravity_extensions.hankel_tail import complete_leading_tail
from invariant_gravity_extensions.length_galaxy_development import regular_disks
from invariant_gravity_extensions.mixed_source import hankel_mixed_jet, leading_tail_mixed_jet
from invariant_gravity_extensions.tensor_potential import C3TensorPotential
from invariant_gravity_extensions.vertical_green import Sech2VerticalGreen


def serial(value):
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serial(v) for key, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_config = json.loads((ROOT/'configs/gravity_source_tail_audit_v2.json').read_bytes())
    source_path = 'work/gravity-first-principles/map-source-003/source_profiles.json'
    units_path = 'work/gravity-first-principles/map-source-003/result.json'
    transform_path = 'work/gravity-first-principles/potential-join-001/transform_r128_k64.json'
    config = {'scope': 'primary source only; R=0..4 kpc, z=0..0.8 kpc; numerical pilot, not full-domain admission',
        'coarse_spacing_kpc': [.25, .1], 'fine_spacing_kpc': [.125, .05],
        'probe_rule': 'one-quarter into every coarse cell in both coordinates; common fixed off-grid probes',
        'precision': source_config['precision'], 'vertical_intervals': 2400, 'vertical_extent': 24.,
        'cutoff': 400., 'source_profiles': source_path, 'source_result': units_path, 'transform': transform_path,
        'targets': {'force': .0001, 'hessian': .002, 'third': .01},
        'target_origin': 'unchanged force/H/T values from gravity_matched_source_audit_v1.json',
        'physical_source_changed': False, 'new_observations': False, 'full_action_solver_admitted': False}
    if args.config is not None:
        extension = json.loads(args.config.read_bytes())
        allowed = {'scope', 'coarse_radii_kpc', 'coarse_heights_kpc', 'variant', 'extra_probe_radii_kpc', 'extra_probe_heights_kpc'}
        if set(extension)-allowed:
            raise ValueError('Pilot extension may change only registered geometry and source variant')
        config.update(extension)
        if 'coarse_radii_kpc' in extension or 'coarse_heights_kpc' in extension:
            config.pop('coarse_spacing_kpc')
            config.pop('fine_spacing_kpc')
            config['grid_rule'] = 'Explicit nonuniform coarse arrays; fine grid bisects every coarse interval'
        config['probe_rule'] = 'Quarter into every current coarse cell, plus explicitly retained earlier probes; retained probes may coincide with new nodes'

    def write(name, value):
        with (args.output/name).open('x', encoding='utf8', newline='\n') as handle:
            json.dump(serial(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')

    paths = [Path(__file__), ROOT/source_path, ROOT/units_path, ROOT/transform_path,
        ROOT/'configs/gravity_source_tail_audit_v2.json', ROOT/'configs/gravity_matched_source_audit_v1.json',
        ROOT/'tests/test_gravity_tensor_potential.py', ROOT/'tests/test_gravity_mixed_source.py',
        *sorted((ROOT/'src/invariant_gravity_extensions').glob('*.py'))]
    if args.config is not None:
        paths.append(args.config.resolve())
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}
    for p in [source_path, units_path, transform_path]:
        if hashes[p] != source_config['input_files'][p]:
            raise ValueError(f'Frozen source input changed: {p}')
    for p in paths:
        target = args.output/'input-snapshots'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    write('started.json', {'config': config, 'input_hashes': hashes, 'started_utc': datetime.now(UTC).isoformat(),
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
    try:
        profile = json.loads((ROOT/source_path).read_bytes())['profiles'][-1]
        G = json.loads((ROOT/units_path).read_bytes())['config']['units']['G_kpc_kms2_msun']
        raw = json.loads((ROOT/transform_path).read_bytes())
        k, w, S = [np.array(raw[key]) for key in ['k', 'wavenumber_weights', 'surface_hankel']]
        _, disks = regular_disks(profile, config.get('variant', {'id': 'primary'}))
        coarse_r = np.array(config.get('coarse_radii_kpc', np.arange(17)*.25))
        coarse_z = np.array(config.get('coarse_heights_kpc', np.arange(9)*.1))
        for grid in [coarse_r, coarse_z]:
            if grid.ndim != 1 or len(grid) < 2 or grid[0] != 0 or np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
                raise ValueError('finite increasing source grids starting at zero required')
        fine_r = np.sort(np.r_[coarse_r, coarse_r[:-1]+np.diff(coarse_r)/2])
        fine_z = np.sort(np.r_[coarse_z, coarse_z[:-1]+np.diff(coarse_z)/2])
        probe_r = coarse_r[:-1]+np.diff(coarse_r)/4
        probe_z = coarse_z[:-1]+np.diff(coarse_z)/4
        probe_r = np.unique(np.r_[probe_r, config.get('extra_probe_radii_kpc', [])])
        probe_z = np.unique(np.r_[probe_z, config.get('extra_probe_heights_kpc', [])])
        r, z = np.unique(np.r_[fine_r, probe_r]), np.unique(np.r_[fine_z, probe_z])
        source = Sech2VerticalGreen(intervals=2400, extent=24.)
        vertical = []
        for name in raw['components']:
            print(f'Vertical source: {name}', flush=True)
            h = disks[name].height
            vertical.append(source.jet(k*h, z/h)/h**np.arange(4)[:, None, None])
        vertical = np.array(vertical)
        print('Computing sixteen mixed partials', flush=True)
        mixed = hankel_mixed_jet(k, w, S, vertical, r, z, G)
        tail, _ = leading_tail_mixed_jet(disks, raw['components'], k, w, S, r, z, G, 400., precision=config['precision'])
        mixed += tail
        write('mixed_table.json', {'radius_kpc': r, 'height_kpc': z, 'mixed': mixed})
        print('Direct reference at off-grid probes', flush=True)
        iz = np.searchsorted(z, probe_z)
        near = cylindrical_jet(k, w, S, vertical[:, :, iz], probe_r, probe_z, G)
        reference, _ = complete_leading_tail(near, disks, raw['components'], k, w, S, probe_r, probe_z,
            G, 400., precision=config['precision'])
        RR, ZZ = np.meshgrid(probe_r, probe_z, indexing='ij')
        H = np.sqrt(reference['hessian_norm'])
        scale = {'force': np.maximum(np.linalg.norm(reference['gradient_R_z'], axis=0), 1e-20),
            'hessian': np.maximum(H, 1e-20),
            'third': np.maximum(reference['third_tensor_norm'], H/(np.hypot(RR, ZZ)+min(d.height for d in disks.values())))}
        write('direct_reference.json', reference)
        rows = []
        for name, gr, gz in [('coarse', coarse_r, coarse_z), ('fine', fine_r, fine_z)]:
            ri, zi = np.searchsorted(r, gr), np.searchsorted(z, gz)
            table = mixed[:, :, ri][:, :, :, zi].copy()
            # One global gauge shift; derivatives and source remain unchanged.
            gauge = float(table[0, 0, 0, 0])
            table[0, 0] -= gauge
            potential = C3TensorPotential(gr, gz, table)
            value = potential.fields(RR, ZZ)
            errors = {}
            for key, field, weights in [('force', 'gradient_R_z', [1, 1]),
                ('hessian', 'hessian_RR_Rz_zz_pp', [1, 2, 1, 1]),
                ('third', 'third_RRR_RRz_Rzz_zzz_Rpp_zpp', [1, 3, 3, 1, 3, 3])]:
                delta = value[field]-reference[field]
                errors[key] = np.sqrt(np.einsum('i,irz,irz->rz', weights, delta, delta))/scale[key]
            maxima = {key: float(np.max(v)) for key, v in errors.items()}
            worst = {}
            for key, error in errors.items():
                index = np.unravel_index(np.argmax(error), error.shape)
                worst[key] = {'R_kpc': float(RR[index]), 'z_kpc': float(ZZ[index])}
            rows.append({'grid': name, 'shape': [len(gr), len(gz)], 'maxima': maxima, 'worst': worst,
                'within_pilot_targets': all(maxima[key] < target for key, target in config['targets'].items())})
            write(f'interpolated_{name}.json', {'fields': value, 'errors': errors, 'potential_gauge': gauge})
        for p in paths:
            if sha256(p.read_bytes()).hexdigest() != hashes[p.relative_to(ROOT).as_posix()]:
                raise ValueError(f'Execution input changed: {p}')
        write('result.json', {'config': config, 'rows': rows, 'completed_utc': datetime.now(UTC).isoformat(),
            'probe_count': RR.size, 'full_source_interpolation_admitted': False,
            'higher_mixed_derivative_quadrature_refinement': 'pending',
            'new_quality_verified_observational_tests': 0})
        print(json.dumps(rows, indent=2), flush=True)
    except Exception as exc:
        write('failure.json', {'type': type(exc).__name__, 'message': str(exc), 'utc': datetime.now(UTC).isoformat()})
        raise


if __name__ == '__main__':
    main()
