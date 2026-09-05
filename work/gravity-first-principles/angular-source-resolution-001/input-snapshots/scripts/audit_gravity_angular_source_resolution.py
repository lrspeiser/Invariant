"""Audit the existing positive source's angular representation without responses."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from invariant_gravity_extensions.length_galaxy_development import regular_disks
from invariant_gravity_extensions.source_projection import project_even_source, projection_metrics


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
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
    parser.add_argument('--config', type=Path, default=ROOT/'configs/gravity_angular_source_resolution_v1.json')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_bytes())
    paths = [Path(__file__), args.config.resolve(), *[ROOT/config[k] for k in ['source_profiles', 'source_result']],
             *[ROOT/p for p in config['control_tests']], *sorted((ROOT/'src/invariant_gravity_extensions').glob('*.py'))]
    # The predecessor is only rehashed, not parsed for velocity outcomes.
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}
    if sha256((ROOT/config['predecessor_result']).read_bytes()).hexdigest() != config['predecessor_result_sha256']:
        raise ValueError('Predecessor changed')
    if hashes[config['source_result']] != config['source_result_sha256']:
        raise ValueError('Source result changed')
    for path in paths:
        target = args.output/'input-snapshots'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())

    def write(name, value):
        with (args.output/name).open('x', encoding='utf8', newline='\n') as handle:
            json.dump(serial(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')

    provenance = {'config': config, 'input_hashes': hashes, 'started_utc': datetime.now(UTC).isoformat(),
                  'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'new_velocity_scoring': False, 'new_gravity_predictions': False, 'new_raw_or_reserved_data': False}
    write('started.json', provenance)
    try:
        control = subprocess.run([sys.executable, '-m', 'pytest', *config['control_tests'], '-q'], cwd=ROOT,
                                 env={**os.environ, 'PYTHONPATH': str(ROOT/'src'), 'OPENBLAS_NUM_THREADS': '1'},
                                 capture_output=True, text=True, check=False)
        write('controls.json', {'command': control.args, 'exit_code': control.returncode, 'stdout': control.stdout, 'stderr': control.stderr})
        if control.returncode:
            raise RuntimeError('Polynomial source controls failed')
        maps = json.loads((ROOT/config['source_profiles']).read_bytes())
        rows = []
        for variant in config['variants']:
            _, disks = regular_disks(maps['profiles'][-1], variant)
            for source_nodes in config['source_half_gauss_nodes']:
                print(f"Source projection {variant['id']}, half-rule {source_nodes}", flush=True)
                p = project_even_source(disks, config['shell_radii_kpc'], max(config['orders']), source_nodes)
                write(f"coefficients_{variant['id']}_{source_nodes}.json", p)
                for evaluation_nodes in config['evaluation_half_gauss_nodes']:
                    print(f"  Independent evaluation half-rule {evaluation_nodes}", flush=True)
                    metrics = projection_metrics(disks, p, config['orders'], evaluation_nodes)
                    for row in metrics:
                        rows.append({'variant': variant, 'source_half_gauss_nodes': source_nodes, **row})
        primary = [r for r in rows if r['source_half_gauss_nodes'] == max(config['source_half_gauss_nodes'])
                   and r['evaluation_half_gauss_nodes'] == max(config['evaluation_half_gauss_nodes'])]
        summary = []
        for row in primary:
            maximum = {key: float(np.max(abs(row[key]))) for key in ['density_L1_fraction_error', 'gradient_L1_fraction_error',
                       'negative_density_fraction', 'relative_shell_mass_error']}
            alternatives = [r for r in rows if r['variant'] == row['variant'] and r['maximum_order'] == row['maximum_order']]
            quadrature = max(float(np.max(abs(np.asarray(a[key])-row[key]))) for a in alternatives for key in maximum)
            target = config['numerical_targets']
            passed = (maximum['density_L1_fraction_error'] < target['maximum_density_L1_fraction_error'] and
                      maximum['gradient_L1_fraction_error'] < target['maximum_gradient_L1_fraction_error'] and
                      maximum['negative_density_fraction'] < target['maximum_negative_density_fraction'] and
                      quadrature < target['maximum_absolute_quadrature_metric_change'])
            summary.append({'variant': row['variant'], 'maximum_order': row['maximum_order'], 'maximum_errors': maximum,
                            'maximum_quadrature_metric_change': quadrature, 'within_registered_source_targets': passed})
        if any(sha256((ROOT/p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
            raise RuntimeError('Input changed during source audit')
        write('result.json', {**provenance, 'rows': rows, 'summary': summary, 'gravity_rejection': False,
                               'full_source_or_field_validation': False})
        write('receipt.json', {'status': 'SOURCE_RESOLUTION_DIAGNOSTIC_RETAINED',
                'result_sha256': sha256((args.output/'result.json').read_bytes()).hexdigest()})
        print(json.dumps(serial(summary)))
    except Exception as exc:
        write('failure.json', {'status': 'SOURCE_AUDIT_EXECUTION_FAILURE_RETAINED', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
