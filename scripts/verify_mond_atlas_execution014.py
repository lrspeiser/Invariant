"""Recheck the exact source/motion milestone and its publication test subset.

Run from the repository root with --output pointing to a new report directory.
Requires the separately retained private observations and synthetic packets.
This is an integrity/integration audit, not scientific observational admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'work/gravity-first-principles'
MILESTONE = REPORTS / 'mond-atlas-execution-014'
TEST_MODULES = [
    'astrometry', 'background_support', 'baryon_recovery', 'blocked',
    'channel_order', 'emission_exclusion', 'force_sampling', 'mask_injection',
    'native_selection', 'native_spectral', 'nodal', 'noise', 'noise_mean',
    'noise_robustness', 'offline', 'pattern_learning', 'preprocessing',
    'projection', 'rectangular', 'smoothing_null', 'vertical',
    'lensing_pilot', 'motion_controls', 'stellar_transfer', 'stellar_transfer_guard',
]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class Audit:
    def __init__(self):
        self.cache = {}
        self.checks = []

    def check(self, path, expected, size=None, label=None):
        path = Path(str(path).replace('\\', '/'))
        path = path if path.is_absolute() else ROOT / path
        path = path.resolve()
        if path not in self.cache:
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            self.cache[path] = (digest, path.stat().st_size)
        digest, actual_size = self.cache[path]
        if digest != expected or (size is not None and size != actual_size):
            raise ValueError(f'Integrity mismatch: {path}')
        self.checks.append(dict(path=path.relative_to(ROOT).as_posix(),
                                sha256=digest, bytes=actual_size, role=label))

    def records(self, records, label):
        for record in records:
            self.check(record['path'], record['sha256'], record.get('bytes'), label)

    def mapping(self, bindings, label):
        for path, expected in bindings.items():
            self.check(path, expected, label=label)

    def nested_records(self, value, label):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                self.check(value['path'], value['sha256'], value.get('bytes'), label)
            for child in value.values():
                self.nested_records(child, label)
        elif isinstance(value, list):
            for child in value:
                self.nested_records(child, label)


def verify(output):
    output = output.resolve()
    if not output.is_relative_to(REPORTS.resolve()):
        raise ValueError('Output must remain under the research report directory')
    output.mkdir(parents=True, exist_ok=False)
    audit = Audit()
    prior = read(REPORTS / 'mond-atlas-execution-013/publication-manifest.json')
    substitutions = {
        'docs/MOND_OBSERVATION_ATLAS_GOAL.md': MILESTONE / 'prior-goal-handoff.md',
        'docs/GRAVITY_PATTERN_SYSTEM_TASKS.md': MILESTONE / 'prior-task-plan.md',
    }
    for row in prior['files']:
        archive = substitutions.get(row['path'])
        target = archive if archive is not None and archive.is_file() else row['path']
        audit.check(target, row['sha256'], row['bytes'], 'prior-013-manifest')

    lens = REPORTS / 'mond-atlas-lensing-pilot-001'
    motion = REPORTS / 'mond-atlas-motion-controls-001'
    lens_manifest = read(lens / 'deliverable-manifest.json')['files']
    motion_manifest = read(motion / 'publication-manifest.json')['files']
    audit.records(lens_manifest, 'lensing-publication-package')
    audit.records(motion_manifest, 'motion-publication-package')
    lens_sources = read(lens / 'replay-002/source-manifest.json')
    audit.records(lens_sources, 'lensing-original-download')
    audit.nested_records(read(lens / 'replay-002/systems.json'), 'lensing-derived-packets')
    for path, expected in read(lens / 'replay-002/input-bindings.json')['bindings'].items():
        # This task started against an earlier mutable handoff. Its exact bytes
        # are already archived in execution-013; never waive the checksum.
        target = (REPORTS / 'mond-atlas-execution-013/prior-goal-handoff.md'
                  if path == 'docs/MOND_OBSERVATION_ATLAS_GOAL.md' else path)
        audit.check(target, expected, label='lensing-input-binding')
    for name in ('measurements.csv', 'selected-source-tables.json', 'source-manifest.json'):
        if (lens / 'ingest-001' / name).read_bytes() != (lens / 'replay-002' / name).read_bytes():
            raise ValueError(f'Lensing offline replay differs: {name}')

    stellar_samples = 0
    for version in ('001', '002'):
        packet = REPORTS / f'mond-atlas-stellar-transfer-{version}'
        audit.mapping(read(packet / 'prospective-bindings.json')['bindings'], 'stellar-prospective')
        summary = read(packet / 'summary.json')
        if summary['errors'] or summary['new_motion_scores'] != 0:
            raise ValueError('Unexpected stellar run error or response score')
        for row in summary['results']:
            audit.mapping(row['source_bindings'], 'stellar-image-source')
            audit.check(row['private_samples'], row['private_samples_sha256'], label='stellar-private-samples')
            stellar_samples += 1
    audit.mapping(read(REPORTS / 'mond-atlas-stellar-transfer-findings-001/summary.json')['bindings'],
                  'stellar-comparison-report')

    motion_samples = 0
    for run in ('run-001', 'run-002'):
        packet = motion / run
        audit.mapping(read(packet / 'artifact-hashes.json'), 'motion-executed-artifact')
        for case in sorted(packet.glob('case-*.json')):
            audit.nested_records(read(case)['synthetic_arrays'], 'motion-private-synthetic')
            motion_samples += 1
    start = read(motion / 'run-002/execution-start.json')
    audit.mapping(start['implementation_hashes'], 'motion-final-implementation')
    for key, path in {
        'config_sha256': 'configs/mond_atlas_motion_controls_v1.json',
        'preflight_sha256': str(motion / 'PREFLIGHT.md'),
        'shared_cube_sha256': 'scripts/mond_atlas_cube.py',
    }.items():
        audit.check(path, start['verified_input_hashes'][key], label='motion-pre-response-freeze')
    gate = read(motion / 'run-002/response-access-gate.json')
    if not gate['all_required_numerical_gates_passed'] or gate['observational_scoring_allowed']:
        raise ValueError('Motion numerical/access gate differs')
    audit.check(motion / 'run-002/numerical-controls.json', gate['controls_sha256'], label='motion-controls')

    sys.path.insert(0, str(ROOT / 'scripts'))
    sys.path.insert(0, str(ROOT / 'tests'))
    modules = ['test_mond_atlas_' + suffix for suffix in TEST_MODULES]
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    with (output / 'unit-tests.log').open('w', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    passed = result.wasSuccessful() and not result.skipped
    receipt = {
        'status': 'PASS' if passed else 'FAIL',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'tests': result.testsRun, 'failures': len(result.failures),
        'errors': len(result.errors), 'skipped': len(result.skipped),
        'test_modules': modules, 'prior_manifest_files_verified': len(prior['files']),
        'prior_mutable_context_archives': {k: v.relative_to(ROOT).as_posix()
                                          for k, v in substitutions.items() if v.is_file()},
        'lensing_manifest_entries_rehashed': len(lens_manifest),
        'motion_manifest_entries_rehashed': len(motion_manifest),
        'lensing_original_downloads_rehashed': len(lens_sources),
        'lensing_original_download_bytes': sum(row['bytes'] for row in lens_sources),
        'stellar_private_sample_packets_rehashed': stellar_samples,
        'motion_private_synthetic_packets_rehashed': motion_samples,
        'unique_files_rehashed': len(audit.cache),
        'integrity_checks': audit.checks,
        'observed_full_field_admission': False,
        'new_gravity_law_established': False,
        'limitations': 'Integrity and numerical checks do not validate unknown source depth, absolute mass, observed covariance or relativistic light propagation.',
    }
    (output / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('integrity_checks', 'test_modules')}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    verify(parser.parse_args().output)
