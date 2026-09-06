"""Audit existing atlas assets and this interpreter; no model fitting or downloads."""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from mond_atlas_common import ROOT, digest, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a new immutable output directory')
    inputs = [
        'work/gravity-first-principles/mond-atlas-catalog-004/assets.csv',
        'work/gravity-first-principles/mond-atlas-catalog-004/sources.json',
        'work/gravity-first-principles/mond-atlas-execution-010/execution-status.json',
        'configs/lensing_direct_observable_evaluator_readiness.json',
        'runs/engine/lensing-direct-observable-evaluator-readiness.json',
        'scripts/run_mond_atlas_ml_readiness.py',
    ]
    with (ROOT / inputs[0]).open(newline='', encoding='utf-8-sig') as stream:
        assets = list(csv.DictReader(stream))
    sources = read_json(ROOT / inputs[1])
    pattern = re.compile(r'lens(?:ing)?|slacs|bells|euclid', re.I)
    lens_assets = [row for row in assets if pattern.search(' '.join(row.values()))]
    lens_sources = [row for row in sources if pattern.search(json.dumps(row))]
    old_config = read_json(ROOT / inputs[3])
    old_receipt = read_json(ROOT / inputs[4])
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                              '--format=csv,noheader'], capture_output=True, text=True,
                             timeout=20, check=False)
        gpu_result = dict(returncode=gpu.returncode, stdout=gpu.stdout.strip(), stderr=gpu.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        gpu_result = dict(error=str(exc))
    result = dict(
        status='DEVELOPMENT_PATTERN_SEARCH_POSSIBLE_FULL_INFERENCE_NOT_READY',
        audited_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Catalog-004 assets/sources, execution-010 counts, historical lensing readiness, and active interpreter. Not an exhaustive whole-machine holdings search.',
        bindings={p: digest(ROOT / p) for p in inputs},
        catalog_asset_rows=len(assets),
        catalog_asset_roles=dict(Counter(row['role'] for row in assets)),
        catalog_assets_missing_on_disk=[row['path'] for row in assets if not (ROOT / row['path']).is_file()],
        lensing_candidate_asset_rows=lens_assets,
        lensing_candidate_source_records=lens_sources,
        matched_lensing_sample_in_audited_atlas=False,
        historical_lensing_config_source_packet_count=len(old_config['authorized_real_source_packets']),
        historical_lensing_receipt_scientific_pass=old_receipt.get('scientific_pass_claimed'),
        historical_lensing_config_is_not_current_user_authorization=True,
        atlas_counts=read_json(ROOT / inputs[2])['counts'],
        runtime=dict(executable=sys.executable, version=platform.python_version(),
            importable_modules={name: importlib.util.find_spec(name) is not None for name in
                ['numpy', 'torch', 'cupy', 'jax', 'scipy', 'sklearn', 'xgboost', 'lightgbm']},
            gpu=gpu_result, cuda_kernel_or_training_executed=False,
            conclusion='Device visibility is separate from a tested ML runtime. This audit does not test kernels or other interpreters.'),
        public_extensions=[
            dict(name='Euclid Q1', url='https://www.euclid-ec.org/science/q1/',
                 status='Public image/spectra/catalog release checked; no assets acquired by this audit; no match to the HI pilots established'),
            dict(name='Relativistic MOND lensing example', url='https://arxiv.org/abs/astro-ph/0403694',
                 status='Theory reference, not evidence that a complete modern model passes all data'),
            dict(name='QUMOND field equations', url='https://arxiv.org/abs/0911.5464',
                 status='Combined-source nonlinear field model; requires explicit relativistic light-propagation prescription for lensing'),
        ],
        no_new_observed_gravity_scores=True,
        goal_complete=False,
    )
    output.mkdir(parents=True)
    write_json(output / 'readiness.json', result)
    print(json.dumps({k: result[k] for k in ['catalog_asset_rows', 'catalog_asset_roles',
        'catalog_assets_missing_on_disk', 'matched_lensing_sample_in_audited_atlas', 'runtime']}))


if __name__ == '__main__':
    main()
