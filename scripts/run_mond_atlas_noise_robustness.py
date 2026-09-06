"""Retain every declared background split, including failures and duplicates."""
from __future__ import annotations
import argparse, copy, hashlib, io, unittest
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT, digest, read_json, write_json, write_csv
import run_mond_atlas_noise as noise


def reversed_masks(east, north, config):
    train, test = noise.masks(east, north, config)
    yy, xx = np.indices(east.shape)
    # Recover the full validation block interiors by selecting their block ids.
    side = config['block_side_pixels']; a, b = config['block_interior']
    block = (yy//side)*1000+xx//side
    lo, hi = config['sky_annulus_arcsec']; radius = np.hypot(east, north)
    interior = (xx%side>=a)&(xx%side<b)&(yy%side>=a)&(yy%side<b)
    candidate = (radius>lo)&(radius<hi)&(xx>6)&(yy>6)&(xx<east.shape[1]-7)&(yy<east.shape[0]-7)&interior
    # Forward train already identifies the selected blocks without relying on
    # whether sparse validation happened to sample every small boundary block.
    reverse_train = candidate & ~np.isin(block, np.unique(block[train]))
    stride = config['validation_stride_pixels']
    reverse_test = train & (xx%stride==0)&(yy%stride==0)
    return reverse_train, reverse_test


def evaluate(packet, config, train, test):
    """Reuse the frozen estimator while supplying a precomputed geometry split."""
    original = noise.masks
    try:
        noise.masks = lambda east, north, cfg: (train, test)
        return noise.check_packet(packet, config)
    finally:
        noise.masks = original


def run(config_path, output):
    config = read_json(config_path)
    if config['admission_disposition'] != 'SOURCE_BLOCKED' or config['best_split_selection_permitted']:
        raise ValueError('invalid prospective robustness scope')
    parent = read_json(ROOT/config['parent_protocol']); previous = read_json(ROOT/config['parent_summary'])
    for path, expected in previous['input_hashes'].items():
        if digest(ROOT/path) != expected: raise ValueError('parent input changed: '+path)
    for name, expected in previous['code_hashes'].items():
        if digest(ROOT/'scripts'/name) != expected: raise ValueError('frozen covariance code changed: '+name)
    if output.exists(): raise FileExistsError('immutable output')
    output.mkdir(parents=True)
    paths = [config_path, ROOT/config['parent_protocol'], ROOT/config['parent_summary'], Path(__file__),
             ROOT/'scripts/run_mond_atlas_noise.py', ROOT/'scripts/mond_atlas_cube.py']
    bindings = {str(p.relative_to(ROOT)):digest(p) for p in paths}
    write_json(output/'prospective-bindings.json', dict(config=config, bindings=bindings))
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_mond_atlas_noise*.py')
    log = io.StringIO(); tests = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (output/'validation.log').write_text(log.getvalue(), encoding='utf-8', newline='\n')
    if not tests.wasSuccessful(): raise RuntimeError(log.getvalue())
    rows = []; galaxies = []; failures = []
    for audit in read_json(ROOT/parent['source_audit']):
        name = audit['name']; path = ROOT/parent['source_packets']/(name+'.npz')
        previous_galaxy = read_json((ROOT/config['parent_summary']).parent/(name+'.json'))
        if digest(path) != previous_galaxy['packet_sha256']: raise ValueError('cube packet changed: '+name)
        with np.load(path, allow_pickle=False) as packet:
            east = packet['east']; north = packet['north']
            lo, hi = parent['sky_annulus_arcsec']; radius = np.hypot(east, north)
            background = (radius>lo)&(radius<hi)
            data = dict(cube=np.where(background[None,:,:], packet['cube'], 0.), east=east, north=north)
        seen = {}; metrics = []; details = []
        for seed in config['split_seeds']:
            cfg = copy.deepcopy(parent); cfg['split_seed'] = seed
            for direction in config['calibration_validation_directions']:
                label = str(seed)+'-'+direction
                try:
                    train, test = noise.masks(east,north,cfg) if direction=='forward' else reversed_masks(east,north,cfg)
                    identity = hashlib.sha256(train.tobytes()+test.tobytes()).hexdigest()
                    duplicate = seen.get(identity); seen.setdefault(identity, label)
                    result, arrays = evaluate(data,cfg,train,test)
                    record = dict(galaxy=name, split=label, split_sha256=identity, duplicate_of=duplicate,
                        diagnostic_pass=result['diagnostic_pass'], mean_square=result['joint_validation_mean_square'],
                        channel_lag1=result['joint_validation_channel_lag1'],
                        quadrant_min=min(q['whitened_mean_square'] for q in result['quadrants']),
                        quadrant_max=max(q['whitened_mean_square'] for q in result['quadrants']),
                        failed_gates=';'.join(k for k,v in result['diagnostic_gates'].items() if not v))
                    rows.append(record); details.append(dict(**record, result=result))
                    if duplicate is None: metrics.append(record)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    failure = dict(galaxy=name, split=label, error=str(exc)); failures.append(failure)
                    details.append(dict(**failure, execution_failed=True))
        summary = dict(galaxy=name, packet_sha256=digest(path), declared_partitions=len(config['split_seeds'])*len(config['calibration_validation_directions']),
            successful_evaluations=len(details)-sum(bool(d.get('execution_failed')) for d in details),
            unique_partitions=len(metrics), unique_passes=sum(r['diagnostic_pass'] for r in metrics),
            unique_failures=sum(not r['diagnostic_pass'] for r in metrics),
            all_declared_splits_pass=bool(metrics) and all(r['diagnostic_pass'] for r in metrics) and not any(d.get('execution_failed') for d in details),
            mean_square_min=min((r['mean_square'] for r in metrics),default=None),mean_square_max=max((r['mean_square'] for r in metrics),default=None),
            channel_lag1_min=min((r['channel_lag1'] for r in metrics),default=None),channel_lag1_max=max((r['channel_lag1'] for r in metrics),default=None),
            failed_gates=';'.join(sorted({g for r in metrics for g in r['failed_gates'].split(';') if g})))
        galaxies.append(summary); write_json(output/(name+'.json'), dict(summary=summary, partitions=details))
        print(summary, flush=True)
    write_csv(output/'partitions.csv',rows); write_csv(output/'galaxies.csv',galaxies)
    write_json(output/'summary.json',dict(status='BACKGROUND_SPLIT_ROBUSTNESS_EXECUTED',
        admission_disposition='SOURCE_BLOCKED', config=config, bindings=bindings, galaxies=len(galaxies),
        partition_evaluations=len(rows),execution_failures=failures,
        split_stable_pass=[r['galaxy'] for r in galaxies if r['all_declared_splits_pass']],
        split_sensitive_or_failed=[r['galaxy'] for r in galaxies if not r['all_declared_splits_pass']],
        validated_noise_unit_tests=tests.testsRun, galaxy_motion_scores_computed=0,
        admitted_galaxy_cube_predictions=0, goal_complete=False))
    print(dict(galaxies=len(galaxies), partition_evaluations=len(rows), execution_failures=len(failures)),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/mond_atlas_noise_robustness_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.config.resolve(),args.output.resolve())
