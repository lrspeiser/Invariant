"""Execute frozen source refinement on fixed observed NGC2976 image cells."""
from __future__ import annotations
import argparse
import datetime
import sys
import time
import unittest
from pathlib import Path
import numpy as np
import cupy as cp
from threadpoolctl import threadpool_limits
from mond_atlas_common import ROOT, read_json, write_json, write_csv, digest
from mond_atlas_source_resolution import cell_projection_matrix, project, fit_fixed_image
from mond_atlas_source_projection import weighted_relative_rms


def run(config_path, output, private):
    config = read_json(config_path)
    if config['admission_disposition'] != 'SOURCE_BLOCKED':
        raise ValueError('source-only disposition required')
    package = ROOT/'work/gravity-first-principles/mond-atlas-source-resolution-001'
    for path, expected in read_json(package/'freeze.json')['bindings'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('changed frozen input: '+path)
    if output.exists() or private.exists():
        raise FileExistsError('immutable output exists')
    if (not output.is_relative_to(ROOT/'work/gravity-first-principles')
            or not private.is_relative_to(ROOT/'work/private')):
        raise ValueError('output outside research roots')
    output.mkdir(parents=True)
    private.mkdir(parents=True)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(output/'execution-start.json', dict(started_utc=started,
        config_sha256=digest(config_path), runner_sha256=digest(Path(__file__)),
        operator_sha256=digest(ROOT/config['operator']),
        tests_sha256=digest(ROOT/config['benchmark_tests']),
        source_packet_opened=False, observed_motion_or_lensing_response_opened=False))
    sys.path.insert(0, str(ROOT/'tests'))
    suite = unittest.defaultTestLoader.loadTestsFromName('test_mond_atlas_source_resolution')
    with (output/'unit-tests.log').open('w',encoding='utf-8') as stream:
        tests = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    if not tests.wasSuccessful() or tests.skipped:
        write_json(output/'failure.json', dict(disposition='BENCHMARK_FAILED',source_packet_opened=False))
        raise ValueError('independent finite-cell benchmark failed')
    source_opened = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with np.load(ROOT/config['source_packet']) as f:
        packet = {key:f[key] for key in f.files}
    cfg = config['source_fit']
    rows, assets, annuli = [], [], []
    start = time.perf_counter()
    peak_pool = 0
    cp.get_default_memory_pool().set_limit(size=1024**3)
    with threadpool_limits(limits=1):
        for component in config['components']:
            axis = packet[component+'_axis']
            spacing = float(axis[1]-axis[0])
            radius = np.hypot(*np.meshgrid(axis,axis,indexing='ij'))
            mean, coverage = packet[component+'_mean'], packet[component+'_coverage']
            valid = np.isfinite(mean)&(coverage >= cfg['minimum_cell_coverage'])
            target = np.where(valid,mean,0)
            weight = np.where(valid&(radius < cfg['fitted_radius_kpc']),np.clip(coverage,0,1),0)
            evaluation = np.where(valid&(radius < cfg['reported_radius_kpc']),np.clip(coverage,0,1),0)
            floor = float(np.sqrt(np.sum(evaluation*np.minimum(target,0)**2)/np.sum(evaluation*target**2)))
            for height in config['height_grid_kpc']:
                previous = None
                for factor in config['source_refinement_factors']:
                    case_id = f'{component}-h{str(height).replace(".","p")}-f{factor}'
                    latent = np.linspace(axis[0],axis[-1],(len(axis)-1)*factor+1)
                    latent_spacing = spacing/factor
                    support = np.hypot(*np.meshgrid(latent,latent,indexing='ij')) < cfg['source_support_radius_kpc']
                    left = cell_projection_matrix(axis,spacing,latent,latent_spacing,0)
                    right = cell_projection_matrix(axis,spacing,latent,latent_spacing,
                        height*np.tan(np.deg2rad(config['inclination_deg'])))
                    case_start = time.perf_counter()
                    recovered, fit = fit_fixed_image(target,weight,left,right,support,factor,
                        cfg['regularization'],cfg['max_iterations'],cfg['projected_gradient_relative_rms_tolerance'],backend='cupy')
                    prediction = project(recovered,left,right)
                    after = weighted_relative_rms(prediction,target,evaluation)
                    peak_pool = max(peak_pool,cp.get_default_memory_pool().total_bytes())
                    path = private/(case_id+'.npz')
                    np.savez_compressed(path,observed_axis=axis,latent_axis=latent,
                        intrinsic_effective_surface=recovered,projected_surface=prediction,
                        target=target,source_mean=mean,coverage=coverage,fit_weight=weight,
                        evaluation_weight=evaluation,support=support,vertical_layers=[[1.,height]])
                    row = dict(id=case_id,component=component,height_kpc=height,refinement_factor=factor,
                        source_node_spacing_kpc=latent_spacing,observed_cell_width_kpc=spacing,
                        latent_nodes_per_axis=len(latent),observed_cells_per_axis=len(axis),
                        refitted_source_image_rms=after,nonnegative_image_floor=floor,
                        excess_squared_rms_above_nonnegative_floor=max(0,after**2-floor**2),
                        prediction_change_relative_to_target_rms=None if previous is None else
                            float(np.sqrt(np.sum(evaluation*(prediction-previous)**2)/np.sum(evaluation*target**2))),
                        gross_refitted_mismatch=after > config['gross_source_image_mismatch_threshold'],
                        optimizer_converged=fit['converged'],iterations=fit['iterations'],
                        scaled_projected_gradient_rms=fit['scaled_projected_gradient_rms'],
                        normalized_objective=fit['history'][-1]['objective'],
                        conditional_source_integral=float(recovered.sum()*latent_spacing**2*1e6),
                        fitted_cells=int(np.sum(weight > 0)),reported_cells=int(np.sum(evaluation > 0)),
                        negative_reported_cells=int(np.sum((evaluation > 0)&(target < 0))),
                        elapsed_seconds=time.perf_counter()-case_start,
                        noise_likelihood=False,independent_prediction=False)
                    rows.append(row)
                    assets.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=digest(path),id=case_id))
                    write_json(output/(case_id+'-optimizer.json'),fit)
                    for rmin in np.arange(0,cfg['reported_radius_kpc'],.5):
                        w = np.where((radius >= rmin)&(radius < rmin+.5),evaluation,0)
                        if np.sum(w*target**2) > 0:
                            annuli.append(dict(id=case_id,radius_inner_kpc=float(rmin),
                                relative_image_rms=weighted_relative_rms(prediction,target,w)))
                    previous = prediction
                    print(f'{case_id}: RMS={after:.6f}, converged={fit["converged"]}, iterations={fit["iterations"]}',flush=True)
                    cp.get_default_memory_pool().free_all_blocks()
    write_csv(output/'source-resolution.csv',rows)
    write_csv(output/'source-resolution-annuli.csv',annuli)
    write_json(output/'summary.json',dict(status='FIXED_IMAGE_SOURCE_RESOLUTION_DIAGNOSTIC',
        disposition='SOURCE_BLOCKED',object_id=config['object_id'],config=config,
        config_sha256=digest(config_path),runner_sha256=digest(Path(__file__)),
        operator_sha256=digest(ROOT/config['operator']),source_packet_opened_utc=source_opened,
        independent_benchmarks_passed_before_source_open=True,tests_passed=tests.testsRun,
        rows=rows,assets=assets,source_models=len(rows),
        all_optimizers_converged=all(r['optimizer_converged'] for r in rows),
        cuda_device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cupy_version=cp.__version__,
        peak_default_pool_bytes=peak_pool,memory_limit_bytes=1024**3,
        new_observed_motion_scores=0,new_lensing_scores=0,new_gravity_fields=0,
        source_3d_observed=False,elapsed_seconds=time.perf_counter()-start))
    write_json(output/'artifact-hashes.json',{p.relative_to(ROOT).as_posix():digest(p)
        for p in sorted(output.iterdir()) if p.is_file()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/mond_atlas_source_resolution_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--private',type=Path,required=True)
    args = parser.parse_args()
    run(args.config.resolve(),args.output.resolve(),args.private.resolve())
