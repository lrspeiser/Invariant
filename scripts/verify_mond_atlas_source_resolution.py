"""Replay saved source fits on the CPU and verify fixed observation weights."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from mond_atlas_common import ROOT, read_json, write_json, digest
from mond_atlas_source_resolution import cell_projection_matrix, project, adjoint, roughness_gradient
from mond_atlas_source_projection import weighted_relative_rms


def verify(summary_path, output):
    if output.exists():
        raise FileExistsError('immutable verification exists')
    summary = read_json(summary_path)
    config = summary['config']
    if digest(ROOT/config['source_packet']) != config['source_packet_sha256']:
        raise ValueError('original observation packet changed')
    with np.load(ROOT/config['source_packet']) as f:
        original = {k:f[k] for k in f.files}
    rows = {r['id']:r for r in summary['rows']}
    if len(rows) != 36 or len(summary['assets']) != 36:
        raise ValueError('incomplete refinement cases')
    checks = []
    with threadpool_limits(limits=1):
        for asset in summary['assets']:
            path = ROOT/asset['path']
            if digest(path) != asset['sha256']:
                raise ValueError('private packet hash changed')
            row = rows[asset['id']]
            c = row['component']
            info = read_json(summary_path.parent/(row['id']+'-optimizer.json'))
            with np.load(path) as f:
                saved = {k:f[k] for k in f.files}
            axis, latent = saved['observed_axis'], saved['latent_axis']
            np.testing.assert_array_equal(axis,original[c+'_axis'])
            np.testing.assert_array_equal(saved['source_mean'],original[c+'_mean'])
            np.testing.assert_array_equal(saved['coverage'],original[c+'_coverage'])
            r = np.hypot(*np.meshgrid(axis,axis,indexing='ij'))
            mean,coverage = original[c+'_mean'],original[c+'_coverage']
            valid = np.isfinite(mean)&(coverage >= config['source_fit']['minimum_cell_coverage'])
            expected_target = np.where(valid,mean,0)
            expected_fit = np.where(valid&(r < config['source_fit']['fitted_radius_kpc']),np.clip(coverage,0,1),0)
            expected_eval = np.where(valid&(r < config['source_fit']['reported_radius_kpc']),np.clip(coverage,0,1),0)
            for name,value in [('target',expected_target),('fit_weight',expected_fit),('evaluation_weight',expected_eval)]:
                np.testing.assert_array_equal(saved[name],value)
            d,h = row['observed_cell_width_kpc'],row['source_node_spacing_kpc']
            a = cell_projection_matrix(axis,d,latent,h,0)
            b = cell_projection_matrix(axis,d,latent,h,row['height_kpc']*np.tan(np.deg2rad(config['inclination_deg'])))
            source = saved['intrinsic_effective_surface']
            expected_support = np.hypot(*np.meshgrid(latent,latent,indexing='ij')) < config['source_fit']['source_support_radius_kpc']
            np.testing.assert_array_equal(saved['support'],expected_support)
            if np.min(source) < 0 or np.any(source[~expected_support] != 0):
                raise ValueError('physical source constraint violation')
            prediction = project(source,a,b)
            projection_error = float(np.max(np.abs(prediction-saved['projected_surface']))/max(1.,np.max(np.abs(prediction))))
            if projection_error > 3e-12:
                raise ValueError('saved projection mismatch')
            rms = weighted_relative_rms(prediction,expected_target,expected_eval)
            if abs(rms-row['refitted_source_image_rms']) > 1e-12:
                raise ValueError('reported RMS mismatch')
            scale = info['normalizing_intensity']
            s = source/scale
            grad = adjoint(expected_fit*(prediction-expected_target)/scale,a,b)
            grad += config['source_fit']['regularization']*roughness_gradient(s)
            L = info['lipschitz_bound']
            step = s-np.where(expected_support,np.maximum(s-grad/L,0),0)
            stationarity = float(np.sqrt(np.mean(step**2))*L*row['refinement_factor']**2)
            if abs(stationarity-row['scaled_projected_gradient_rms']) > 1e-10:
                raise ValueError('stationarity replay mismatch')
            if row['optimizer_converged'] and stationarity > config['source_fit']['projected_gradient_relative_rms_tolerance']+1e-10:
                raise ValueError('false convergence claim')
            checks.append(dict(id=row['id'],private_sha256=asset['sha256'],
                same_observation_and_weights=True,relative_projection_max_error=projection_error,
                rms_replay_error=abs(rms-row['refitted_source_image_rms']),
                cpu_scaled_stationarity=stationarity,nonnegative_and_support_constraints=True))
    write_json(output,dict(status='PASS',summary_sha256=digest(summary_path),
        verification_script_sha256=digest(Path(__file__)),private_packets_rehashed=len(checks),
        fixed_observation_cases_checked=len(checks),cpu_projection_and_stationarity_replays=len(checks),
        source_3d_observed=False,new_observed_gravity_scores=0,checks=checks))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,default=ROOT/'work/gravity-first-principles/mond-atlas-source-resolution-001/run-001/summary.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    verify(args.summary.resolve(),args.output.resolve())
