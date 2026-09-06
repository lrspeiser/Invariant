"""Reuse the validated common source basis on the registered NGC2976 pilot."""
from __future__ import annotations
import argparse
import sys
import time
import unittest
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
from mond_atlas_nodal_projection import nodal_projection_matrix,project_nodes,fit_nodes
from mond_atlas_source_projection import weighted_relative_rms


def run(config_path,output,private):
    config=read_json(config_path)
    if config['admission_disposition']!='SOURCE_BLOCKED':raise ValueError('source-only disposition required')
    for path,expected in read_json(ROOT/'work/gravity-first-principles/mond-atlas-ngc2976-projection-001/freeze.json')['bindings'].items():
        if digest(ROOT/path)!=expected:raise ValueError('changed frozen input: '+path)
    if output.exists() or private.exists():raise FileExistsError('immutable output exists')
    if not output.is_relative_to(ROOT/'work/gravity-first-principles') or not private.is_relative_to(ROOT/'work/private'):raise ValueError('output outside research roots')
    output.mkdir(parents=True);private.mkdir(parents=True)
    write_json(output/'execution-start.json',dict(config_sha256=digest(config_path),runner_sha256=digest(Path(__file__)),
               source_packet_opened=False,observed_motion_response_opened=False))
    sys.path.insert(0,str(ROOT/'tests'))
    suite=unittest.defaultTestLoader.loadTestsFromName('test_mond_atlas_nodal')
    with (output/'unit-tests.log').open('w',encoding='utf-8') as stream:
        tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    if not tests.wasSuccessful() or tests.skipped:raise ValueError('independent source projection benchmark failed')
    if digest(ROOT/config['source_summary'])!=config['source_summary_sha256'] or digest(ROOT/config['source_packet'])!=config['source_packet_sha256']:raise ValueError('registered source changed')
    with np.load(ROOT/config['source_packet']) as f:packet={k:f[k] for k in f.files}
    cfg=config['source_fit'];inc=config['inclination_deg'];rows=[];assets=[];annuli=[];start=time.perf_counter()
    with threadpool_limits(limits=1):
        for component in config['components']:
            axis=packet[component+'_axis'];spacing=float(axis[1]-axis[0]);x,y=np.meshgrid(axis,axis,indexing='ij');radius=np.hypot(x,y)
            mean=packet[component+'_mean'];coverage=packet[component+'_coverage']
            valid=np.isfinite(mean)&(coverage>=cfg['minimum_cell_coverage'])
            target=np.where(valid,mean,0);weight=np.where(valid&(radius<cfg['fitted_radius_kpc']),np.clip(coverage,0,1),0)
            evaluation=np.where(valid&(radius<cfg['reported_radius_kpc']),np.clip(coverage,0,1),0)
            support=radius<cfg['source_support_radius_kpc'];left=nodal_projection_matrix(len(axis),spacing,0,inc)
            base=packet[component+'_annular']
            for height in config['height_grid_kpc']:
                case_id=component+'-h'+str(height).replace('.','p')
                right=nodal_projection_matrix(len(axis),spacing,height,inc)
                original=project_nodes(base,left,right)
                recovered,fit=fit_nodes(target,weight,left,right,support,cfg['regularization'],cfg['max_iterations'],cfg['projected_gradient_relative_rms_tolerance'])
                prediction=project_nodes(recovered,left,right)
                before=weighted_relative_rms(original,target,evaluation);after=weighted_relative_rms(prediction,target,evaluation)
                path=private/(case_id+'.npz')
                np.savez_compressed(path,axis=axis,intrinsic_effective_surface=recovered,projected_surface=prediction,
                                    source_mean=mean,coverage=coverage,fit_weight=weight,evaluation_weight=evaluation,
                                    support=support,vertical_layers=[[1.,height]])
                row=dict(id=case_id,component=component,height_kpc=height,unchanged_source_image_rms=before,
                         refitted_source_image_rms=after,gross_refitted_mismatch=after>config['gross_source_image_mismatch_threshold'],
                         optimizer_converged=fit['converged'],iterations=fit['iterations'],
                         conditional_source_integral=float(recovered.sum()*spacing**2*1e6),
                         fitted_cells=int(np.sum(weight>0)),reported_cells=int(np.sum(evaluation>0)),
                         noise_likelihood=False,independent_prediction=False)
                rows.append(row);assets.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=digest(path),id=case_id))
                write_json(output/(case_id+'-optimizer.json'),fit)
                for rmin in np.arange(0,cfg['reported_radius_kpc'],.5):
                    w=np.where((radius>=rmin)&(radius<rmin+.5),evaluation,0)
                    if np.sum(w*target**2)>0:annuli.append(dict(id=case_id,radius_inner_kpc=float(rmin),relative_image_rms=weighted_relative_rms(prediction,target,w)))
                print(row,flush=True)
    write_csv(output/'source-closure.csv',rows);write_csv(output/'source-closure-annuli.csv',annuli)
    write_json(output/'summary.json',dict(status='REGISTERED_COMMON_BASIS_SOURCE_DIAGNOSTIC',disposition='SOURCE_BLOCKED',
               object_id=config['object_id'],config=config,config_sha256=digest(config_path),runner_sha256=digest(Path(__file__)),
               rows=rows,assets=assets,tests_passed=tests.testsRun,all_optimizers_converged=all(r['optimizer_converged'] for r in rows),
               source_models=len(rows),new_observed_motion_scores=0,new_gravity_fields=0,source_3d_observed=False,
               elapsed_seconds=time.perf_counter()-start))
    write_json(output/'artifact-hashes.json',{p.relative_to(ROOT).as_posix():digest(p) for p in sorted(output.iterdir()) if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/mond_atlas_ngc2976_projection_v1.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--private',type=Path,required=True)
    args=p.parse_args();run(args.config.resolve(),args.output.resolve(),args.private.resolve())
