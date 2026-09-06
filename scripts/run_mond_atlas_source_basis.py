"""Audit the cell-to-node source change and fit one consistent image/field basis."""
from __future__ import annotations
import argparse,io,unittest
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
from mond_atlas_nodal_projection import nodal_projection_matrix,project_nodes,fit_nodes
from mond_atlas_source_projection import weighted_relative_rms


def run(config_path,output,private):
    config=read_json(config_path);assert config['admission_disposition']=='SOURCE_BLOCKED'
    protocol=read_json(ROOT/config['previous_source_protocol']);cfg=protocol['source_fit'];audit=read_json(ROOT/protocol['source_audit'])
    source_path=ROOT/audit['source_packet'];assert digest(source_path)==audit['source_packet_sha256']
    products={}
    for key in ('single_projection_run','mixed_projection_run'):
        summary=read_json(ROOT/config[key])
        for p in summary['products']:products[p['path'].replace('\\','/')]=p['sha256']
    for case in config['cases']:assert digest(ROOT/case['previous_source'])==products[case['previous_source']]
    if output.exists() or private.exists():raise FileExistsError('immutable output')
    output.mkdir(parents=True);private.mkdir(parents=True)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_mond_atlas_nodal.py')
    log=io.StringIO();tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (output/'validation.log').write_text(log.getvalue(),encoding='utf-8',newline='\n')
    if not tests.wasSuccessful():raise RuntimeError(log.getvalue())
    with np.load(source_path) as f:packet={k:f[k] for k in f.files}
    inc=audit['protocol']['geometry']['inclination_deg'];rows=[];assets=[]
    for case in config['cases']:
        name=case['component'];axis=packet[name+'_axis'];d=float(axis[1]-axis[0]);x,y=np.meshgrid(axis,axis,indexing='ij');r=np.hypot(x,y)
        mean=packet[name+'_mean'];coverage=packet[name+'_coverage'];valid=np.isfinite(mean)&(coverage>=cfg['minimum_cell_coverage'])
        target=np.where(valid,mean,0);weight=np.where(valid&(r<cfg['fitted_radius_kpc']),np.clip(coverage,0,1),0)
        evaluation=np.where(valid&(r<cfg['reported_radius_kpc']),np.clip(coverage,0,1),0);support=r<cfg['source_support_radius_kpc']
        left=nodal_projection_matrix(len(axis),d,0,inc)
        right=sum(f*nodal_projection_matrix(len(axis),d,h,inc) for f,h in case['vertical_layers'])
        with np.load(ROOT/case['previous_source']) as f:old=f['intrinsic_effective_surface'];old_prediction=f['projected_surface']
        actual_old_basis=project_nodes(old,left,right)
        recovered,fit=fit_nodes(target,weight,left,right,support,cfg['regularization'],cfg['max_iterations'],cfg['projected_gradient_relative_rms_tolerance'])
        predicted=project_nodes(recovered,left,right)
        before=weighted_relative_rms(old_prediction,target,evaluation);changed=weighted_relative_rms(actual_old_basis,target,evaluation);after=weighted_relative_rms(predicted,target,evaluation)
        path=private/(case['id']+'.npz')
        np.savez_compressed(path,axis=axis,intrinsic_effective_surface=recovered,projected_surface=predicted,
            source_mean=mean,coverage=coverage,fit_weight=weight,evaluation_weight=evaluation,vertical_layers=case['vertical_layers'])
        result=dict(id=case['id'],component=name,old_cell_basis_image_rms=before,old_values_with_actual_field_basis_image_rms=changed,
            refitted_common_basis_image_rms=after,basis_change_image_difference=weighted_relative_rms(actual_old_basis,old_prediction,evaluation),
            old_source_integral=float(old.sum()*d*d*1e6),new_source_integral=float(recovered.sum()*d*d*1e6),
            optimizer_converged=fit['converged'],iterations=fit['iterations'],noise_calibrated_likelihood=False,source_posterior_admitted=False)
        rows.append(result);assets.append(dict(path=str(path.relative_to(ROOT)),sha256=digest(path),id=case['id']))
        write_json(output/(case['id']+'-optimizer.json'),fit);print(result,flush=True)
    write_csv(output/'source-basis-comparison.csv',rows)
    write_json(output/'summary.json',dict(status='COMMON_IMAGE_AND_FIELD_SOURCE_BASIS_DIAGNOSTIC',admission_disposition='SOURCE_BLOCKED',
        config=config,config_sha256=digest(config_path),source_cases_executed=len(rows),all_optimizers_converged=all(r['optimizer_converged'] for r in rows),
        products=assets,code_hashes={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),ROOT/'scripts/mond_atlas_nodal_projection.py',ROOT/'tests/test_mond_atlas_nodal.py')},
        source_bindings={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/config['previous_source_protocol'],ROOT/protocol['source_audit'],ROOT/config['single_projection_run'],ROOT/config['mixed_projection_run'])},
        independent_benchmark_tests=tests.testsRun,response_files_opened=[],kinematic_response_scores_computed=0,goal_complete=False,
        limitations=['The new inverse and field interpolation share the same bilinear continuum basis. Field grid convergence is still needed for these newly fitted source coefficients.',
            'Original images were rebinned by native pixel-center area assignment; exact native-pixel footprints and spatially varying masks/beams remain an approximation.',
            'Coverage-weighted image RMS is not a source noise likelihood or a confidence interval for heights.',
            'Absolute photometry, source covariance, missing components and physical exterior fields are not supplied by this correction.']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/mond_atlas_source_basis_v1.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--private',type=Path,required=True)
    a=p.parse_args();run(a.config.resolve(),a.output.resolve(),a.private.resolve())
