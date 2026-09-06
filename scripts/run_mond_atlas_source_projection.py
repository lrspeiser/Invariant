"""Audit/reconstruct projected source images; explicitly no kinematic scoring."""
from __future__ import annotations
import argparse,io,time,unittest
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
from mond_atlas_source_projection import projection_matrix,project,fit_nonnegative,weighted_relative_rms


def run(protocol_path,output,private):
    protocol=read_json(protocol_path)
    if protocol['admission_disposition']!='SOURCE_BLOCKED':raise ValueError('source-only audit requires explicit nonadmission')
    audit_path=ROOT/protocol['source_audit'];audit=read_json(audit_path);packet_path=ROOT/audit['source_packet']
    if digest(packet_path)!=audit['source_packet_sha256']:raise ValueError('source packet hash mismatch')
    if output.exists() or private.exists():raise FileExistsError('immutable source audit output')
    output.mkdir(parents=True);private.mkdir(parents=True)
    # Complete independent controls before inspecting source-image closure values.
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_mond_atlas_projection.py')
    log=io.StringIO();test_result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (output/'validation.log').write_text(log.getvalue(),encoding='utf-8',newline='\n')
    if not test_result.wasSuccessful():raise RuntimeError(log.getvalue())
    with np.load(packet_path) as f:packet={k:f[k] for k in f.files}
    source_protocol=audit['protocol'];inc=source_protocol['geometry']['inclination_deg'];cfg=protocol['source_fit']
    rows=[];annuli=[];products=[]
    for component in ('stellar_luminosity','atomic_helium','co21'):
        axis=packet[component+'_axis'];d=float(axis[1]-axis[0]);x,y=np.meshgrid(axis,axis,indexing='ij');r=np.hypot(x,y)
        mean=packet[component+'_mean'];coverage=packet[component+'_coverage']
        valid=np.isfinite(mean)&(coverage>=cfg['minimum_cell_coverage'])
        weight=np.where(valid&(r<cfg['fitted_radius_kpc']),np.clip(coverage,0,1),0)
        evaluation=np.where(valid&(r<cfg['reported_radius_kpc']),np.clip(coverage,0,1),0)
        target=np.where(valid,mean,0);support=r<cfg['source_support_radius_kpc'];start_surface=packet[component+'_annular']
        for height in protocol['height_grid_kpc']:
            start=time.monotonic();label=component+'-h'+str(height).replace('.','p');print('START '+label,flush=True)
            matrix=projection_matrix(len(axis),d,height,inc)
            old_projected=project(start_surface,matrix)
            before=weighted_relative_rms(old_projected,target,evaluation)
            recovered,fit=fit_nonnegative(target,weight,matrix,support,cfg['regularization'],cfg['max_iterations'],cfg['projected_gradient_relative_rms_tolerance'])
            prediction=project(recovered,matrix);after=weighted_relative_rms(prediction,target,evaluation)
            path=private/(label+'.npz')
            np.savez_compressed(path,axis=axis,intrinsic_effective_surface=recovered,projected_surface=prediction,original_projected_surface=old_projected,
                source_mean=mean,coverage=coverage,fit_weight=weight,evaluation_weight=evaluation,support=support)
            result=dict(component=component,height_kpc=height,projection_minor_scale_kpc=float(height*np.tan(np.deg2rad(inc))),
                unchanged_lift_relative_image_rms=before,refitted_source_relative_image_rms=after,
                gross_unchanged_lift_mismatch=before>protocol['benchmarks']['gross_source_image_mismatch_relative_rms_threshold'],
                gross_refitted_source_mismatch=after>protocol['benchmarks']['gross_source_image_mismatch_relative_rms_threshold'],
                original_conditional_integral=float(start_surface.sum()*d*d*1e6),recovered_conditional_integral=float(recovered.sum()*d*d*1e6),
                finite_field_projection_flux_fraction=float(prediction.sum()/max(recovered.sum(),1e-30)),
                optimizer_converged=fit['converged'],iterations=fit['iterations'],projected_gradient_relative_rms=fit['projected_gradient_relative_rms'],
                source_image_pixels_evaluated=int(np.sum(evaluation>0)),seconds=time.monotonic()-start,
                noise_calibrated_likelihood=False,observed_height_inferred=False)
            rows.append(result)
            products.append(dict(path=str(path.relative_to(ROOT)),sha256=digest(path),component=component,height_kpc=height))
            write_json(output/(label+'-optimizer.json'),fit)
            for lo in range(0,int(cfg['reported_radius_kpc']),1):
                w=np.where((r>=lo)&(r<lo+1),evaluation,0)
                if np.sum(w)>0 and np.sum(w*target**2)>0:
                    annuli.append(dict(component=component,height_kpc=height,inner_kpc=lo,outer_kpc=lo+1,
                        unchanged_lift_relative_image_rms=weighted_relative_rms(old_projected,target,w),
                        refitted_source_relative_image_rms=weighted_relative_rms(prediction,target,w)))
            print('DONE '+label+' old='+str(round(before,5))+' repaired='+str(round(after,5))+' converged='+str(fit['converged']),flush=True)
    write_csv(output/'source-closure.csv',rows);write_csv(output/'source-closure-annuli.csv',annuli)
    geometry=read_json(ROOT/source_protocol['geometry_source'])
    geometry_availability=[dict(path=a['path'],present=(ROOT/a['path']).is_file(),recorded_sha256=a['raw_sha256']) for a in geometry['sources']]
    write_json(output/'summary.json',dict(status='SOURCE_IMAGE_DIAGNOSTIC_NOT_ADMITTED_FOR_MOTION',admission_disposition=protocol['admission_disposition'],
        previous_goal_turn_classification='progress',source_cases_executed=len(rows),independent_benchmark_tests=test_result.testsRun,
        all_optimizers_converged=all(r['optimizer_converged'] for r in rows),
        protocol_sha256=digest(protocol_path),protocol=protocol,source_audit_sha256=digest(audit_path),
        code_hashes={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),ROOT/'scripts/mond_atlas_source_projection.py',ROOT/'tests/test_mond_atlas_projection.py')},
        products=products,geometry_raw_source_availability=geometry_availability,
        response_files_opened=[],kinematic_response_scores_computed=0,admitted_galaxy_cube_predictions=0,goal_complete=False,
        scientific_limits=['A source fit can restore projected light while leaving multiple heights possible. It does not identify true depth.',
            'The initial inverse implementation used excluded pixels in its starting point. A held-pixel mutation test caught this before real-source application; the starting point was corrected and all six controls passed.',
            'RMS uses coverage weights, not measured source covariance; 5 percent is a construction diagnostic, not posterior acceptance.',
            'Reconstruction uses the full available source image. No independent held-source-image predictive score is claimed.',
            'Native pixel-center binning and instrumental blurring remain; these effective-resolution sources are not deconvolved intrinsic mass maps.',
            'Original raw S4G geometry tables referenced by the stored configuration are absent from this workspace and the checked original checkout. The stored derived record remains available, but fresh raw-record verification is incomplete.',
            'No new field force or motion prediction is admitted by this source-only experiment.']))
    print(dict(cases=len(rows),converged=sum(r['optimizer_converged'] for r in rows),goal_complete=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/mond_atlas_source_projection_v1.json')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--private',type=Path,required=True)
    args=parser.parse_args();run(args.protocol.resolve(),args.output.resolve(),args.private.resolve())
