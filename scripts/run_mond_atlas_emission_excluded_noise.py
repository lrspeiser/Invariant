"""Apply one fixed emission exclusion to every previously declared noise split."""
from __future__ import annotations
import argparse, copy, hashlib, io, unittest
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT, read_json, write_json, write_csv, digest
from mond_atlas_image_io import read_primary_image
from mond_atlas_background_support import block_fraction, dilate_disk
from mond_atlas_emission_exclusion import exclude_support
from mond_atlas_noise_mean import evaluate_mean_branches
from run_mond_atlas_noise import masks
from run_mond_atlas_noise_robustness import reversed_masks


def run(config_path, output):
    config = read_json(config_path); support_config = read_json(ROOT/config['support_protocol'])
    source_path = ROOT/config['support_summary']; support_result = read_json(source_path)
    parent = read_json(ROOT/support_config['parent_noise_protocol'])
    partition = read_json(ROOT/support_config['partition_protocol'])
    old_root = (ROOT/support_config['partition_results']).parent
    if config['admission_disposition']!='SOURCE_BLOCKED' or config['new_gravity_response_scoring_permitted']:
        raise ValueError('background-only prospective scope required')
    for path, expected in support_result['bindings'].items():
        if digest(ROOT/path)!=expected: raise ValueError('support audit binding changed: '+path)
    if output.exists(): raise FileExistsError('immutable outputs')
    output.mkdir(parents=True)
    paths = [config_path, ROOT/config['support_protocol'],source_path,ROOT/config['mean_protocol'],
        ROOT/support_config['parent_noise_protocol'],ROOT/support_config['partition_protocol'],Path(__file__),
        ROOT/'scripts/mond_atlas_emission_exclusion.py',ROOT/'tests/test_mond_atlas_emission_exclusion.py',
        ROOT/'scripts/mond_atlas_noise_mean.py',ROOT/'scripts/run_mond_atlas_noise.py',
        ROOT/'scripts/run_mond_atlas_noise_robustness.py',ROOT/'scripts/mond_atlas_cube.py',
        ROOT/'scripts/mond_atlas_background_support.py',ROOT/'scripts/mond_atlas_image_io.py']
    bindings = {str(p.relative_to(ROOT)):digest(p) for p in paths}
    write_json(output/'prospective-bindings.json',dict(config=config,bindings=bindings,
        prior_exposure='Existing galaxies, support maps and original noise diagnostics were inspected in development. The new excluded-mask diagnostics were not.'))
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_mond_atlas_emission_exclusion.py')
    log = io.StringIO(); tests = unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (output/'validation.log').write_text(log.getvalue(),encoding='utf-8',newline='\n')
    if not tests.wasSuccessful():raise RuntimeError(log.getvalue())
    moments = {r['name']:r for r in read_json(ROOT/support_config['native_moment_receipt'])['files'] if r['resolution']=='NA' and r['moment']==0}
    rows=[]; galaxies=[]; sources={}; failures=[]
    for audit in read_json(ROOT/support_config['cube_audit']):
        name=audit['name']; moment=moments[name]
        packet_path=ROOT/parent['source_packets']/(name+'.npz'); old_path=old_root/(name+'.json'); old=read_json(old_path)
        old_support_path=source_path.parent/(name+'.json'); old_support=read_json(old_support_path)
        for path in [ROOT/moment['file'],packet_path,old_path]:
            relative=str(path.relative_to(ROOT)); actual=digest(path)
            if actual!=support_result['source_bindings'][relative]:raise ValueError('support source changed: '+relative)
            sources[relative]=actual
        sources[str(old_support_path.relative_to(ROOT))]=digest(old_support_path)
        image,header=read_primary_image(ROOT/moment['file'])
        fraction=block_fraction(np.isfinite(image)&(image>0),support_config['block_factor'])
        radius=int(np.ceil(4*audit['extra_smoothing_sigma_arcsec']/(abs(header['CDELT1'])*3600*support_config['block_factor'])))+2
        if radius!=old_support['summary']['warning_dilation_radius_cells']:raise ValueError('declared support radius changed')
        expanded=dilate_disk(fraction>0,radius)
        with np.load(packet_path,allow_pickle=False) as packet:
            east,north=packet['east'],packet['north'];lo,hi=parent['sky_annulus_arcsec'];r=np.hypot(east,north)
            allowed=(r>lo)&(r<hi)&~expanded
            data=dict(cube=np.where(allowed[None,:,:],packet['cube'],0.),east=east,north=north)
        previous={p['split']:p for p in old['partitions']};seen={};details=[]
        for seed in partition['split_seeds']:
            cfg=copy.deepcopy(parent);cfg['split_seed']=seed
            for direction in partition['calibration_validation_directions']:
                train,test=masks(east,north,cfg) if direction=='forward' else reversed_masks(east,north,cfg)
                label=str(seed)+'-'+direction;identity=hashlib.sha256(train.tobytes()+test.tobytes()).hexdigest()
                if identity!=previous[label]['split_sha256']:raise ValueError('original split changed')
                train,test,coverage=exclude_support(train,test,expanded,east,north,cfg)
                changed_identity=hashlib.sha256(train.tobytes()+test.tobytes()).hexdigest()
                duplicate=seen.get(changed_identity);seen.setdefault(changed_identity,label)
                row=dict(galaxy=name,split=label,original_split_sha256=identity,excluded_split_sha256=changed_identity,
                    duplicate_of=duplicate,**coverage,diagnostic_pass=False,status='INSUFFICIENT_BACKGROUND_SUPPORT')
                detail=dict(**row)
                if coverage['sufficient_support']:
                    try:
                        result,arrays=evaluate_mean_branches(data,cfg,train,test)
                        diagnostic=result['branches'][config['covariance_branch']]
                        row.update(status='BACKGROUND_DIAGNOSTIC_EVALUATED',diagnostic_pass=diagnostic['diagnostic_pass'],
                            mean_square=diagnostic['mean_square'],channel_lag1=diagnostic['channel_lag1'],
                            quadrant_min=min(q['mean_square'] for q in diagnostic['quadrants']),
                            quadrant_max=max(q['mean_square'] for q in diagnostic['quadrants']),
                            failed_gates=';'.join(k for k,v in diagnostic['gates'].items() if not v))
                        detail=dict(**row,result=result)
                    except (ValueError,ArithmeticError,np.linalg.LinAlgError) as exc:
                        row.update(status='COVARIANCE_ESTIMATION_FAILED',error=str(exc));detail=dict(**row)
                        failures.append(dict(galaxy=name,split=label,error=str(exc)))
                rows.append(row);details.append(detail)
        selected=[r for r in rows if r['galaxy']==name]
        evaluated=[r for r in selected if r['status']=='BACKGROUND_DIAGNOSTIC_EVALUATED']
        summary=dict(galaxy=name,declared_partitions=len(selected),unique_excluded_partitions=len(seen),
            sufficient_support_partitions=sum(r['sufficient_support'] for r in selected),evaluated_partitions=len(evaluated),
            passing_partitions=sum(r['diagnostic_pass'] for r in evaluated),
            all_declared_splits_sufficient_support=all(r['sufficient_support'] for r in selected),
            all_declared_splits_pass=all(r['diagnostic_pass'] for r in selected),
            calibration_pixels_min=min(r['calibration_pixels'] for r in selected),validation_pixels_min=min(r['validation_pixels'] for r in selected),
            mean_square_min=min((r['mean_square'] for r in evaluated),default=None),mean_square_max=max((r['mean_square'] for r in evaluated),default=None),
            channel_lag1_min=min((r['channel_lag1'] for r in evaluated),default=None),channel_lag1_max=max((r['channel_lag1'] for r in evaluated),default=None),
            background_selection_independence_established=False,galaxy_motion_likelihood_admitted=False)
        galaxies.append(summary);write_json(output/(name+'.json'),dict(summary=summary,partitions=details))
        print(name,dict(sufficient=summary['sufficient_support_partitions'],evaluated=summary['evaluated_partitions'],passed=summary['passing_partitions'],declared=summary['declared_partitions']),flush=True)
    write_csv(output/'galaxies.csv',galaxies);write_csv(output/'partitions.csv',rows)
    result=dict(status='FIXED_EMISSION_EXCLUSION_CONTROL_EXECUTED',admission_disposition='SOURCE_BLOCKED',config=config,
        bindings=bindings,source_bindings=sources,galaxies=len(galaxies),declared_partitions=len(rows),
        sufficient_support_partitions=sum(r['sufficient_support'] for r in rows),
        evaluated_partitions=sum(r['status']=='BACKGROUND_DIAGNOSTIC_EVALUATED' for r in rows),
        passing_partitions=sum(r['diagnostic_pass'] for r in rows),covariance_estimation_failures=failures,
        all_declared_splits_pass=[r['galaxy'] for r in galaxies if r['all_declared_splits_pass']],
        all_declared_splits_sufficient_support=[r['galaxy'] for r in galaxies if r['all_declared_splits_sufficient_support']],
        independent_unit_tests=tests.testsRun,gravity_motion_scores_computed=0,admitted_galaxy_cube_predictions=0,goal_complete=False)
    write_json(output/'summary.json',result);print({k:result[k] for k in ['galaxies','declared_partitions','sufficient_support_partitions','evaluated_partitions','passing_partitions','all_declared_splits_pass']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/mond_atlas_emission_excluded_noise_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.config.resolve(),args.output.resolve())
