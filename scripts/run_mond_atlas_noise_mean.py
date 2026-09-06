"""Test background-mean uncertainty on every previously declared partition."""
from __future__ import annotations
import argparse,copy,hashlib,io,unittest
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
from mond_atlas_noise_mean import evaluate_mean_branches
from run_mond_atlas_noise import masks
from run_mond_atlas_noise_robustness import reversed_masks


def run(config_path,output):
    config=read_json(config_path);parent=read_json(ROOT/config['parent_protocol'])
    partition=read_json(ROOT/config['partition_protocol']);prior_path=ROOT/config['partition_result'];prior=read_json(prior_path)
    if config['admission_disposition']!='SOURCE_BLOCKED':raise ValueError('background diagnostic only')
    for path,expected in prior['bindings'].items():
        if digest(ROOT/path)!=expected:raise ValueError('prior binding changed: '+path)
    if output.exists():raise FileExistsError('immutable outputs')
    output.mkdir(parents=True)
    paths=[config_path,ROOT/config['parent_protocol'],ROOT/config['partition_protocol'],prior_path,
        Path(__file__),ROOT/'scripts/mond_atlas_noise_mean.py',ROOT/'tests/test_mond_atlas_noise_mean.py',
        ROOT/'scripts/run_mond_atlas_noise.py',ROOT/'scripts/run_mond_atlas_noise_robustness.py',ROOT/'scripts/mond_atlas_cube.py']
    bindings={str(p.relative_to(ROOT)):digest(p) for p in paths}
    write_json(output/'prospective-bindings.json',dict(config=config,bindings=bindings,
        previous_partition_exposure='All 192 original background partitions are development-exposed.',new_galaxy_motion_scores=0))
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_mond_atlas_noise_mean.py')
    log=io.StringIO();tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (output/'validation.log').write_text(log.getvalue(),encoding='utf-8',newline='\n')
    if not tests.wasSuccessful():raise RuntimeError(log.getvalue())
    rows=[];galaxies=[];failures=[];source_bindings={}
    for audit in read_json(ROOT/parent['source_audit']):
        name=audit['name'];path=ROOT/parent['source_packets']/(name+'.npz')
        old_path=prior_path.parent/(name+'.json');old=read_json(old_path)
        if digest(path)!=old['summary']['packet_sha256']:raise ValueError('source cube changed: '+name)
        source_bindings[str(path.relative_to(ROOT))]=digest(path);source_bindings[str(old_path.relative_to(ROOT))]=digest(old_path)
        with np.load(path,allow_pickle=False) as packet:
            east,north=packet['east'],packet['north'];lo,hi=parent['sky_annulus_arcsec'];r=np.hypot(east,north)
            data=dict(cube=np.where(((r>lo)&(r<hi))[None,:,:],packet['cube'],0.),east=east,north=north)
        previous={p['split']:p for p in old['partitions']};details=[]
        for seed in partition['split_seeds']:
            cfg=copy.deepcopy(parent);cfg['split_seed']=seed
            for direction in partition['calibration_validation_directions']:
                label=str(seed)+'-'+direction
                try:
                    train,test=masks(east,north,cfg) if direction=='forward' else reversed_masks(east,north,cfg)
                    identity=hashlib.sha256(train.tobytes()+test.tobytes()).hexdigest()
                    if identity!=previous[label]['split_sha256']:raise ValueError('partition changed')
                    result,arrays=evaluate_mean_branches(data,cfg,train,test)
                    replay=result['branches']['previous_fixed_mean']
                    if abs(replay['mean_square']-previous[label]['mean_square'])>1e-12 or abs(replay['channel_lag1']-previous[label]['channel_lag1'])>1e-12:raise ValueError('stored background diagnostic replay failed')
                    detail=dict(galaxy=name,split=label,split_sha256=identity,duplicate_of=previous[label]['duplicate_of'],**result)
                    details.append(detail)
                    for branch,diagnostic in result['branches'].items():
                        rows.append(dict(galaxy=name,split=label,branch=branch,split_sha256=identity,
                            duplicate_of=previous[label]['duplicate_of'],diagnostic_pass=diagnostic['diagnostic_pass'],
                            mean_square=diagnostic['mean_square'],channel_lag1=diagnostic['channel_lag1'],
                            quadrant_min=min(q['mean_square'] for q in diagnostic['quadrants']),
                            quadrant_max=max(q['mean_square'] for q in diagnostic['quadrants']),
                            failed_gates=';'.join(k for k,v in diagnostic['gates'].items() if not v),**result['mean_accounting']))
                except (ValueError,np.linalg.LinAlgError,ArithmeticError) as exc:
                    failures.append(dict(galaxy=name,split=label,error=str(exc)))
        summaries=[]
        for branch in config['branches']:
            selected=[r for r in rows if r['galaxy']==name and r['branch']==branch and r['duplicate_of'] is None]
            summaries.append(dict(galaxy=name,branch=branch,partitions=len(selected),
                passing_partitions=sum(r['diagnostic_pass'] for r in selected),
                all_declared_partitions_pass=len(selected)==old['summary']['unique_partitions'] and all(r['diagnostic_pass'] for r in selected),
                mean_square_min=min((r['mean_square'] for r in selected),default=None),mean_square_max=max((r['mean_square'] for r in selected),default=None),
                channel_lag1_min=min((r['channel_lag1'] for r in selected),default=None),channel_lag1_max=max((r['channel_lag1'] for r in selected),default=None),
                mean_variance_factor_min=min((r['calibration_mean_variance_factor'] for r in selected),default=None),
                mean_variance_factor_max=max((r['calibration_mean_variance_factor'] for r in selected),default=None),
                failed_gates=';'.join(sorted({k for r in selected for k in r['failed_gates'].split(';') if k}))))
        galaxies.extend(summaries);write_json(output/(name+'.json'),dict(summaries=summaries,partitions=details))
        print(name,{r['branch']:str(r['passing_partitions'])+'/'+str(r['partitions']) for r in summaries},flush=True)
    write_csv(output/'partitions.csv',rows);write_csv(output/'galaxies.csv',galaxies)
    result=dict(status='BACKGROUND_MEAN_UNCERTAINTY_DIAGNOSTIC' if not failures else 'INCOMPLETE_BACKGROUND_MEAN_DIAGNOSTIC',
        admission_disposition='SOURCE_BLOCKED',config=config,bindings=bindings,source_bindings=source_bindings,
        galaxies=len({r['galaxy'] for r in galaxies}),partition_branch_evaluations=len(rows),execution_failures=failures,
        all_original_partition_replays_pass=not failures,
        split_stable_pass_by_branch={b:[r['galaxy'] for r in galaxies if r['branch']==b and r['all_declared_partitions_pass']] for b in config['branches']},
        independent_unit_tests=tests.testsRun,galaxy_motion_scores_computed=0,admitted_galaxy_cube_predictions=0,goal_complete=False)
    write_json(output/'summary.json',result);print({k:result[k] for k in ['galaxies','partition_branch_evaluations','execution_failures','split_stable_pass_by_branch']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/mond_atlas_noise_mean_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.config.resolve(),args.output.resolve())
