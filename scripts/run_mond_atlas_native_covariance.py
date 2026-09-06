"""Execute the predeclared NGC2976 background transfer study on one CPU thread."""
from __future__ import annotations

import os
for _thread_variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable] = '1'

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import sys
import traceback
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import astropy
from astropy.io import fits
from threadpoolctl import threadpool_limits, threadpool_info
from mond_atlas_native_covariance import (
    block_geometry, extract_background, sky_design, fit_and_select_training, summarize_model)

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, default=jsonable, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows):
    if not rows:
        raise ValueError('empty declared report')
    with Path(path).open('x', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def make_plot(output, private, geometry, ranking, results):
    os.environ['MPLCONFIGDIR'] = str(private/'matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    order = [r['model_id'] for r in sorted(ranking, key=lambda r:r['declaration_order'])]
    selected = ranking[0]['model_id']
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5), layout='constrained')
    ax = axes[0]
    for row in geometry:
        color = '#a16a2a' if row['region']=='training' else '#007e91'
        ax.add_patch(Rectangle((row['x0'],row['y0']),24,24,color=color,alpha=.85))
    ax.set(xlim=(0,1024),ylim=(0,1024),aspect='equal',xlabel='Native x pixel (east points left)',ylabel='Native y pixel',title='Fixed guarded background cores')
    ax.text(.03,.98,'East validation: teal\nWest training: ochre',transform=ax.transAxes,va='top',fontsize=9)
    ax = axes[1]
    ax.axvspan(.8,1.2,color='#b9dbc2',alpha=.4)
    labels=[]
    for i,key in enumerate(order):
        row=results[key]['validation']; color='#007e91' if key==selected else '#747e89'
        ax.plot([row['block_q_min'],row['block_q_max']],[i,i],color=color,alpha=.5,lw=2)
        ax.scatter([row['mean_q_over_n']],[i],color=color,s=45,zorder=3)
        labels.append(key.replace('channel_constant','constant').replace('channel_affine_sky','affine').replace('__',' / ').replace('_shrink_',' a=').replace('010','.10').replace('050','.50'))
    ax.set(yticks=range(len(order)),yticklabels=labels,xlabel='Held-east q / channel count',title='Dots: block means; lines: block ranges')
    ax.tick_params(axis='y',labelsize=8);ax.axvline(1,color='#324659',lw=.8,ls='--')
    ax = axes[2]
    for region,color in [('training','#a16a2a'),('validation','#007e91')]:
        groups=results[selected][region]['spatial']
        for centered,style in [(False,'-'),(True,'--')]:
            entries=[r for r in groups if r['axis']=='x' and r['local_channel_means_removed']==centered]
            entries.sort(key=lambda r:r['lag_native_pixels'])
            ax.plot([r['lag_native_pixels']*1.5 for r in entries],[r['mean_product'] for r in entries],style,marker='o',ms=3,color=color,label=region+(' centered' if centered else ' raw'))
    ax.set(xlabel='Horizontal separation (arcsec)',ylabel='Whitened residual product',title='Selected model: measured spatial dependence',xscale='log')
    ax.axhline(0,color='#324659',lw=.8);ax.legend(fontsize=8)
    fig.suptitle('NGC2976 external background — conditional channel models, no cube-likelihood admission',fontsize=14)
    fig.savefig(output/'background-transfer.png',dpi=160)
    plt.close(fig)


def run(config_path, output, private):
    if output.exists() or private.exists():
        raise FileExistsError('immutable new public and private directories required')
    output.mkdir(parents=True);private.mkdir(parents=True)
    access=[]
    try:
        config=json.loads(config_path.read_text())
        freeze_path=ROOT/'work/gravity-first-principles/mond-atlas-native-covariance-001/protocol-freeze.json'
        freeze=json.loads(freeze_path.read_text())
        if digest(config_path)!=freeze['config_sha256']:
            raise ValueError('preimplementation protocol changed')
        if config['admission_disposition']!='SOURCE_BLOCKED' or config['gravity_motion_scores_permitted']:
            raise ValueError('observational response scoring prohibited')
        for key in ('cube','support'):
            if digest(ROOT/config[key+'_path'])!=config[key+'_sha256']:
                raise ValueError('hash-bound '+key+' changed')
        bindings=[config_path, freeze_path, Path(__file__),ROOT/'scripts/mond_atlas_native_covariance.py',
            ROOT/'tests/test_mond_atlas_native_covariance.py',ROOT/config['cube_path'],ROOT/config['support_path'],
            ROOT/config['prior_protocol'],ROOT/config['prior_summary'],
            ROOT/'docs/OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md',
            ROOT/'work/gravity-first-principles/mond-atlas-native-selection-001/source-evidence.json',
            ROOT/'work/gravity-first-principles/mond-atlas-execution-017/README.md']
        bound={p.relative_to(ROOT).as_posix():digest(p) for p in bindings}
        write_json(output/'prospective-bindings.json',dict(timestamp_utc=now(),bindings=bound,
            prior_protocol_freeze_sha256=digest(freeze_path),new_background_values_read=False,
            gravity_motion_response_access=False,previous_development_exposure=True))
        with np.load(ROOT/config['support_path']) as packet:
            supports={'training':packet['calibration'].astype(bool),'validation':packet['validation'].astype(bool)}
        geometry=block_geometry(supports,config['regions'])
        training_rows=[r for r in geometry if r['region']=='training']
        validation_rows=[r for r in geometry if r['region']=='validation']
        for region,rows in [('training',training_rows),('validation',validation_rows)]:
            if len(rows)<config['regions']['minimum_'+region+'_blocks']:
                raise ValueError('insufficient '+region+' blocks: '+str(len(rows)))
        write_csv(output/'block-geometry.csv',geometry)
        np.savez_compressed(private/'geometry-supports.npz',**supports)
        predicted_bytes=len(geometry)*config['regions']['block_core_side_native_pixels']**2*42*8
        if predicted_bytes>config['resource_limits']['maximum_extracted_background_bytes']:
            raise ValueError('extraction memory bound exceeded')
        write_json(output/'geometry-freeze.json',dict(timestamp_utc=now(),training_blocks=len(training_rows),validation_blocks=len(validation_rows),
            block_geometry_sha256=digest(output/'block-geometry.csv'),prior_support_sha256=config['support_sha256'],
            native_core_side_pixels=24,native_lattice_step_pixels=48,edge_gap_pixels=24,
            native_vectors_per_block=576,expected_extracted_bytes=predicted_bytes,
            new_background_values_read=False,training_fold_counts={str(i):sum(r['fold']==i for r in training_rows) for i in range(3)}))
        print(f'Frozen geometry: {len(training_rows)} west / {len(validation_rows)} east blocks. Running independent controls.',flush=True)
        suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_mond_atlas_native_covariance.py')
        log=io.StringIO();test=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        (output/'unit-tests.log').write_text(log.getvalue(),encoding='utf8')
        spec=importlib.util.spec_from_file_location('native_covariance_benchmark',ROOT/'tests/test_mond_atlas_native_covariance.py')
        benchmark_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(benchmark_module)
        benchmark=benchmark_module.manufactured_report(config['benchmarks'])
        benchmark.update(unit_tests_run=test.testsRun,unit_tests_passed=test.wasSuccessful(),timestamp_utc=now(),new_background_values_read=False,
            note='The numerical receipt repeats the same deterministic manufactured draw used by the unit test; not an extra independent realization.')
        write_json(output/'benchmarks.json',benchmark)
        if not test.wasSuccessful() or not benchmark['passed']:
            raise ArithmeticError('independent benchmark failed')
        header=fits.getheader(ROOT/config['cube_path'])
        if [header[k] for k in ['NAXIS1','NAXIS2','NAXIS3','BUNIT','CTYPE1','CTYPE2','CTYPE3']] != [1024,1024,42,'JY/BEAM','RA---SIN','DEC--SIN','VELO-HEL']:
            raise ValueError('unexpected fixed native geometry or units')
        if any(header.get('CROTA'+str(i),0)!=0 for i in (1,2,3)):
            raise ValueError('unexpected rotated WCS')
        training_design=sky_design(training_rows,header);validation_design=sky_design(validation_rows,header)
        with fits.open(ROOT/config['cube_path'],memmap=True) as hdus:
            training=extract_background(hdus[0].data[0],training_rows)
        access.append(dict(event='western_training_cores_extracted',timestamp_utc=now(),blocks=len(training),vectors=int(np.prod(training.shape[:-1])),scalar_values=training.size,eastern_values_read=False))
        print('Controls passed. Fitting and selecting on western blocks only.',flush=True)
        models,ranking,cv=fit_and_select_training(training,training_design,training_rows,config)
        write_csv(output/'training-cv-blocks.csv',cv)
        write_json(output/'fitted-models.json',models)
        write_json(output/'training-selection.json',dict(timestamp_utc=now(),ranking=ranking,selected_model_id=ranking[0]['model_id'],
            eastern_background_values_read=False,criterion='equal-western-block mean marginal log density per channel on geometry-defined inner folds'))
        selection_binding={p.name:digest(p) for p in [output/'fitted-models.json',output/'training-cv-blocks.csv',output/'training-selection.json']}
        write_json(output/'selection-frozen-before-east.json',dict(timestamp_utc=now(),files=selection_binding,eastern_values_read=False))
        access.append(dict(event='training_selection_and_all_models_frozen',timestamp_utc=now(),selected_model=ranking[0]['model_id'],eastern_values_read=False))
        print('Selected on west: '+ranking[0]['model_id']+'. Opening held-east cores.',flush=True)
        with fits.open(ROOT/config['cube_path'],memmap=True) as hdus:
            validation=extract_background(hdus[0].data[0],validation_rows)
        access.append(dict(event='eastern_validation_cores_extracted',timestamp_utc=now(),blocks=len(validation),vectors=int(np.prod(validation.shape[:-1])),scalar_values=validation.size,eastern_values_read=True))
        np.savez_compressed(private/'background-blocks.npz',training=training,validation=validation,
            training_design=training_design,validation_design=validation_design)
        reports={};block_reports=[];channel_reports=[];spatial_reports=[]
        for key,model in models.items():
            reports[key]={}
            for region,data,design,rows in [('training',training,training_design,training_rows),('validation',validation,validation_design,validation_rows)]:
                report,blocks,channels,spatial=summarize_model(data,design,rows,model,config)
                reports[key][region]=report
                block_reports.extend(dict(model_id=key,region=region,**b) for b in blocks)
                channel_reports.extend(dict(model_id=key,region=region,**r) for r in channels)
                spatial_reports.extend(dict(model_id=key,region=region,**r) for r in spatial)
            print(key+': east q/N='+format(reports[key]['validation']['mean_q_over_n'],'.4f')+', mean check='+str(reports[key]['validation']['descriptive_checks']['mean_residual_small']),flush=True)
        write_json(output/'model-results.json',reports)
        write_csv(output/'block-diagnostics.csv',block_reports)
        write_csv(output/'channel-diagnostics.csv',channel_reports)
        write_csv(output/'spatial-pair-diagnostics.csv',spatial_reports)
        make_plot(output,private,geometry,ranking,reports)
        if any(digest(ROOT/p)!=value for p,value in bound.items()):
            raise ValueError('bound source or implementation changed during execution')
        if any(digest(output/p)!=value for p,value in selection_binding.items()):
            raise ValueError('training-only ranking or models changed after eastern access')
        pools=threadpool_info()
        if any(p.get('num_threads',1)!=1 for p in pools):
            raise ValueError('numerical thread limit violated')
        chosen=ranking[0]['model_id']
        summary=dict(status='EXECUTED_CONDITIONAL_BACKGROUND_COVARIANCE_TRANSFER',admission_disposition='SOURCE_BLOCKED',
            galaxy='NGC2976',training_blocks=len(training_rows),validation_blocks=len(validation_rows),vectors_per_block=576,
            channels=42,candidates=len(models),training_inner_folds=3,selected_model_id=chosen,
            selected_east_mean_q_over_n=reports[chosen]['validation']['mean_q_over_n'],
            selected_east_all_descriptive_checks_pass=reports[chosen]['validation']['all_descriptive_checks_pass'],
            selected_east_descriptive_checks=reports[chosen]['validation']['descriptive_checks'],
            alternatives_passing_all_descriptive_checks=[k for k in models if reports[k]['validation']['all_descriptive_checks_pass']],
            unit_tests_run=test.testsRun,independent_benchmarks_passed=True,prospective_bindings_verified=len(bound),
            training_selection_unchanged_after_east_access=True,background_extraction_bytes=training.nbytes+validation.nbytes,
            new_download_bytes=0,numerical_cpu_threads=1,gpu_used=False,observed_gravity_motion_scores=0,admitted_observed_cube_likelihoods=0,
            runtime=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,astropy=astropy.__version__,threadpools=pools),
            limitations=['Development-exposed NGC2976/background; new folds are not pristine confirmation.',
                'MOM0 screening reuses the observation; pure noise, no foreground emission and independent source selection are unestablished.',
                'Channel covariance is fitted to background residuals after explicit mean models; mean errors or foreground structure may remain.',
                'Spectral Gaussian scores are conditional on estimated parameters and are averaged by block; they are not a joint spatial cube likelihood.',
                'Guarded disjoint blocks are descriptive units; spatial independence and effective replication count are not certified.',
                'Conditional even-channel forecasts reuse odd channels at the same pixel, not a fresh-noise realization.',
                'Source/beam/selection/covariance uncertainty, exact spectral response and foreground contamination still block observed cube admission.'])
        write_json(output/'access-log.json',access)
        write_json(output/'summary.json',summary)
        files={p.relative_to(ROOT).as_posix():digest(p) for base in (output,private) for p in sorted(base.rglob('*')) if p.is_file()}
        write_json(output/'run-manifest.json',dict(files=files,manifest_self_excluded=True))
        print(json.dumps({k:v for k,v in summary.items() if k not in ('runtime','limitations')},indent=2),flush=True)
    except Exception as error:
        failure=output/'failure-receipt.json'
        if not failure.exists():
            write_json(failure,dict(status='FAILED_RETAINED',timestamp_utc=now(),error_type=type(error).__name__,error=str(error),
                traceback=traceback.format_exc(),access_log=access,observed_cube_admission=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'configs/mond_atlas_native_covariance_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--private-output',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.config.resolve(),args.output.resolve(),args.private_output.resolve())
