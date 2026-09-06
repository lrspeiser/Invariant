"""Execute nested whole-galaxy residual learning on the real SPARC development set."""
from __future__ import annotations
import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT, digest, read_json, write_json, write_csv
from mond_atlas_pattern_learning import galaxy_folds, nested_predictions, synthetic_controls


def run(args):
    config_path=args.config.resolve(); config=read_json(config_path)
    output=args.output.resolve()
    if output.exists(): raise FileExistsError('Use a new immutable output directory')
    output.mkdir(parents=True)
    inputs=[config_path,ROOT/config['input'],Path(__file__),ROOT/'scripts/mond_atlas_pattern_learning.py',
        ROOT/'tests/test_mond_atlas_pattern_learning.py']
    bindings={p.relative_to(ROOT).as_posix():digest(p) for p in inputs}
    write_json(output/'prospective-bindings.json',dict(config=config,bindings=bindings,
        development_data_previously_exposed=True,new_full_field_response_access=False))
    xp=np; runtime=dict(python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,backend=args.backend)
    if args.backend=='cuda':
        import cupy as cp
        cp.get_default_memory_pool().set_limit(size=config['cuda_memory_pool_limit_bytes'])
        runtime.update(cupy=cp.__version__,cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
            cuda_driver=cp.cuda.runtime.driverGetVersion(),device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
            device_memory_before=dict(zip(['free','total'],map(int,cp.cuda.runtime.memGetInfo()))),
            pool_limit_bytes=config['cuda_memory_pool_limit_bytes'])
        xp=cp
    controls=synthetic_controls(xp)
    write_json(output/'controls.json',controls)
    gates=config['benchmarks']
    assert controls['cpu_backend_max_abs']<gates['cpu_gpu_prediction_absolute_tolerance']
    assert controls['sklearn_max_abs']<gates['independent_sklearn_prediction_absolute_tolerance']
    assert controls['positive_control_rmse_ratio']<gates['positive_control_rmse_ratio_max']
    # Controls execute before this run opens the observational CSV.
    with (ROOT/config['input']).open(newline='',encoding='utf-8-sig') as stream:
        rows=list(csv.DictReader(stream))
    features=config['baseline_features']+config['stellar_features']+config['gas_features']
    accepted=[]; excluded=[]
    for row in rows:
        if row['selected'].lower()!='true':
            excluded.append(dict(galaxy=row['galaxy'],reason='existing_radial_selection')); continue
        try:
            values={key:float(row[key]) for key in config['baseline_features']+['hubble_type','gas_mass_fraction_proxy']}
            for source,target in [('effective_stellar_surface_brightness','log10_effective_stellar_surface_brightness'),
                                  ('disk_scale_length','log10_disk_scale_length')]:
                value=float(row[source]); values[target]=np.log10(value) if value>0 else np.nan
            y=float(row[config['target']]); x=[values[key] for key in features]
            if not np.isfinite([y]+x).all(): raise ValueError('nonfinite')
            accepted.append(dict(galaxy=row['galaxy'],target=y,**values))
        except (ValueError,TypeError):
            excluded.append(dict(galaxy=row['galaxy'],reason='missing_or_nonpositive_declared_input'))
    write_csv(output/'sample.csv',accepted); write_csv(output/'excluded.csv',excluded)
    names=[r['galaxy'] for r in accepted]
    x=np.array([[r[k] for k in features] for r in accepted]); y=np.array([r['target'] for r in accepted])
    columns={'baseline':[0,1,2,3],'stellar':[0,1,2,3,4,5,6],
             'gas':[0,1,2,3,7],'combined':list(range(8))}
    all_predictions=[]; fold_rows=[]; selection_rows=[]; cache={}; start=time.perf_counter()
    for seed in config['fold_seeds']:
        fold=galaxy_folds(names,seed,config['outer_fold_count'])
        fold_rows.extend(dict(galaxy=n,seed=seed,fold=int(fold[i])) for i,n in enumerate(names))
        for estimator in config['estimators']:
            for bundle in config['feature_bundles']:
                prediction,selected=nested_predictions(x[:,columns[bundle]],y,fold,estimator,config,xp)
                cache[(seed,estimator,bundle)]=prediction
                selection_rows.extend(dict(seed=seed,estimator=estimator,bundle=bundle,**r) for r in selected)
                all_predictions.extend(dict(galaxy=n,seed=seed,fold=int(fold[i]),estimator=estimator,bundle=bundle,
                    target=float(y[i]),prediction=float(prediction[i]),squared_error=float((prediction[i]-y[i])**2)) for i,n in enumerate(names))
                print(f'{seed} {estimator} {bundle}: {np.sqrt(np.mean((prediction-y)**2)):.6f} dex',flush=True)
    write_csv(output/'folds.csv',fold_rows); write_csv(output/'hyperparameters.csv',selection_rows)
    write_csv(output/'predictions.csv',all_predictions)
    rng=np.random.default_rng(config['null_seed']); metrics=[]
    for estimator in config['estimators']:
        baseline=np.array([cache[(s,estimator,'baseline')] for s in config['fold_seeds']])
        base_error=np.mean((baseline-y)**2,axis=0)
        for bundle in config['feature_bundles']:
            predicted=np.array([cache[(s,estimator,bundle)] for s in config['fold_seeds']])
            error=np.mean((predicted-y)**2,axis=0); delta=base_error-error
            bootstrap=rng.choice(delta,(config['bootstrap_replicates'],len(y))).mean(axis=1)
            seed_gains=[float(100*(np.mean((baseline[i]-y)**2)-np.mean((predicted[i]-y)**2))/np.mean((baseline[i]-y)**2)) for i in range(len(predicted))]
            metrics.append(dict(estimator=estimator,bundle=bundle,galaxies=len(y),rmse_dex=float(np.sqrt(error.mean())),
                raw_fixed_mond_rmse_dex=float(np.sqrt(np.mean(y*y))),mse_gain_percent_over_same_estimator_baseline=float(100*delta.mean()/base_error.mean()),
                mse_gain_dex2=float(delta.mean()),conditional_paired_bootstrap95_low=float(np.quantile(bootstrap,.025)),
                conditional_paired_bootstrap95_high=float(np.quantile(bootstrap,.975)),
                minimum_seed_gain_percent=min(seed_gains),maximum_seed_gain_percent=max(seed_gains)))
    write_csv(output/'metrics.csv',metrics)
    seed=config['fold_seeds'][0]; fold=galaxy_folds(names,seed,config['outer_fold_count'])
    baseline=cache[(seed,'rbf_kernel_ridge','baseline')]; observed=cache[(seed,'rbf_kernel_ridge','combined')]
    base_mse=float(np.mean((baseline-y)**2)); observed_gain=base_mse-float(np.mean((observed-y)**2))
    # Rank bins condition loosely on acceleration; not a full conditional randomization test.
    bins=np.empty(len(y),dtype=int); bins[np.argsort(x[:,0],kind='stable')]=np.arange(len(y))*4//len(y)
    null=[]
    for iteration in range(config['null_replicates']):
        shuffled=x.copy()
        for group in np.unique(bins):
            indices=np.flatnonzero(bins==group); shuffled[indices,4:]=x[rng.permutation(indices),4:]
        prediction,_=nested_predictions(shuffled,y,fold,'rbf_kernel_ridge',config,xp)
        gain=base_mse-float(np.mean((prediction-y)**2))
        null.append(dict(replicate=iteration,mse_gain_dex2=gain))
        print(f'structure shuffle {iteration+1}/{config["null_replicates"]}',flush=True)
    write_csv(output/'structure-shuffles.csv',null)
    if args.backend=='cuda':
        runtime['pool_used_bytes']=int(xp.get_default_memory_pool().used_bytes())
        runtime['pool_retained_bytes']=int(xp.get_default_memory_pool().total_bytes())
    runtime['fit_wall_seconds']=time.perf_counter()-start
    write_json(output/'runtime.json',runtime)
    summary=dict(status='EXPLORATORY_REAL_GALAXY_GPU_LEARNING_EXECUTED' if args.backend=='cuda' else 'EXPLORATORY_REAL_GALAXY_CPU_LEARNING_EXECUTED',
        galaxy_count=len(y),excluded_count=len(excluded),fold_seeds=config['fold_seeds'],outer_folds=config['outer_fold_count'],
        estimator_bundle_comparisons=len(metrics),oof_prediction_rows=len(all_predictions),metrics=metrics,controls=controls,
        first_seed_structure_gain_dex2=observed_gain,structure_shuffle_count=len(null),
        structure_shuffle_reference_fraction=(1+sum(r['mse_gain_dex2']>=observed_gain for r in null))/(len(null)+1),
        shuffle_fraction_is_calibrated_discovery_p_value=False,limitations=config['limitations'],
        genuine_unexposed_confirmation=False,new_full_field_predictions=0,goal_complete=False,bindings=bindings)
    write_json(output/'summary.json',summary)
    print(json.dumps(dict(galaxies=len(y),comparisons=len(metrics),wall_seconds=runtime['fit_wall_seconds'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/mond_atlas_pattern_learning_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=['cpu','cuda'],default='cuda')
    run(parser.parse_args())
