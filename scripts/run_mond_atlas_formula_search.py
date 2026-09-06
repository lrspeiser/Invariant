"""Run the frozen sparse-expression experiment on the existing development sample."""
import argparse
import csv
import subprocess
import sys
import time
import numpy as np
from pathlib import Path
from mond_atlas_common import ROOT, digest, read_json, write_json, write_csv
from mond_atlas_pattern_learning import galaxy_folds
from mond_atlas_formula_search import controls, nested, replay


def run(output, backend):
    config_path = ROOT/'configs/mond_atlas_formula_search_v1.json'; config = read_json(config_path)
    source = ROOT/'work/gravity-first-principles/mond-atlas-pattern-learning-001/sample.csv'
    output.mkdir(parents=True, exist_ok=False)
    bound = [config_path, source, Path(__file__), ROOT/'scripts/mond_atlas_formula_search.py',
             ROOT/'scripts/mond_atlas_pattern_learning.py', ROOT/'scripts/mond_atlas_common.py',
             ROOT/'tests/test_mond_atlas_formula_search.py', output.parent/'PREFLIGHT.md']
    write_json(output/'bindings.json', dict(bindings={p.relative_to(ROOT).as_posix():digest(p) for p in bound},
        config=config, development_exposed=True, new_field_access=False))
    test = subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_mond_atlas_formula_search.py'],
                          cwd=ROOT, capture_output=True, text=True)
    (output/'tests.log').write_text(test.stdout+test.stderr)
    if test.returncode: raise RuntimeError('Pre-access tests failed')
    xp = np; runtime = dict(backend=backend, python=sys.version)
    if backend == 'cuda':
        import cupy as cp
        cp.get_default_memory_pool().set_limit(size=1024**3); xp=cp
        runtime.update(device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), cupy=cp.__version__)
    c = controls(xp); write_json(output/'controls.json',c)
    assert c['cpu_gpu_max_abs'] < config['controls']['cpu_gpu_atol']
    assert c['independent_ridge_max_abs'] < config['controls']['independent_ridge_atol']
    assert c['planted_rmse_ratio'] < config['controls']['planted_rmse_ratio_max']
    assert c['replay_max_abs'] < 1e-8
    with source.open(newline='',encoding='utf-8-sig') as stream: rows=list(csv.DictReader(stream))
    names=[r['galaxy'] for r in rows]
    if len(names)!=len(set(names)): raise ValueError('Duplicate galaxy identifiers')
    x=np.array([[float(r[k]) for k in config['features']] for r in rows]); y=np.array([float(r['target']) for r in rows])
    assert np.isfinite(x).all() and np.isfinite(y).all()
    predictions=[]; selections=[]; metrics=[]; differences=[]; max_replay=0.; start=time.perf_counter()
    for seed in config['fold_seeds']:
        folds=galaxy_folds(names,seed,config['fold_count'])
        pred, records=nested(x,y,folds,config,xp)
        for r in records:
            for model in ('adaptive','baseline'):
                check=replay(x[folds==r['fold']],r['formulas'][model])
                max_replay=max(max_replay,float(np.max(abs(check-pred[model][folds==r['fold']]))))
            selections.append(dict(seed=seed,**r))
        base=(pred['baseline']-y)**2; err=(pred['adaptive']-y)**2; differences.append(base-err)
        metrics.append(dict(seed=seed,baseline_rmse_dex=float(np.sqrt(base.mean())),adaptive_rmse_dex=float(np.sqrt(err.mean())),
                            mse_gain_percent=float(100*(base.mean()-err.mean())/base.mean())))
        predictions.extend(dict(galaxy=n,seed=seed,fold=int(folds[i]),target=float(y[i]),
                                baseline=float(pred['baseline'][i]),adaptive=float(pred['adaptive'][i])) for i,n in enumerate(names))
        print(metrics[-1],flush=True)
    write_csv(output/'predictions.csv',predictions); write_json(output/'selections.json',selections)
    write_csv(output/'metrics.csv',metrics)
    rng=np.random.default_rng(config['null_seed']); null=[]
    bins=np.empty(len(y),dtype=int); bins[np.argsort(x[:,0],kind='stable')]=np.arange(len(y))*4//len(y)
    folds=galaxy_folds(names,config['fold_seeds'][0],config['fold_count'])
    for i in range(config['null_replicates']):
        shuffled=x.copy()
        for group in np.unique(bins):
            ix=np.flatnonzero(bins==group); shuffled[ix,4:]=x[rng.permutation(ix),4:]
        pred, records=nested(shuffled,y,folds,config,xp)
        b=float(np.mean((pred['baseline']-y)**2)); e=float(np.mean((pred['adaptive']-y)**2))
        null.append(dict(replicate=i,mse_gain_percent=100*(b-e)/b))
        write_json(output/f'shuffle-{i:02d}.json',dict(selection=records,predictions={k:v.tolist() for k,v in pred.items()},
                    shuffled_structure=shuffled[:,4:].tolist(),metric=null[-1]))
        print('shuffle',i+1,null[-1]['mse_gain_percent'],flush=True)
    write_csv(output/'shuffles.csv',null)
    delta=np.mean(differences,axis=0); boot=rng.choice(delta,(config['bootstrap_replicates'],len(y))).mean(axis=1)
    runtime['wall_seconds']=time.perf_counter()-start
    if backend=='cuda': runtime['pool_retained_bytes']=int(xp.get_default_memory_pool().total_bytes())
    write_json(output/'runtime.json',runtime)
    assert max_replay<1e-8
    summary=dict(status='EXPLORATORY_SPARSE_SEARCH_EXECUTED',galaxies=len(y),metrics=metrics,
        mean_paired_mse_gain_dex2=float(delta.mean()),conditional_bootstrap95_dex2=np.quantile(boot,[.025,.975]).tolist(),
        selected_depths=[r['adaptive']['depth'] for r in selections],formula_replay_max_abs=max_replay,
        shuffle_reference_fraction=(1+sum(r['mse_gain_percent']>=metrics[0]['mse_gain_percent'] for r in null))/(len(null)+1),
        shuffle_fraction_is_calibrated_p_value=False,limitations=config['limitations'],new_gravity_law=False,
        new_observed_field_scores=0,goal_complete=False)
    write_json(output/'summary.json',summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=['cpu','cuda'],default='cuda'); args=parser.parse_args()
    run(args.output.resolve(),args.backend)
