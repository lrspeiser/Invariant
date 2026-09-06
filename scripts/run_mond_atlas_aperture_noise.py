"""Frozen external-background aperture transfer; preserves every scale result."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import argparse,csv,json,subprocess,sys
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
from mond_atlas_aperture_noise import fit,scores


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    config_path=ROOT/'configs/mond_atlas_aperture_noise_v1.json'; c=read_json(config_path)
    source=ROOT/c['input']; assert digest(source)==c['input_sha256']
    paths=[config_path,source,Path(__file__),ROOT/'scripts/mond_atlas_aperture_noise.py',
           ROOT/'scripts/mond_atlas_native_covariance.py',ROOT/'tests/test_mond_atlas_aperture_noise.py',
           ROOT/'tests/test_mond_atlas_native_covariance.py',out.parent/'PREFLIGHT.md']
    write_json(out/'bindings.json',dict(files={p.relative_to(ROOT).as_posix():digest(p) for p in paths},
                new_values_opened=False,previous_development_exposure=True))
    logs=[]
    for name in ['test_mond_atlas_native_covariance.py','test_mond_atlas_aperture_noise.py']:
        result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',name],cwd=ROOT,capture_output=True,text=True)
        logs.append(result.stdout+result.stderr)
        (out/'tests.log').write_text('\n'.join(logs),encoding='utf-8')
        if result.returncode: raise RuntimeError('Pre-access benchmark failed')
    with np.load(source) as packet: training=packet['training']
    mean,models=fit(training,c['sides_native_pixels'])
    model_path=out/'models-before-east.json'
    write_json(model_path,dict(mean=mean.tolist(),covariances={str(k):v.tolist() for k,v in models.items()},
                              eastern_values_read_in_this_run=False))
    model_hash=digest(model_path)
    with np.load(source) as packet: east=packet['validation']
    assert training.shape==(29,24,24,42) and east.shape==(27,24,24,42)
    rows=[];summary=[]
    for side in c['sides_native_pixels']:
        for kind,cov in [('independent_pixels',models[1]/side**2),('empirical_aperture',models[side])]:
            score=scores(east,mean,cov,side)
            rows.extend(dict(side_pixels=side,model=kind,core=i,**{k:float(v[i]) for k,v in score.items()}) for i in range(len(east)))
            q=float(score['q_over_n'].mean())
            summary.append(dict(side_pixels=side,side_arcsec=side*1.5,tiles_per_core=(24//side)**2,model=kind,
                mean_q_over_n=q,minimum_core_q=float(score['q_over_n'].min()),maximum_core_q=float(score['q_over_n'].max()),
                mean_logpdf_per_channel=float(score['logpdf_per_channel'].mean()),
                east_trace_second_moment=float(score['trace_second_moment'].mean()),predicted_trace=float(np.trace(cov)),
                trace_ratio_east_to_prediction=float(score['trace_second_moment'].mean()/np.trace(cov)),
                descriptive_q_pass=c['descriptive_q_range'][0]<=q<=c['descriptive_q_range'][1]))
    assert digest(model_path)==model_hash
    write_csv(out/'core-scores.csv',rows);write_csv(out/'scale-summary.csv',summary)
    write_json(out/'summary.json',dict(status='EXECUTED_BACKGROUND_APERTURE_TRANSFER',disposition='SOURCE_BLOCKED',
        training_cores=29,validation_cores=27,models_frozen_before_east_sha256=model_hash,tests_passed=17,
        scales=summary,limitations=c['limitations'],observed_gravity_scores=0,admitted_cube_likelihoods=0,goal_complete=False))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();run(args.output.resolve())
