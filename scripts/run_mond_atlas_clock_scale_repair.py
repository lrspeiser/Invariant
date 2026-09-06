"""One registered post-hoc scale adjustment, no observational reads on import."""
import json,sys,subprocess,time
from pathlib import Path
import numpy as np
from mond_atlas_clock_relay import G,A0,candidate_grid,nfw_mass_shape,predict_logv,loss_select
from mond_atlas_common import ROOT,digest,write_json,write_csv,read_json
from mond_atlas_pattern_learning import galaxy_folds
from run_mond_atlas_clock_relay import load_sources

PACKAGE=ROOT/'work/gravity-first-principles/mond-atlas-clock-relay-001'
OWN=PACKAGE/'physics/scale-repair'
FAMILIES=['mond_fixed','mond_adjusted','clock_potential','kernel_point','finite_p2','finite_p3','finite_mixture']


def predict_repaired(s,c):
    if c['family'].startswith('mond'):return predict_logv(s,c)
    r=np.asarray(s['r']);rd=np.asarray(s['rd']);mf=c['mf']
    GM=G*1e9*(.5*mf*np.asarray(s['luminosity'])+1.33*np.asarray(s['hi']))
    vb2=np.asarray(s['gas'])*abs(np.asarray(s['gas']))+mf*(.5*np.asarray(s['disk'])**2+.7*np.asarray(s['bulge'])**2)
    if c['family']=='clock_potential':
        psi0=c['clock_factor']*np.sqrt(GM*A0)
        extra=c['beta']*GM/((r+rd)*(r+rd+GM/psi0))
    else:
        L=c['length_factor']*np.sqrt(GM/A0);x=r/L
        if c['family']=='kernel_point':extra=c['eta']*GM/r**2*nfw_mass_shape(np.minimum(x,c['cutoff']))
        else:
            p2=1/(1+x)**2;p3=x/(1+x)**3
            shape=p2 if c['family']=='finite_p2' else p3
            if c['family']=='finite_mixture':shape=(1-c['q'])*p2+c['q']*p3
            extra=c['eta']*GM/L**2*shape
    return .5*np.log10(vb2+r*extra)


def run():
    out=OWN/'run001';out.mkdir(parents=True,exist_ok=False)
    try:
        config=read_json(ROOT/'configs/mond_atlas_clock_relay_v1.json')
        binding=[Path(__file__),ROOT/'tests/test_mond_atlas_clock_scale_repair.py',ROOT/'scripts/mond_atlas_clock_relay.py',ROOT/'scripts/run_mond_atlas_clock_relay.py',ROOT/'scripts/mond_atlas_pattern_learning.py',ROOT/'configs/mond_atlas_clock_relay_v1.json',OWN/'PREFLIGHT.md',PACKAGE/'run001/summary.json']
        inventory=read_json(PACKAGE/'source-audit/inventory.json')
        for key in ('source_archive','source_metadata','registered_development_names'):
            p=ROOT/config[key]
            if digest(p)!=inventory['files'][config[key]]['sha256']:raise RuntimeError('Source hash mismatch')
            binding.append(p)
        write_json(out/'pre-access-bindings.json',dict(status='POST_HOC_DEVELOPMENT_ONLY',frozen_original_config=config,files={p.relative_to(ROOT).as_posix():digest(p) for p in binding}))
        test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_mond_atlas_clock_scale_repair.py','-v'],cwd=ROOT,capture_output=True,text=True)
        (out/'tests.log').write_text(test.stdout+test.stderr,encoding='utf-8')
        if test.returncode:raise RuntimeError('Pre-access tests failed')
        candidates=[c for c in candidate_grid(config) if c['family'] in FAMILIES]
        write_json(out/'candidates.json',candidates)
        s,y,errors,ids,radial_ids,names,meta,exclusions,members=load_sources(config)
        write_json(out/'access-receipt.json',dict(members=members,registered_opened=139,reserved_bodies_opened=0,eligible_names=names,galaxies=len(names),radii=len(y)))
        gi=np.array([names.index(n) for n in ids]);pred=np.array([predict_repaired(s,c) for c in candidates])
        if not np.isfinite(pred).all():raise RuntimeError('Nonfinite predictions')
        losses=np.array([np.mean((pred[:,gi==i]-y[None,gi==i])**2,axis=1) for i in range(len(names))]).T
        selections=[];training=[];heldrows=[];galrows=[];metrics=[];bounds=[]
        for seed in config['fold_seeds']:
            folds=galaxy_folds(names,seed,config['fold_count'])
            write_csv(out/f'folds-{seed}.csv',[dict(galaxy=n,fold=int(folds[i])) for i,n in enumerate(names)])
            for family in FAMILIES:
                options=np.array([i for i,c in enumerate(candidates) if c['family']==family]);chosen=np.zeros(len(names),int)
                for fold in range(config['fold_count']):
                    train=folds!=fold;ix=int(options[loss_select(losses[options],train)]);chosen[~train]=ix
                    selections.append(dict(seed=seed,family=family,fold=fold,candidate_index=ix,candidate=candidates[ix]))
                    for i in options:training.append(dict(seed=seed,family=family,fold=fold,candidate_index=int(i),training_mse=float(losses[i,train].mean())))
                    for key,value in candidates[ix].items():
                        if key in ('family','cutoff'):continue
                        available=sorted(set(candidates[i][key] for i in options))
                        if len(available)>1:bounds.append(dict(seed=seed,family=family,fold=fold,parameter=key,value=value,lower_bound=value==available[0],upper_bound=value==available[-1]))
                hp=pred[chosen[gi],np.arange(len(y))]
                for j,n in enumerate(ids):heldrows.append(dict(seed=seed,family=family,galaxy=n,radial_index=int(radial_ids[j]),candidate_index=int(chosen[gi[j]]),log10_predicted_speed=float(hp[j]),log10_predicted_over_observed=float(hp[j]-y[j])))
                mse=np.array([np.mean((hp[gi==i]-y[gi==i])**2) for i in range(len(names))])
                for i,n in enumerate(names):galrows.append(dict(seed=seed,family=family,galaxy=n,mse_logspeed=float(mse[i])))
                metrics.append(dict(seed=seed,family=family,mse=float(mse.mean()),rmse_dex=float(np.sqrt(mse.mean()))))
        for name,rows in [('all-training-losses.csv',training),('all-held-radial-residuals.csv',heldrows),('galaxy-scores.csv',galrows),('metrics.csv',metrics),('parameter-boundaries.csv',bounds)]:write_csv(out/name,rows)
        write_json(out/'selections.json',selections)
        original={v['family']:v for v in read_json(PACKAGE/'run001/summary.json')['metrics']}
        means={f:float(np.mean([v['mse'] for v in metrics if v['family']==f])) for f in FAMILIES}
        summary=[]
        for f in FAMILIES:summary.append(dict(family=f,rmse_dex=float(np.sqrt(means[f])),mse_improvement_vs_original_percent=100*(1-means[f]/original[f]['rmse_dex']**2),mse_improvement_vs_fixed_mond_percent=100*(1-means[f]/means['mond_fixed']),mse_improvement_vs_adjusted_mond_percent=100*(1-means[f]/means['mond_adjusted'])))
        write_json(out/'summary.json',dict(status='POST_HOC_DEVELOPMENT_ONLY',galaxies=len(names),radii=len(y),candidates=len(candidates),metrics=summary,time_energy_claim=False))
        print(json.dumps(summary,indent=2))
    except Exception as exc:
        write_json(out/'failure.json',dict(error=repr(exc)));raise


if __name__=='__main__':run()
