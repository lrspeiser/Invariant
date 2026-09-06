"""Whole-galaxy prediction tests of frozen radial relay/clock formula grids."""
import argparse,csv,hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT,digest,write_json,write_csv,read_json
from mond_atlas_pattern_learning import galaxy_folds
from mond_atlas_clock_relay import candidate_grid,predict_logv,loss_select

PACKAGE=ROOT/'work/gravity-first-principles/mond-atlas-clock-relay-001'

def load_sources(config):
    names=read_json(ROOT/config['registered_development_names'])['names']
    if len(names)!=139 or len(set(names))!=139:raise RuntimeError('Registered identity mismatch')
    meta={}
    for line in (ROOT/config['source_metadata']).read_text(encoding='utf-8').splitlines():
        f=line.split()
        if f and f[0] in names:
            if len(f)!=19:raise RuntimeError('Metadata schema')
            meta[f[0]]=dict(hubble_type=int(f[1]),distance=float(f[2]),inc=float(f[5]),luminosity=float(f[7]),rd=float(f[11]),hi=float(f[13]),quality=int(f[17]))
    arrays={k:[] for k in ['r','gas','disk','bulge','sb','luminosity','hi','rd']}
    observed=[];uncertainty=[];identifiers=[];indices=[];exclusions=[];members=[];galaxies=[]
    with zipfile.ZipFile(ROOT/config['source_archive']) as archive:
        lookup={Path(p).name:p for p in archive.namelist()}
        for name in sorted(names):
            member=lookup[name+'_rotmod.dat'];raw=archive.read(member)
            members.append(dict(galaxy=name,member=member,sha256=hashlib.sha256(raw).hexdigest()))
            rows=[l.split() for l in raw.decode('utf-8').splitlines() if l.strip() and not l.startswith('#')]
            if any(len(v)!=8 for v in rows):raise RuntimeError('Curve schema')
            x=np.asarray(rows,float);m=meta[name]
            valid=np.isfinite(x).all(axis=1)&(x[:,0]>0)&(x[:,1]>0)&(x[:,2]>0)
            for mf in config['mass_factors']:
                valid&=(x[:,3]*abs(x[:,3])+mf*(.5*x[:,4]**2+.7*x[:,5]**2))>0
            reasons=[]
            if m['quality']>2:reasons.append('quality')
            if not 30<=m['inc']<=80:reasons.append('inclination')
            if m['rd']<=0 or m['luminosity']<=0 or m['hi']<0:reasons.append('source_scale_or_mass')
            if valid.sum()<5:reasons.append('insufficient_eligible_radii')
            if reasons:
                exclusions.append(dict(galaxy=name,reason=';'.join(reasons),total_radii=len(x),eligible_radii=int(valid.sum())))
                continue
            galaxies.append(name)
            for j,row in enumerate(x):
                if not valid[j]:
                    exclusions.append(dict(galaxy=name,reason='invalid_radius_'+str(j),total_radii=1,eligible_radii=0));continue
                for key,column in [('r',0),('gas',3),('disk',4),('bulge',5),('sb',6)]:arrays[key].append(row[column])
                for key in ['luminosity','hi','rd']:arrays[key].append(m[key])
                observed.append(row[1]);uncertainty.append(row[2]);identifiers.append(name);indices.append(j)
    return {k:np.asarray(v) for k,v in arrays.items()},np.log10(observed),np.asarray(uncertainty),identifiers,indices,galaxies,meta,exclusions,members

def run(output,backend):
    start=time.perf_counter();output.mkdir(parents=True,exist_ok=False)
    config_path=ROOT/'configs/mond_atlas_clock_relay_v1.json';config=read_json(config_path)
    bound=[config_path,Path(__file__),ROOT/'scripts/mond_atlas_clock_relay.py',ROOT/'tests/test_mond_atlas_clock_relay.py',ROOT/'scripts/mond_atlas_pattern_learning.py',ROOT/'scripts/mond_atlas_common.py',PACKAGE/'PREFLIGHT.md']
    inventory=read_json(PACKAGE/'source-audit/inventory.json')
    for p in [config['source_archive'],config['source_metadata'],config['registered_development_names']]:
        if digest(ROOT/p)!=inventory['files'][p]['sha256']:raise RuntimeError('Source hash mismatch')
        bound.append(ROOT/p)
    write_json(output/'pre-access-bindings.json',dict(disposition='DATA_AND_PAPER_ADMITTED',scope='RADIAL_EMPIRICAL_ONLY',config=config,bindings={p.relative_to(ROOT).as_posix():digest(p) for p in bound},historically_exposed=True))
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_mond_atlas_clock_relay.py','-v'],cwd=ROOT,capture_output=True,text=True)
    (output/'tests.log').write_text(test.stdout+test.stderr,encoding='utf-8')
    if test.returncode:raise RuntimeError('Pre-response tests failed')
    candidates=candidate_grid(config);write_json(output/'candidate-formulas.json',candidates)
    xp=np;runtime=dict(backend=backend,python=sys.version)
    if backend=='cuda':
        import cupy as cp
        xp=cp;cp.get_default_memory_pool().set_limit(size=1024**3)
        runtime.update(cupy=cp.__version__,device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
    # Manufactured CPU/GPU parity before response access, all candidate families.
    manufactured=dict(r=np.array([.1,1,3,10,100.]),gas=np.array([-1,5,10,20,30.]),disk=np.array([10,20,30,40,50.]),bulge=np.ones(5)*3,sb=np.array([1,3,10,30,100.]),luminosity=np.ones(5)*3,hi=np.ones(5)*2,rd=np.ones(5)*2)
    mismatch=0.
    for candidate in candidates:
        a=predict_logv(manufactured,candidate,np);b=predict_logv(manufactured,candidate,xp)
        if xp is not np:b=xp.asnumpy(b)
        mismatch=max(mismatch,float(np.max(abs(a-b))))
    write_json(output/'pre-access-gpu-control.json',dict(candidate_count=len(candidates),cpu_gpu_max_abs=mismatch,passed=mismatch<1e-10))
    if mismatch>=1e-10:raise RuntimeError('GPU parity failed')
    sources,y,errors,ids,radial_ids,names,meta,exclusions,members=load_sources(config)
    write_json(output/'access-receipt.json',dict(opened_archive_members=members,opened_identities=len(members),reserved_member_bodies_opened=0,source_provenance='SPARC2016 source components and metadata; no imported halo quantities'))
    write_csv(output/'exclusions.csv',exclusions,fields=['galaxy','reason','total_radii','eligible_radii'])
    write_json(output/'cohort.json',dict(names=names,galaxies=len(names),radial_rows=len(y),registered_names=139,excluded_galaxies=139-len(names)))
    gindex=np.array([names.index(n) for n in ids]);source_gpu={k:xp.asarray(v) for k,v in sources.items()}
    preds=xp.stack([predict_logv(source_gpu,c,xp) for c in candidates])
    if not bool(xp.isfinite(preds).all()):raise RuntimeError('Nonfinite candidate; keep failure')
    pred=xp.asnumpy(preds) if xp is not np else preds
    se=(pred-y[None,:])**2
    losses=np.column_stack([se[:,gindex==i].mean(axis=1) for i in range(len(names))])
    if not np.isfinite(losses).all():raise RuntimeError('Nonfinite loss')
    write_json(output/'source-prediction-binding.json',dict(source_definitions=config['source_definitions'],prediction_shape=list(pred.shape),candidate_loss_sha256=hashlib.sha256(losses.tobytes()).hexdigest()))
    families=config['families']+['training_selected'];allmetrics=[];galrows=[];selections=[];selection_losses=[];selectedrows=[];strata=[];replay_error=0.
    saved={};base=next(i for i,c in enumerate(candidates) if c['family']=='mond_fixed')
    for seed in config['fold_seeds']:
        folds=galaxy_folds(names,seed,config['fold_count'])
        write_csv(output/f'folds-{seed}.csv',[dict(galaxy=n,fold=int(folds[i])) for i,n in enumerate(names)])
        for family in families:
            options=np.arange(len(candidates)) if family=='training_selected' else np.array([i for i,c in enumerate(candidates) if c['family']==family])
            chosen_per_gal=np.zeros(len(names),int)
            for held in range(config['fold_count']):
                train=folds!=held;ix=int(options[loss_select(losses[options],train)])
                chosen_per_gal[folds==held]=ix
                selections.append(dict(seed=seed,family=family,fold=held,candidate_index=ix,training_galaxies=int(train.sum()),test_galaxies=int((~train).sum()),candidate=candidates[ix]))
                selection_losses.extend(dict(seed=seed,family=family,fold=held,candidate_index=int(i),training_mse=float(losses[i,train].mean())) for i in options)
            held_prediction=pred[chosen_per_gal[gindex],np.arange(len(y))]
            for ix in np.unique(chosen_per_gal):
                rows=chosen_per_gal[gindex]==ix
                check=predict_logv({k:v[rows] for k,v in sources.items()},candidates[ix],np)
                replay_error=max(replay_error,float(np.max(abs(check-held_prediction[rows]))))
            galmse=np.array([np.mean((held_prediction[gindex==i]-y[gindex==i])**2) for i in range(len(names))])
            kmse=np.array([np.mean((10**held_prediction[gindex==i]-10**y[gindex==i])**2) for i in range(len(names))])
            saved[(seed,family)]=galmse
            allmetrics.append(dict(seed=seed,family=family,galaxies=len(names),radii=len(y),equal_galaxy_logspeed_rmse=float(np.sqrt(galmse.mean())),equal_galaxy_speed_rmse_kms=float(np.sqrt(kmse.mean())),mse_gain_vs_fixed_mond_percent=float(100*(losses[base].mean()-galmse.mean())/losses[base].mean())))
            for i,n in enumerate(names):
                galrows.append(dict(seed=seed,family=family,galaxy=n,fold=int(folds[i]),candidate_index=int(chosen_per_gal[i]),radii=int((gindex==i).sum()),mse_logspeed=float(galmse[i]),mse_speed_kms=float(kmse[i])))
            if family=='training_selected':
                selectedrows.extend(dict(seed=seed,galaxy=n,radial_index=int(radial_ids[i]),candidate_index=int(chosen_per_gal[gindex[i]]),log10_predicted_speed=float(held_prediction[i]),log10_predicted_over_observed=float(held_prediction[i]-y[i])) for i,n in enumerate(ids))
            groups={'inner_r_over_Rd_lt1':sources['r']/sources['rd']<1,'middle_1to3':(sources['r']/sources['rd']>=1)&(sources['r']/sources['rd']<3),'outer_ge3':sources['r']/sources['rd']>=3,
                'gas_rich_proxy':1.33*sources['hi']/(1.33*sources['hi']+.5*sources['luminosity'])>=.5,
                'stellar_rich_proxy':1.33*sources['hi']/(1.33*sources['hi']+.5*sources['luminosity'])<.5}
            for group,mask in groups.items():
                vals=[np.mean((held_prediction[(gindex==i)&mask]-y[(gindex==i)&mask])**2) for i in range(len(names)) if np.any((gindex==i)&mask)]
                strata.append(dict(seed=seed,family=family,group=group,galaxies=len(vals),radii=int(mask.sum()),logspeed_rmse=float(np.sqrt(np.mean(vals)))))
    write_csv(output/'metrics.csv',allmetrics);write_csv(output/'galaxy-held-scores.csv',galrows);write_json(output/'selections.json',selections)
    write_csv(output/'all-training-losses.csv',selection_losses);write_csv(output/'selected-radial-residuals.csv',selectedrows);write_csv(output/'strata.csv',strata)
    rng=np.random.default_rng(config['bootstrap_seed']);summary=[]
    for family in families:
        meanloss=np.mean([saved[(seed,family)] for seed in config['fold_seeds']],axis=0);delta=losses[base]-meanloss
        boot=rng.choice(delta,(config['bootstrap_replicates'],len(names))).mean(axis=1)
        summary.append(dict(family=family,rmse_dex=float(np.sqrt(meanloss.mean())),mse_gain_vs_fixed_mond_percent=float(100*delta.mean()/losses[base].mean()),paired_bootstrap95_mse_dex2=np.quantile(boot,[.025,.975]).tolist(),galaxies_improved_vs_fixed_mond=int((delta>0).sum())))
    if replay_error>=1e-10:raise RuntimeError('CPU held prediction replay failed')
    runtime.update(seconds=time.perf_counter()-start,cpu_held_replay_max_abs=replay_error,candidates=len(candidates),galaxies=len(names),radii=len(y))
    write_json(output/'runtime.json',runtime);write_json(output/'summary.json',dict(status='EXPLORATORY_REAL_RADIAL_COMPARISON',metrics=summary,energy_conservation_established=False,time_energy_transfer_measured=False,full_3d_prediction=False,source_only_parameters=True,blocked_branches=config['blocked_branches']))
    print(json.dumps(dict(runtime=runtime,metrics=summary),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--backend',choices=['cpu','cuda'],default='cuda');a=p.parse_args();run(a.output.resolve(),a.backend)
