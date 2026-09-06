"""Frozen continuous nonclock radial development comparisons."""
import json,sys,subprocess
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from mond_atlas_clock_relay import A0
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest
from mond_atlas_pattern_learning import galaxy_folds
from run_mond_atlas_clock_relay import load_sources

P=ROOT/'work/gravity-first-principles/mond-atlas-nonclock-robust-001/coherence'
SPECS={
 'newton':(['mf'],[.8],[2.]),
 'mond_adjusted':(['mf','log_a0'],[.8,np.log(.1)],[2.,np.log(3.)]),
 'coherence_n1':(['mf','A','log_sigma0'],[.8,0,np.log(.1)],[2.,100,np.log(1e4)]),
 'coherence_free':(['mf','A','log_sigma0','n'],[.8,0,np.log(.1),.25],[2.,100,np.log(1e4),4.]),
 'relay_passive':(['mf','k','eta'],[.8,0,0],[2.,10,1]),
 'relay_active':(['mf','k','eta'],[.8,0,0],[2.,10,30])}
FAMILIES=['mond_fixed',*SPECS]

def predict(s,f,p):
    if f=='mond_fixed':f='mond_adjusted';p=[1.,0.]
    mf=p[0];gb=(s['gas']*abs(s['gas'])+mf*(.5*s['disk']**2+.7*s['bulge']**2))/s['r']
    sigma=.5*mf*np.maximum(s['sb'],0)
    if f=='newton':boostlog=np.zeros_like(gb)
    elif f=='mond_adjusted':boostlog=np.log(.5+np.sqrt(.25+A0*np.exp(p[1])/gb))
    elif f.startswith('coherence'):
        n=p[3] if f=='coherence_free' else 1.
        boostlog=np.log1p(p[1]/(1+(sigma/np.exp(p[2]))**n))
    elif f.startswith('relay'):
        tau=p[1]*sigma/100
        with np.errstate(divide='ignore',invalid='ignore'):
            boostlog=np.logaddexp(-tau,np.log(p[2])+np.log(-np.expm1(-tau)))
    else:raise ValueError(f)
    return .5*(np.log(s['r']*gb)+boostlog)/np.log(10)

def fit(s,y,galaxy_index,train,f):
    # Only training source rows and labels enter the optimizer.
    rows=np.asarray(train,dtype=bool);ss={k:v[rows] for k,v in s.items()};yy=y[rows];gg=galaxy_index[rows]
    unique,counts=np.unique(gg,return_counts=True)
    count=dict(zip(unique,counts));w=np.array([1/np.sqrt(len(unique)*count[i]) for i in gg])
    labels,lower,upper=SPECS[f];lower=np.array(lower);upper=np.array(upper)
    attempts=[]
    for start in (.15,.5,.85):
        try:
            result=least_squares(lambda p:(predict(ss,f,p)-yy)*w,lower+start*(upper-lower),bounds=(lower,upper),max_nfev=1500,ftol=1e-9,xtol=1e-9,gtol=1e-9)
            loss=float(np.sum(result.fun**2));good=bool(result.success and np.isfinite(loss))
            attempts.append(dict(start=start,success=good,parameters=result.x.tolist(),training_mse=loss,nfev=int(result.nfev),message=result.message,boundary_parameters=[labels[i] for i,x in enumerate(result.x) if min(x-lower[i],upper[i]-x)<=1e-4*(upper[i]-lower[i])]))
        except Exception as exc:attempts.append(dict(start=start,success=False,error=repr(exc)))
    good=[a for a in attempts if a['success']]
    if not good:return None,attempts
    return min(good,key=lambda a:a['training_mse']),attempts

def run():
    out=P/'run001';out.mkdir(parents=True,exist_ok=False)
    try:
        config=read_json(ROOT/'configs/mond_atlas_clock_relay_v1.json')
        paths=[Path(__file__),ROOT/'tests/test_mond_atlas_coherence_robust.py',P/'PREFLIGHT.md',ROOT/'scripts/run_mond_atlas_clock_relay.py',ROOT/'configs/mond_atlas_clock_relay_v1.json']+[ROOT/config[k] for k in ('source_archive','source_metadata','registered_development_names')]
        inventory=read_json(ROOT/'work/gravity-first-principles/mond-atlas-clock-relay-001/source-audit/inventory.json')
        for k in ('source_archive','source_metadata','registered_development_names'):
            if digest(ROOT/config[k])!=inventory['files'][config[k]]['sha256']:raise RuntimeError('Source hash mismatch')
        write_json(out/'pre-access-bindings.json',dict(status='POST_HOC_DEVELOPMENT_ONLY',specs=SPECS,files={p.relative_to(ROOT).as_posix():digest(p) for p in paths}))
        test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_mond_atlas_coherence_robust.py','-v'],cwd=ROOT,capture_output=True,text=True)
        (out/'tests.log').write_text(test.stdout+test.stderr,encoding='utf-8')
        if test.returncode:raise RuntimeError('Pre-response tests failed')
        s,y,errors,ids,radial_ids,names,meta,exclusions,members=load_sources(config)
        write_json(out/'access-receipt.json',dict(members=members,eligible_names=names,radii=len(y),reserved_member_bodies_opened=0))
        gi=np.array([names.index(n) for n in ids]);attempts=[];choices=[];heldrows=[];galrows=[];strata=[];metrics=[]
        for seed in config['fold_seeds']:
            folds=galaxy_folds(names,seed,config['fold_count']);foldrows=folds[gi]
            write_csv(out/f'folds-{seed}.csv',[dict(galaxy=n,fold=int(folds[i])) for i,n in enumerate(names)])
            hp={f:np.zeros_like(y) for f in FAMILIES}
            for fold in range(config['fold_count']):
                train=foldrows!=fold;selected={}
                for f in SPECS:
                    best,tries=fit(s,y,gi,train,f)
                    attempts.extend(dict(seed=seed,fold=fold,family=f,**a) for a in tries)
                    if best is None:
                        write_json(out/'attempts.json',attempts);raise RuntimeError('No successful optimizer starts '+f)
                    selected[f]=dict(evaluator=f,**best)
                for f in FAMILIES:
                    if f=='mond_fixed':choice=dict(evaluator=f,parameters=[],training_mse=float(np.mean([np.mean((predict(s,f,[])[(gi==i)&train]-y[(gi==i)&train])**2) for i in range(len(names)) if folds[i]!=fold])),boundary_parameters=[])
                    else:
                        choice=selected[f]
                        if f.startswith(('coherence','relay')) and selected['newton']['training_mse']<=choice['training_mse']:choice=selected['newton']
                    choices.append(dict(seed=seed,fold=fold,family=f,**choice))
                    hp[f][~train]=predict({k:v[~train] for k,v in s.items()},choice['evaluator'],choice['parameters'])
            for f in FAMILIES:
                residual=hp[f]-y;mse=[]
                for i,n in enumerate(names):
                    val=float(np.mean(residual[gi==i]**2));mse.append(val);galrows.append(dict(seed=seed,family=f,galaxy=n,mse=val,signed_mean_dex=float(np.mean(residual[gi==i]))))
                metrics.append(dict(seed=seed,family=f,mse=float(np.mean(mse)),rmse_dex=float(np.sqrt(np.mean(mse)))))
                for j,n in enumerate(ids):heldrows.append(dict(seed=seed,family=f,galaxy=n,radial_index=int(radial_ids[j]),log10_predicted_speed=float(hp[f][j]),residual_dex=float(residual[j])))
                ratio=s['r']/s['rd']
                for region,mask in [('inner',ratio<1),('middle',(ratio>=1)&(ratio<3)),('outer',ratio>=3)]:
                    vals=[residual[(gi==i)&mask] for i in range(len(names)) if np.any((gi==i)&mask)]
                    strata.append(dict(seed=seed,family=f,region=region,galaxies=len(vals),radii=int(mask.sum()),signed_mean_dex=float(np.mean([v.mean() for v in vals])),rmse_dex=float(np.sqrt(np.mean([np.mean(v*v) for v in vals])))))
        write_json(out/'attempts.json',attempts);write_json(out/'selections.json',choices)
        for name,rows in [('metrics.csv',metrics),('galaxy-scores.csv',galrows),('held-predictions.csv',heldrows),('strata.csv',strata)]:write_csv(out/name,rows)
        means={f:np.mean([m['mse'] for m in metrics if m['family']==f]) for f in FAMILIES}
        summary=dict(status='POST_HOC_DEVELOPMENT_ONLY',galaxies=len(names),radii=len(y),optimizer_starts=len(attempts),optimizer_failures=sum(not a['success'] for a in attempts),metrics=[dict(family=f,rmse_dex=float(np.sqrt(means[f])),mse_gain_vs_continuous_mond_percent=float(100*(1-means[f]/means['mond_adjusted']))) for f in FAMILIES],free_n_gain_vs_n1_percent=float(100*(1-means['coherence_free']/means['coherence_n1'])))
        write_json(out/'summary.json',summary);print(json.dumps(summary,indent=2))
    except Exception as exc:write_json(out/'failure.json',dict(error=repr(exc)));raise

if __name__=='__main__':run()
