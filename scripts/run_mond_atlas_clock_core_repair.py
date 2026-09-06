"""Post-hoc developmental central-shape repair; frozen grids, same exposed cohort."""
import csv,hashlib,itertools,json,subprocess,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'work/gravity-first-principles/mond-atlas-clock-relay-001'
DEST=PACKAGE/'source-audit/core-repair'
G=4.30091727003628e-6
A0=1.2e-10*3.085677581491367e19/1e6
FAMILIES=['clock_core_original_scale','clock_core_mass_scale']

def grid():
    return [dict(family=f,mf=mf,beta=beta,scale=scale) for f in FAMILIES
        for mf,beta,scale in itertools.product([.8,1.,1.2],[0.,.3,1.,3.,10.,30.],[.1,1.,10.,100.])]

def extra_acceleration(r,GM,d,beta,psi0):
    return beta*GM*r/((r+d)**2*(r+d+GM/psi0))

def predict(s,c):
    r=s['r'];d=s['rd'];mf=c['mf']
    base=s['gas']*np.abs(s['gas'])+mf*(.5*s['disk']**2+.7*s['bulge']**2)
    GM=G*1e9*(.5*mf*s['luminosity']+1.33*s['hi'])
    psi=c['scale']*(A0*d if c['family']=='clock_core_original_scale' else np.sqrt(GM*A0))
    return .5*np.log10(base+r*extra_acceleration(r,GM,d,c['beta'],psi))

def select(loss,train):
    return int(np.argmin(np.asarray(loss)[:,np.asarray(train,bool)].mean(axis=1)))

def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def table(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run():
    from run_mond_atlas_clock_relay import load_sources
    out=DEST/'run001';out.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'configs/mond_atlas_clock_relay_v1.json').read_text(encoding='utf-8'))
    candidates=grid()
    bind=[Path(__file__),ROOT/'tests/test_mond_atlas_clock_core_repair.py',DEST/'PREFLIGHT.md',ROOT/'configs/mond_atlas_clock_relay_v1.json',ROOT/'scripts/run_mond_atlas_clock_relay.py',ROOT/'scripts/mond_atlas_clock_relay.py',ROOT/'scripts/mond_atlas_common.py',ROOT/'scripts/mond_atlas_pattern_learning.py',PACKAGE/'source-audit/inventory.json']
    inventory=json.loads((PACKAGE/'source-audit/inventory.json').read_text(encoding='utf-8'))
    for key in ['source_archive','source_metadata','registered_development_names']:
        p=ROOT/cfg[key];bind.append(p)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==inventory['files'][cfg[key]]['sha256']
    dump(out/'pre-access-bindings.json',dict(post_hoc_development=True,trigger='Run001 central overprediction already viewed; no confirmation claim',candidate_grid=candidates,bindings={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in bind}))
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_mond_atlas_clock_core_repair.py','-v'],cwd=ROOT,capture_output=True,text=True)
    (out/'tests.log').write_text(test.stdout+test.stderr,encoding='utf-8')
    if test.returncode:raise RuntimeError('Independent manufactured pre-response tests failed')
    sources,y,errors,ids,radialids,names,meta,exclusions,members=load_sources(cfg)
    original=json.loads((PACKAGE/'run001/cohort.json').read_text(encoding='utf-8'))
    assert names==original['names'] and len(y)==original['radial_rows']
    assert len(members)==139
    dump(out/'access-receipt.json',dict(historical_member_bodies_opened=len(members),reserved_member_bodies_opened=0,members=members,galaxies=len(names),radii=len(y)))
    pred=np.array([predict(sources,c) for c in candidates]);assert np.isfinite(pred).all()
    ids=np.array(ids);gi=np.array([names.index(n) for n in ids])
    loss=np.array([np.mean((pred[:,gi==i]-y[gi==i])**2,axis=1) for i in range(len(names))]).T
    choices=[];lossrows=[];heldrows=[];score=[];regionrows=[];galrows=[];held={}
    for seed in cfg['fold_seeds']:
        order=sorted(names,key=lambda n:hashlib.sha256(f'{seed}|{n}'.encode()).digest())
        mapping={n:i%cfg['fold_count'] for i,n in enumerate(order)}
        folds=np.array([mapping[n] for n in names])
        for family in FAMILIES:
            options=np.array([i for i,c in enumerate(candidates) if c['family']==family]);selected=np.zeros(len(names),int)
            for fold in range(cfg['fold_count']):
                train=folds!=fold;ix=int(options[select(loss[options],train)])
                selected[~train]=ix
                choices.append(dict(seed=seed,family=family,fold=fold,candidate_index=ix,candidate=candidates[ix]))
                lossrows.extend(dict(seed=seed,family=family,fold=fold,candidate_index=int(i),training_mse=float(loss[i,train].mean())) for i in options)
            p=pred[selected[gi],np.arange(len(y))];held[(seed,family)]=p
            heldrows.extend(dict(seed=seed,family=family,galaxy=str(n),radial_index=int(radialids[j]),candidate_index=int(selected[gi[j]]),log10_predicted_speed=float(p[j]),log10_predicted_over_observed=float(p[j]-y[j])) for j,n in enumerate(ids))
    # Comparator predictions were frozen in run001 and independently replayed.
    comparisons=['clock_potential','mond_fixed','mond_adjusted']
    lookup={(str(n),int(j)):i for i,(n,j) in enumerate(zip(ids,radialids))}
    for seed in cfg['fold_seeds']:
        for family in comparisons:held[(seed,family)]=np.full(len(y),np.nan)
    replay=PACKAGE/'source-audit/replay/all-family-held-predictions.csv'
    with replay.open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            if row['family'] in comparisons:
                held[(int(row['seed']),row['family'])][lookup[(row['galaxy'],int(row['radial_index']))]]=float(row['log10_predicted_speed'])
    for p in held.values():assert np.isfinite(p).all()
    masks={'all':np.ones(len(y),bool),'inner_lt1':sources['r']/sources['rd']<1,'middle_1to3':(sources['r']/sources['rd']>=1)&(sources['r']/sources['rd']<3),'outer_ge3':sources['r']/sources['rd']>=3}
    for (seed,family),p in held.items():
        residual=p-y
        means=np.array([np.mean(residual[gi==i]**2) for i in range(len(names))])
        score.append(dict(seed=seed,family=family,galaxies=len(names),radii=len(y),mse_logspeed=float(means.mean()),rmse_dex=float(np.sqrt(means.mean()))))
        for i,n in enumerate(names):galrows.append(dict(seed=seed,family=family,galaxy=n,mse_logspeed=float(means[i])))
        for region,mask in masks.items():
            bias=[];mses=[]
            for i,n in enumerate(names):
                subset=mask&(gi==i)
                if subset.any():bias.append(float(residual[subset].mean()));mses.append(float((residual[subset]**2).mean()))
            regionrows.append(dict(seed=seed,family=family,region=region,galaxies=len(bias),radii=int(mask.sum()),mean_signed_logspeed_bias=float(np.mean(bias)),rmse_dex=float(np.sqrt(np.mean(mses))),galaxies_positive_bias=int((np.array(bias)>1e-12).sum())))
    table(out/'all-training-losses.csv',lossrows);dump(out/'selections.json',choices);table(out/'held-predictions.csv',heldrows)
    table(out/'metrics.csv',score);table(out/'region-signed-bias.csv',regionrows);table(out/'galaxy-scores.csv',galrows)
    summary=[]
    baseline={f:np.mean([s['mse_logspeed'] for s in score if s['family']==f]) for f in comparisons}
    for family in FAMILIES+comparisons:
        mse=float(np.mean([s['mse_logspeed'] for s in score if s['family']==family]))
        summary.append(dict(family=family,rmse_dex=float(np.sqrt(mse)),mse_gain_vs_original_clock_percent=float(100*(1-mse/baseline['clock_potential'])),mse_gain_vs_fixed_mond_percent=float(100*(1-mse/baseline['mond_fixed'])),mse_gain_vs_adjusted_mond_percent=float(100*(1-mse/baseline['mond_adjusted']))))
    dump(out/'summary.json',dict(status='POST_HOC_DEVELOPMENT_NOT_CONFIRMATION',summary=summary,source_only_predictor_inputs=True,parameters_selected_using_training_velocities=True,energy_transfer_established=False,comparator_prediction_sha256=hashlib.sha256(replay.read_bytes()).hexdigest()))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
