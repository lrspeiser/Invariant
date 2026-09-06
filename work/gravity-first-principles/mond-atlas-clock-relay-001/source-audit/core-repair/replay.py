"""Independent algebraic evaluator replay; shared previously audited SPARC loader."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5]
HERE=Path(__file__).resolve().parent
RUN=HERE/'run001'
sys.path.insert(0,str(ROOT/'scripts'))
from run_mond_atlas_clock_relay import load_sources
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
rows=lambda p:list(csv.DictReader(p.open(encoding='utf-8',newline='')))
cfg=read(ROOT/'configs/mond_atlas_clock_relay_v1.json')
grid=read(RUN/'pre-access-bindings.json')['candidate_grid']
s,y,errors,ids,radialids,names,meta,exclusions,members=load_sources(cfg)
ids=np.array(ids);r=s['r'];d=s['rd'];a0=1.2e-10*3.085677581491367e19/1e6
pred=[]
for c in grid:
    gm=4.30091727003628e-6*1e9*(.5*c['mf']*s['luminosity']+1.33*s['hi'])
    base=s['gas']*abs(s['gas'])+c['mf']*(.5*s['disk']**2+.7*s['bulge']**2)
    psi=c['scale']*(a0*d if c['family']=='clock_core_original_scale' else np.sqrt(gm*a0))
    # Rearranged to remove division by psi and compute added speed squared directly.
    v2=base+c['beta']*gm*psi*(r/(r+d))**2/(psi*(r+d)+gm)
    pred.append(np.log10(v2)/2)
pred=np.array(pred)
loss=np.array([((pred[:,ids==n]-y[ids==n])**2).mean(axis=1) for n in names]).T
folds={seed:{n:i%5 for i,n in enumerate(sorted(names,key=lambda n:hashlib.sha256(f'{seed}|{n}'.encode()).digest()))} for seed in cfg['fold_seeds']}
trainerr=0.
for row in rows(RUN/'all-training-losses.csv'):
    train=np.array([folds[int(row['seed'])][n]!=int(row['fold']) for n in names])
    value=loss[int(row['candidate_index']),train].mean()
    trainerr=max(trainerr,abs(float(value)-float(row['training_mse'])))
gap=0.
for row in read(RUN/'selections.json'):
    train=np.array([folds[row['seed']][n]!=row['fold'] for n in names])
    options=[i for i,c in enumerate(grid) if c['family']==row['family']]
    gap=max(gap,float(loss[row['candidate_index'],train].mean()-loss[options][:,train].mean(axis=1).min()))
lookup={(str(n),int(j)):i for i,(n,j) in enumerate(zip(ids,radialids))}
helderr=0.
for row in rows(RUN/'held-predictions.csv'):
    ix=lookup[(row['galaxy'],int(row['radial_index']))];value=pred[int(row['candidate_index']),ix]
    helderr=max(helderr,abs(value-float(row['log10_predicted_speed'])),abs(value-y[ix]-float(row['log10_predicted_over_observed'])))
assert max(trainerr,gap,helderr)<1e-10
result=dict(status='PASS',independent_formula_evaluator=True,shared_source_loader=True,candidates=len(grid),galaxies=len(names),radii=len(y),historical_member_bodies_opened=len(members),reserved_member_bodies_opened=0,training_loss_max_abs=trainerr,selection_loss_gap_max=gap,held_prediction_max_abs=helderr,
    bindings={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'scripts/run_mond_atlas_clock_relay.py',*RUN.iterdir()] if p.is_file()})
(HERE/'replay-receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='bindings'},indent=2))
