"""Independent algebraic replay of saved continuous fits; no optimizer refitting."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from run_mond_atlas_clock_relay import load_sources
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
rows=lambda p:list(csv.DictReader(p.open(encoding='utf-8',newline='')))
cfg=read(ROOT/'configs/mond_atlas_clock_relay_v1.json')
s,y,errors,ids,radids,names,meta,exclusions,members=load_sources(cfg)
assert names==read(ROOT/'work/gravity-first-principles/mond-atlas-clock-relay-001/run001/cohort.json')['names']
assert len(names)==102 and len(y)==2212 and len(members)==139
ids=np.array(ids);gi=np.array([names.index(n) for n in ids]);lookup={(str(n),int(j)):i for i,(n,j) in enumerate(zip(ids,radids))}
folds={seed:dict((n,i%5) for i,n in enumerate(sorted(names,key=lambda n:hashlib.sha256(f'{seed}|{n}'.encode()).digest()))) for seed in cfg['fold_seeds']}
rich=(1.33*s['hi']/(1.33*s['hi']+.5*s['luminosity']))>=.5
A0=1.2e-10*3.085677581491367e19/1e6;G=4.30091727003628e-6

def independent(f,p,branch):
    if branch=='coherence':
        if f=='mond_fixed':p=[1.,0.]
        mf=p[0];v2=s['gas']*abs(s['gas'])+mf*(.5*s['disk']**2+.7*s['bulge']**2)
        if f=='newton':return .5*np.log10(v2)
        if f.startswith('mond'):
            return .5*np.log10(v2/2*(1+np.sqrt(1+4*A0*np.exp(p[1])*s['r']/v2)))
        sigma=.5*mf*np.maximum(s['sb'],0)
        if f.startswith('coherence'):
            n=p[3] if f=='coherence_free' else 1
            denominator=np.exp(p[2]*n)+sigma**n
            return .5*np.log10(v2*(1+p[1]*np.exp(p[2]*n)/denominator))
        tau=p[1]*sigma/100;eta=p[2]
        logboost=-tau if eta==0 else np.log(eta+(1-eta)*np.exp(-tau))
        return .5*(np.log(v2)+logboost)/np.log(10)
    mf=p.get('mf',1);r=s['r'];d=s['rd']
    v2=s['gas']*abs(s['gas'])+mf*(.5*s['disk']**2+.7*s['bulge']**2)
    if f.startswith('mond'):return .5*np.log10(v2/2*(1+np.sqrt(1+4*A0*p.get('a0_factor',1)*r/v2)))
    gm=G*1e9*(.5*mf*s['luminosity']+1.33*s['hi']);rM=np.sqrt(gm/A0)
    if f=='finite_flat_bridge':
        extra=p['eta']*p['C']*gm*r*r/((r+p['delta']*d)**2*(r+p['C']*rM))
    else:
        L=p['length_factor']*np.exp((1-p['t'])*np.log(d)+p['t']*np.log(rM))
        if f=='finite_mix':extra=p['A']*gm*((1-p['q'])*r/(r+L)**2+p['q']*r*r/(r+L)**3)
        else:
            x=np.minimum(r/L,p['C']);mass=np.log1p(x)-x/(1+x)
            low=x<.001;mass[low]=sum((-1.)**k*(k-1)/k*x[low]**k for k in range(2,14))
            extra=p['A']*gm*mass/r
    return .5*np.log10(v2+extra)

def mask_for(row):
    if 'direction' in row:return rich if row['direction'] in ('gas_to_stellar','gas_rich_to_stellar_rich') else ~rich
    return np.array([folds[int(row['seed'])][n]!=int(row['fold']) for n in ids])

def loss(pred,mask):return float(np.mean([np.mean((pred[(gi==i)&mask]-y[(gi==i)&mask])**2) for i in range(len(names)) if np.any((gi==i)&mask)]))

branch=sys.argv[1];run=sys.argv[2]
transfer=len(sys.argv)>3 and sys.argv[3]=='transfer'
src=HERE.parent/branch/run
assert (src/'summary.json').exists(), 'Run not complete'
attemptpath=src/('attempts.json' if branch=='coherence' else 'all-optimizer-starts.json')
choicepath=src/('selections.json' if branch=='coherence' else 'selected-parameters.json')
if transfer:attemptpath=src/'transfer-all-starts.json';choicepath=src/'transfer-selected.json'
attempts=read(attemptpath);choices=read(choicepath)
key=lambda x:(x.get('direction',x.get('seed')),x.get('fold'),x['family'])
attempt_error=0;predcache={};success_count=0
for a in attempts:
    if 'parameters' not in a:continue
    p=independent(a['family'],a['parameters'],branch)
    value=loss(p,mask_for(a))
    attempt_error=max(attempt_error,abs(value-a['training_mse']))
    success_count+=int(a['success'])
assert attempt_error<1e-9
selected_gap=0
for c in choices:
    p=independent(c.get('evaluator',c['family']),c['parameters'],branch)
    predcache[key(c)]=p
    if c['family']=='mond_fixed':continue
    permitted=[a for a in attempts if key(a)==key(c) and a['success']]
    if branch=='coherence' and c['family'].startswith(('coherence','relay')):
        permitted += [a for a in attempts if key(a)[:2]==key(c)[:2] and a['family']=='newton' and a['success']]
    gap=loss(p,mask_for(c))-min(a['training_mse'] for a in permitted)
    selected_gap=max(selected_gap,gap)
assert selected_gap<1e-9
held_error=0;held_by={}
heldpath=src/('transfer-predictions.csv' if transfer else 'held-predictions.csv')
for row in rows(heldpath):
    i=lookup[(row['galaxy'],int(row['radial_index']))]
    if 'direction' in row:k=(row['direction'],None,row['family']);hkey=(row['direction'],row['family'])
    else:
        seed=int(row['seed']);k=(seed,folds[seed][row['galaxy']],row['family']);hkey=(seed,row['family'])
    p=predcache[k][i]
    held_error=max(held_error,abs(p-float(row['log10_predicted_speed'])))
    residual=row.get('residual_dex',row.get('log10_predicted_over_observed'))
    held_error=max(held_error,abs(p-y[i]-float(residual)))
    if hkey not in held_by:held_by[hkey]=np.full(len(y),np.nan)
    held_by[hkey][i]=p
assert held_error<1e-9
summary=read(src/'summary.json');metric_error=0
metricrows=rows(src/'transfer-metrics.csv') if transfer else summary['metrics'] if run.startswith('domain') else rows(src/'metrics.csv')
for m in metricrows:
    k=(m['direction'],m['family']) if 'direction' in m else (int(m['seed']),m['family'])
    p=held_by[k];value=np.sqrt(loss(p,np.isfinite(p)))
    metric_error=max(metric_error,abs(value-float(m['rmse_dex'])))
assert metric_error<1e-9
spread=[]
for k in sorted(set(key(a) for a in attempts),key=str):
    a=[v for v in attempts if key(v)==k and v['success']]
    if a:
        values=[v['training_mse'] for v in a]
        spread.append(dict(group=list(k),successful_starts=len(a),min_loss=min(values),max_loss=max(values),relative_spread=(max(values)-min(values))/max(min(values),1e-30)))
out=dict(status='PASS_NUMERICAL_REPLAY_NOT_MECHANISM_VALIDATION',branch=branch,run=run,transfer=transfer,shared_audited_source_loader=True,independent_formula=True,galaxies=len(names),radii=len(y),historical_members=len(members),reserved_members=0,attempts=len(attempts),successful_attempts=sum(a['success'] for a in attempts),failed_attempts=[a for a in attempts if not a['success']],choices=len(choices),held_rows=len(rows(heldpath)),attempt_loss_max_abs=attempt_error,selected_loss_gap_max=selected_gap,held_prediction_max_abs=held_error,metric_rmse_max_abs=metric_error,start_spreads=spread,
    bindings={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'scripts/run_mond_atlas_clock_relay.py',ROOT/'configs/mond_atlas_clock_relay_v1.json',*src.iterdir()] if p.is_file()})
(HERE/f'{branch}-{run}{"-transfer" if transfer else ""}-replay.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ('bindings','start_spreads','failed_attempts')},indent=2))
