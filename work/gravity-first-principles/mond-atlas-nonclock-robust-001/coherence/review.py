"""Independent forward evaluator replays selected predictions and training losses."""
import sys,json,csv
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());sys.path.insert(0,str(ROOT/'scripts'))
from run_mond_atlas_clock_relay import load_sources
P=Path(__file__).resolve().parent

def evaluate(s,f,p):
    if f=='mond_fixed':f='mond_adjusted';p=[1,0]
    mf=p[0];v2=s['gas']*abs(s['gas'])+mf*(.5*s['disk']**2+.7*s['bulge']**2);sig=.5*mf*np.maximum(s['sb'],0)
    if f=='mond_adjusted':
        accel=v2/s['r'];a0=1.2e-10*3.085677581491367e13*np.exp(p[1]);v2=s['r']*.5*(accel+np.sqrt(accel**2+4*a0*accel))
    elif f.startswith('coherence'):
        n=p[3] if f.endswith('free') else 1;v2*=1+p[1]*np.exp(-np.log1p((sig/np.exp(p[2]))**n))
    elif f.startswith('relay'):
        transmission=np.exp(-p[1]*sig/100);v2*=transmission+p[2]*(1-transmission)
    return np.log10(np.sqrt(v2))

def main():
    config=json.loads((ROOT/'configs/mond_atlas_clock_relay_v1.json').read_text(encoding='utf-8'))
    s,y,errors,ids,rids,names,meta,exc,members=load_sources(config);gi=np.array([names.index(n) for n in ids]);lookup={(n,int(r)):i for i,(n,r) in enumerate(zip(ids,rids))}
    summary={};boundary={}
    for runname in ('run001','domain001'):
        run=P/runname;selections=json.loads((run/'selections.json').read_text(encoding='utf-8'));attempts=json.loads((run/'attempts.json').read_text(encoding='utf-8'))
        read=lambda f:list(csv.DictReader((run/f).open(encoding='utf-8',newline='')))
        if runname=='run001':
            folds={seed:{row['galaxy']:int(row['fold']) for row in read(f'folds-{seed}.csv')} for seed in config['fold_seeds']}
        rich=1.33*s['hi']/(1.33*s['hi']+.5*s['luminosity'])>=.5
        maxpred=maxtrain=0.;count=0
        for choice in selections:
            pred=evaluate(s,choice['evaluator'],choice['parameters'])
            if runname=='run001':mask=np.array([folds[choice['seed']][n]!=choice['fold'] for n in ids])
            else:mask=rich if choice['direction']=='gas_to_stellar' else ~rich
            if 'training_mse' in choice:
                value=np.mean([np.mean((pred[(gi==i)&mask]-y[(gi==i)&mask])**2) for i in range(len(names)) if np.any((gi==i)&mask)])
                maxtrain=max(maxtrain,abs(value-choice['training_mse']))
            rows=read('held-predictions.csv')
            for row in rows:
                if row['family']!=choice['family']:continue
                if runname=='run001':
                    if int(row['seed'])!=choice['seed'] or folds[choice['seed']][row['galaxy']]!=choice['fold']:continue
                elif row['direction']!=choice['direction']:continue
                j=lookup[(row['galaxy'],int(row['radial_index']))];maxpred=max(maxpred,abs(pred[j]-float(row['log10_predicted_speed'])));count+=1
            key=runname+'/'+choice['family'];boundary.setdefault(key,dict(selections=0,boundaries={},evaluators={}))
            d=boundary[key];d['selections']+=1
            for k in choice.get('boundary_parameters',[]):d['boundaries'][k]=d['boundaries'].get(k,0)+1
            e=choice['evaluator'];d['evaluators'][e]=d['evaluators'].get(e,0)+1
        summary[runname]=dict(held_rows=count,max_abs_logspeed=float(maxpred),max_abs_training_mse=float(maxtrain),passed=bool(max(maxpred,maxtrain)<1e-10),optimizer_starts=len(attempts),optimizer_failures=sum(not a['success'] for a in attempts))
    out=P/'independent-review-002';out.mkdir(exist_ok=False)
    (out/'receipt.json').write_text(json.dumps(dict(status='selected-forward-replay; shared source loader; optimizer is not independently refit',results=summary,boundary_rates=boundary),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
    if not all(v['passed'] for v in summary.values()):raise RuntimeError('Review failed')

if __name__=='__main__':main()
