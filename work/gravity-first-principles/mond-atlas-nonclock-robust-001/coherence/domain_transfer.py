import sys,json
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());sys.path.insert(0,str(ROOT/'scripts'))
from run_mond_atlas_coherence_robust import fit,predict,SPECS,FAMILIES
from run_mond_atlas_clock_relay import load_sources
from mond_atlas_common import read_json,write_json,write_csv,digest
P=Path(__file__).resolve().parent

def main():
    out=P/'domain001';out.mkdir(exist_ok=False)
    write_json(out/'pre-access-bindings.json',dict(status='POST_HOC_DOMAIN_TRANSFER',files={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),P/'DOMAIN_PREFLIGHT.md',ROOT/'scripts/run_mond_atlas_coherence_robust.py']}))
    s,y,errors,ids,radial_ids,names,meta,exclusions,members=load_sources(read_json(ROOT/'configs/mond_atlas_clock_relay_v1.json'))
    gi=np.array([names.index(n) for n in ids]);rich=1.33*s['hi']/(1.33*s['hi']+.5*s['luminosity'])>=.5
    attempts=[];choices=[];metrics=[];held=[];strata=[]
    for direction,train in [('gas_to_stellar',rich),('stellar_to_gas',~rich)]:
        selected={}
        for f in SPECS:
            best,tries=fit(s,y,gi,train,f);attempts.extend(dict(direction=direction,family=f,**a) for a in tries)
            if best is None:write_json(out/'failure.json',attempts);raise RuntimeError('Fit failure')
            selected[f]=dict(evaluator=f,**best)
        for f in FAMILIES:
            if f=='mond_fixed':choice=dict(evaluator=f,parameters=[])
            else:
                choice=selected[f]
                if f.startswith(('coherence','relay')) and selected['newton']['training_mse']<=choice['training_mse']:choice=selected['newton']
            choices.append(dict(direction=direction,family=f,**choice))
            pred=predict(s,choice['evaluator'],choice['parameters']);res=pred-y
            vals=[res[(gi==i)&~train] for i in range(len(names)) if np.any((gi==i)&~train)]
            metrics.append(dict(direction=direction,family=f,donor_galaxies=len(np.unique(gi[train])),recipient_galaxies=len(vals),rmse_dex=float(np.sqrt(np.mean([np.mean(v*v) for v in vals])))))
            for j in np.where(~train)[0]:held.append(dict(direction=direction,family=f,galaxy=ids[j],radial_index=int(radial_ids[j]),log10_predicted_speed=float(pred[j]),residual_dex=float(res[j])))
            ratio=s['r']/s['rd']
            for region,reg in [('inner',ratio<1),('middle',(ratio>=1)&(ratio<3)),('outer',ratio>=3)]:
                mask=reg&~train;v=[res[(gi==i)&mask] for i in range(len(names)) if np.any((gi==i)&mask)]
                strata.append(dict(direction=direction,family=f,region=region,galaxies=len(v),radii=int(mask.sum()),signed_mean_dex=float(np.mean([a.mean() for a in v])),rmse_dex=float(np.sqrt(np.mean([np.mean(a*a) for a in v])))))
    write_json(out/'attempts.json',attempts);write_json(out/'selections.json',choices)
    write_json(out/'summary.json',dict(status='POST_HOC_DOMAIN_TRANSFER',metrics=metrics,optimizer_starts=len(attempts),optimizer_failures=sum(not a['success'] for a in attempts)))
    write_csv(out/'held-predictions.csv',held);write_csv(out/'strata.csv',strata)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
