"""Real SPARC development residuals and nested whole-galaxy prediction tests."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

base=Path(__file__).parent; root=base/'Invariant'
dest=root/'work/gravity-first-principles/sparc-pattern-analysis-001'
dest.mkdir(exist_ok=False)
paths=['configs/sparc_rotation_curves_full_v1.json','configs/sparc_surface_brightness_exploration_v1.json',
       'runs/gravity/g4/conditional-formula-generator-v4.json','configs/gravity_g0_experiment.json']
inputs={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
history=json.loads((root/paths[2]).read_text())
names=sorted(g['galaxy'] for g in history['galaxies'])
raw={g['name']:g for g in json.loads((root/paths[0]).read_text())['galaxies'] if g['name'] in names}
photo={g['galaxy']:g for g in json.loads((root/paths[1]).read_text())['galaxies']}
assert len(raw)==len(names)==139
registration=dict(input_hashes=inputs,names=names,
    scope='Previously exposed 139-galaxy development data only. New nested CV does not restore independent confirmation.',
    scenarios={'nominal':[.5,.7],'lighter_stars':[.35,.5],'heavier_stars':[.65,.9]},
    targets='fractional velocity residual Vobs/VRAR-1; equal total training weight per galaxy.',
    model_groups=['offset','acceleration','local_structure','nonlocal_structure'],
    folds='Five outer whole-galaxy folds, three inner whole-galaxy folds; SHA256 name ordering, two fixed salts.',
    salts=['sparc-pattern-round1-A','sparc-pattern-round1-B'],ridge=[.01,.1,1.,10.,100.],
    limits='Fixed mass-to-light sensitivity only; no complete distance/inclination/thickness/molecular-gas/covariance marginalization. No motion source or full 3D field law.',
    radial_statistics='Equal-count inner and outer thirds; a within-galaxy radial contrast, not a fixed physical boundary.',
    feature_definitions=['log10(gbar/a0)','its square','log10(r/r_disk_peak)','signed gas fraction','bulge fraction',
        'log10(1+SBdisk+SBbul)','gradient log Vbar^2 versus log r',
        'interior and exterior averages of SB/(SB+100) minus local SB/(SB+100), logarithmic reaches .25 and 1',
        'interior and exterior signed-gas-fraction averages minus local, logarithmic reach 1'])
(dest/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())

def build(ml):
    objects=[]
    for name in names:
        a=np.array(raw[name]['rows'],float); sb=np.array(photo[name]['rows'],float)
        r,v,e,gas,disk,bulge=a.T
        vb=gas*abs(gas)+ml[0]*disk**2+ml[1]*bulge**2
        if np.any(vb<=0):
            raise ValueError(f'Nonpositive baryonic source: {name}, {ml}')
        y=vb/r/3702.81458; vrar=np.sqrt(vb/(-np.expm1(-np.sqrt(y))))
        target=v/vrar-1; log_r=np.log(r); log_y=np.log10(y)
        fg=gas*abs(gas)/vb; fb=ml[1]*bulge**2/vb
        brightness=np.log10(1+sb.sum(axis=1)); q=sb.sum(axis=1)/(sb.sum(axis=1)+100)
        peak=r[np.argmax(disk**2)]; rr=np.log10(r/peak)
        slope=np.gradient(np.log(vb),log_r)
        local=np.column_stack([log_y,log_y**2,rr,fg,fb,brightness,slope])
        cell=np.empty(len(r)); cell[1:-1]=(log_r[2:]-log_r[:-2])/2
        cell[0]=(log_r[1]-log_r[0])/2; cell[-1]=(log_r[-1]-log_r[-2])/2
        dx=log_r[None,:]-log_r[:,None]
        extra=[]
        for reach,source in [(.25,q),(1.,q),(1.,fg)]:
            for side in [-1,1]:
                weight=np.exp(-abs(dx)/reach)*(dx*side>=0)*cell[None,:]
                extra.append(weight@source/weight.sum(axis=1)-source)
        X=np.column_stack([local,*extra])
        m=max(1,len(r)//3)
        objects.append(dict(name=name,X=X,target=target,vrar=vrar,v=v,error=e,r=r,
            median_residual=float(np.median(target)),outer_minus_inner=float(np.median(target[-m:])-np.median(target[:m])),
            brightness=float(np.median(brightness)),gas=float(np.median(fg)),bulge=float(np.median(fb)),
            acceleration=float(np.median(log_y)),radial_span=float(log_r[-1]-log_r[0]),
            source_distance=float(raw[name]['distance_mpc'])))
    return objects

def foldmap(selected,salt,k):
    ordered=sorted(selected,key=lambda i:hashlib.sha256((salt+'|'+names[i]).encode()).hexdigest())
    return {i:j%k for j,i in enumerate(ordered)}

def fit(objects,indices,columns,penalty):
    X=np.concatenate([objects[i]['X'][:,:columns] for i in indices])
    y=np.concatenate([objects[i]['target'] for i in indices])
    w=np.concatenate([np.full(len(objects[i]['target']),1/len(objects[i]['target'])) for i in indices]);w/=w.sum()
    mean=w@X; scale=np.sqrt(w@((X-mean)**2));scale=np.where(scale<1e-10,1.,scale)
    Z=np.column_stack([np.ones(len(X)),(X-mean)/scale])
    regular=np.eye(Z.shape[1])*penalty/len(indices);regular[0,0]=0
    coef=np.linalg.solve(Z.T@(w[:,None]*Z)+regular,Z.T@(w*y))
    return mean,scale,coef

def predict(obj,model,columns):
    mean,scale,coef=model
    return coef[0]+((obj['X'][:,:columns]-mean)/scale)@coef[1:]

def summary(objects,pred):
    before=[];after=[];chi0=chi=km0=km=0.;negative=0
    for obj,p in zip(objects,pred):
        before.append(float(np.mean(obj['target']**2)))
        after.append(float(np.mean((obj['target']-p)**2)))
        predicted=obj['vrar']*(1+p)
        negative+=int(np.sum(predicted<=0))
        chi0+=float(np.sum(((obj['v']-obj['vrar'])/obj['error'])**2))
        chi+=float(np.sum(((obj['v']-predicted)/obj['error'])**2))
        km0+=float(np.mean((obj['v']-obj['vrar'])**2))/len(objects)
        km+=float(np.mean((obj['v']-predicted)**2))/len(objects)
    return dict(equal_galaxy_fractional_mse_gain=1-np.mean(after)/np.mean(before),
        galaxies_improved=int(np.sum(np.array(after)<before)),median_galaxy_gain=float(np.median(1-np.array(after)/before)),
        pooled_chi_square=chi,rar_chi_square=chi0,pooled_chi_square_gain=1-chi/chi0,
        equal_galaxy_kms_mse_gain=1-km/km0,nonpositive_predictions=negative,
        per_galaxy=[dict(name=o['name'],before=b,after=a) for o,b,a in zip(objects,before,after)])

all_results=[]; descriptive=[]; record_rows=[]
for scenario,ml in registration['scenarios'].items():
    objects=build(ml)
    if scenario=='nominal':
        assert abs(sum(np.sum(((o['v']-o['vrar'])/o['error'])**2) for o in objects)-130714.6893155)<.01
        assert sum(len(o['v']) for o in objects)==2720
    for target in ['median_residual','outer_minus_inner']:
        for feature in ['brightness','gas','bulge','acceleration','radial_span','source_distance']:
            stat=spearmanr([o[feature] for o in objects],[o[target] for o in objects])
            descriptive.append(dict(scenario=scenario,target=target,feature=feature,rho=float(stat.statistic),
                unadjusted_p=float(stat.pvalue),scope='Descriptive multiple comparisons, not discovery significance.'))
    for o in objects:
        record_rows.append(dict(scenario=scenario,**{k:v for k,v in o.items() if k not in ['X','target','vrar','v','error','r']}))
    for salt in registration['salts']:
        outer=foldmap(range(139),salt,5)
        for label,columns in [('offset',0),('acceleration',2),('local_structure',7),('nonlocal_structure',13)]:
            predictions=[None]*139; folds=[]
            for fold in range(5):
                train=[i for i in range(139) if outer[i]!=fold]; test=[i for i in range(139) if outer[i]==fold]
                inner=foldmap(train,salt+f'-inner-{fold}',3); losses=[]
                for penalty in registration['ridge']:
                    loss=[]
                    for inner_fold in range(3):
                        tr=[i for i in train if inner[i]!=inner_fold]; va=[i for i in train if inner[i]==inner_fold]
                        model=fit(objects,tr,columns,penalty)
                        loss.extend(np.mean((objects[i]['target']-predict(objects[i],model,columns))**2) for i in va)
                    losses.append(float(np.mean(loss)))
                penalty=registration['ridge'][int(np.argmin(losses))]
                model=fit(objects,train,columns,penalty)
                for i in test:predictions[i]=predict(objects[i],model,columns)
                folds.append(dict(fold=fold,penalty=penalty,inner_losses=losses,test_names=[names[i] for i in test],
                    coefficients=model[2].tolist(),training_mean=model[0].tolist(),training_scale=model[1].tolist()))
            result=summary(objects,predictions)
            all_results.append(dict(scenario=scenario,salt=salt,model=label,folds=folds,**result))
            if scenario=='nominal':
                (dest/f'predictions_{salt[-1]}_{label}.json').write_text(json.dumps([
                    dict(name=o['name'],radius=o['r'].tolist(),observed=o['v'].tolist(),rar=o['vrar'].tolist(),
                         prediction=(o['vrar']*(1+p)).tolist()) for o,p in zip(objects,predictions)]),encoding='utf-8')
        print(f'Completed {scenario}, {salt}',flush=True)
out=dict(registration=registration,descriptive_correlations=descriptive,models=all_results,
    source_summary=record_rows,new_physical_laws_admitted=0)
(dest/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
with (dest/'galaxy_summary.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(record_rows[0]));writer.writeheader();writer.writerows(record_rows)
print(json.dumps([dict(scenario=r['scenario'],salt=r['salt'][-1],model=r['model'],gain=r['equal_galaxy_fractional_mse_gain'],
    improved=r['galaxies_improved'],chi_gain=r['pooled_chi_square_gain']) for r in all_results]))
