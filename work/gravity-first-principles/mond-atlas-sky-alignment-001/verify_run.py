"""Independent sklearn pipeline/nested-selection replay, plus input binding audit."""
import sys,csv,json
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'))
from mond_atlas_common import read_json,digest
p=Path(__file__).parent/'run-001';cfg=read_json(ROOT/'configs/mond_atlas_sky_alignment_v1.json')
for name,expected in read_json(p/'bindings.json')['files'].items():assert digest(ROOT/name)==expected,name
def csvread(name):
    with (p/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
rows=csvread('sample.csv');saved=csvread('predictions.csv');choices=csvread('selected-penalties.csv')
features=cfg['features']+['log10_distance','ext_r'];labels=[r['feature'] for r in csvread('associations.csv')]
x=np.array([[float(r[k]) for k in features] for r in rows]);sky=np.array([[float(r[k]) for k in labels] for r in rows]);y=np.array([float(r['target']) for r in rows]);names=[r['galaxy'] for r in rows]
bundles={'baseline':[], 'quadrupole_axis':[0,1], 'octopole_axis':[2,3], 'galactic_latitude':[6,7], 'ecliptic_axis':[8], 'all_sky':list(range(10))}
maximum=0.;selections=0
for split in sorted(set(r['split'] for r in saved)):
    fold=np.array([int(next(r['fold'] for r in saved if r['galaxy']==n and r['split']==split)) for n in names])
    for bundle,cols in bundles.items():
        a=np.column_stack([x,sky[:,cols]])
        for held in sorted(set(fold)):
            train=fold!=held;test=~train;loss=[]
            for alpha in cfg['ridge_penalties']:
                errors=[]
                for inner in sorted(set(fold[train])):
                    fit=train&(fold!=inner);valid=train&(fold==inner)
                    model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
                    pred=model.fit(a[fit],y[fit]).predict(a[valid]);errors.extend((pred-y[valid])**2)
                loss.append(np.mean(errors))
            alpha=cfg['ridge_penalties'][int(np.argmin(loss))]
            chosen=next(r for r in choices if r['split']==split and r['bundle']==bundle and int(r['fold'])==held)
            assert alpha==float(chosen['alpha'])
            pred=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(a[train],y[train]).predict(a[test])
            expected=np.array([float(next(r['prediction'] for r in saved if r['galaxy']==n and r['split']==split and r['bundle']==bundle)) for n in np.array(names)[test]])
            err=float(np.max(abs(pred-expected)));maximum=max(maximum,err);assert err<1e-8;selections+=1
a=np.column_stack([np.ones(len(x)),x]);ry=y-a@np.linalg.lstsq(a,y,rcond=None)[0]
for i,r in enumerate(csvread('associations.csv')):
    rz=sky[:,i]-a@np.linalg.lstsq(a,sky[:,i],rcond=None)[0]
    rho=ry@rz/(np.linalg.norm(ry)*np.linalg.norm(rz));assert abs(rho-float(r['partial_r']))<1e-10
print(json.dumps(dict(status='PASS',nested_selections_verified=selections,prediction_rows_verified=len(saved),
                     partial_correlations_verified=10,maximum_prediction_difference=maximum),indent=2))
