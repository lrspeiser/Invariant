"""Finalize bounded-descriptor predictions, metric comparisons, and report."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
base=Path(__file__).parent;root=base/'Invariant';outputs=base.parent/'outputs'
evidence=root/'work/gravity-first-principles'
parent=evidence/'sparc-pattern-analysis-001'
dest=evidence/'sparc-pattern-findings-001';dest.mkdir(exist_ok=False)
data=json.loads((parent/'result.json').read_text());registration=data['registration'];names=registration['names']
raw={g['name']:g for g in json.loads((root/'configs/sparc_rotation_curves_full_v1.json').read_text())['galaxies'] if g['name'] in names}
photo={g['galaxy']:g for g in json.loads((root/'configs/sparc_surface_brightness_exploration_v1.json').read_text())['galaxies']}
code=(parent/'runner.py').read_text().replace('fg=gas*abs(gas)/vb; fb=ml[1]*bulge**2/vb','denom=gas**2+ml[0]*disk**2+ml[1]*bulge**2; fg=gas**2/denom; fb=ml[1]*bulge**2/denom')
t=ast.parse(code);exec(compile(ast.Module(body=[n for n in t.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'bounded_functions','exec'))
objects=build([.5,.7]);original_raw=raw
raw=json.loads(json.dumps(raw))
for g in raw.values():
    for row in g['rows']: row[1]=str(float(row[1])+123.);row[2]=str(float(row[2])+17.)
poisoned=build([.5,.7]);raw=original_raw
assert all(np.array_equal(a['X'],b['X']) for a,b in zip(objects,poisoned))
objects=[dict(o,X=o['X'][:,list(range(7))+[11,12]]) for o in objects]
models=[];predictions=[]
for salt in registration['salts']:
    outer=foldmap(range(139),salt,5);pred=[None]*139;folds=[]
    for fold in range(5):
        train=[i for i in range(139) if outer[i]!=fold];test=[i for i in range(139) if outer[i]==fold]
        assert not set(train)&set(test)
        inner=foldmap(train,salt+f'-inner-{fold}',3);losses=[]
        for penalty in registration['ridge']:
            scores=[]
            for j in range(3):
                tr=[i for i in train if inner[i]!=j];va=[i for i in train if inner[i]==j]
                model=fit(objects,tr,9,penalty)
                scores.extend(np.mean((objects[i]['target']-predict(objects[i],model,9))**2) for i in va)
            losses.append(np.mean(scores))
        penalty=registration['ridge'][int(np.argmin(losses))];model=fit(objects,train,9,penalty)
        for i in test:pred[i]=predict(objects[i],model,9)
        folds.append(dict(fold=fold,penalty=penalty,physical_coefficients=(model[2][1:]/model[1]).tolist()))
    result=summary(objects,pred);models.append(dict(salt=salt,folds=folds,**result))
    for o,p in zip(objects,pred):
        predictions.append(dict(salt=salt,name=o['name'],r=o['r'].tolist(),vobs=o['v'].tolist(),
            error=o['error'].tolist(),rar=o['vrar'].tolist(),predicted=(o['vrar']*(1+p)).tolist(),
            gas_contrast_exterior=o['X'][:,-1].tolist()))
(dest/'predictions.json').write_text(json.dumps(predictions),encoding='utf-8')
hashes={name:hashlib.sha256((evidence/name/'result.json').read_bytes()).hexdigest() for name in [
    'sparc-pattern-analysis-001','sparc-pattern-robustness-001','sparc-profile-ablation-001','sparc-coverage-control-001','sparc-bounded-composition-001']}
result=dict(models=models,feature_poison_control_passed=True,input_hashes=hashes,scope='Post-selection development predictions; no independent validation or physical law admission.')
(dest/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps([{k:v for k,v in m.items() if k not in ['per_galaxy','folds']} for m in models]))
print('Exterior bounded coefficients',[[f['physical_coefficients'][-1] for f in m['folds']] for m in models])
