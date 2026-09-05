"""Follow-up: normalization versus shape, paired bootstrap, and alignment controls."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
base=Path(__file__).parent; root=base/'Invariant'
parent=root/'work/gravity-first-principles/sparc-pattern-analysis-001'
dest=root/'work/gravity-first-principles/sparc-pattern-robustness-001'
dest.mkdir(exist_ok=False)
original=json.loads((parent/'result.json').read_text())
registration=dict(original['registration'])
registration['followup']=dict(parent_sha256=hashlib.sha256((parent/'result.json').read_bytes()).hexdigest(),
    scope='Post-result development robustness audit, not independent confirmation.',
    bootstrap='2000 paired galaxy bootstrap draws of fixed out-of-fold losses; does not repeat model selection or account for historical search.',
    shape='Subtract observed mean fractional residual per galaxy to isolate shape. Diagnostic only: the test-galaxy normalization is not predicted.',
    shuffle='12 seeded permutations of the six nonlocal feature columns together within each galaxy; local columns unchanged. Tests radial alignment, not every source correlation.')
(dest/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
names=registration['names']
raw={g['name']:g for g in json.loads((root/'configs/sparc_rotation_curves_full_v1.json').read_text())['galaxies'] if g['name'] in names}
photo={g['galaxy']:g for g in json.loads((root/'configs/sparc_surface_brightness_exploration_v1.json').read_text())['galaxies']}
tree=ast.parse((parent/'runner.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'parent_functions','exec'))
objects=build([.5,.7]);rng=np.random.default_rng(20260905)
total=np.mean([np.mean(o['target']**2) for o in objects])
offset=np.mean([np.mean(o['target'])**2 for o in objects])
variance=dict(total_fractional_mse=float(total),mean_offset_component=float(offset),
    fraction_in_galaxy_mean_offsets=float(offset/total),within_curve_component=float(total-offset))
bootstrap=[]
for salt in registration['salts']:
    subset=[r for r in original['models'] if r['scenario']=='nominal' and r['salt']==salt]
    local=next(r for r in subset if r['model']=='local_structure')
    nonlocal_=next(r for r in subset if r['model']=='nonlocal_structure')
    b=np.array([r['before'] for r in local['per_galaxy']]); l=np.array([r['after'] for r in local['per_galaxy']]); nl=np.array([r['after'] for r in nonlocal_['per_galaxy']])
    index=rng.integers(0,len(b),(2000,len(b)))
    gain=1-nl[index].mean(axis=1)/b[index].mean(axis=1)
    extra=1-nl[index].mean(axis=1)/l[index].mean(axis=1)
    bootstrap.append(dict(salt=salt,rar_gain_interval=np.quantile(gain,[.025,.5,.975]).tolist(),
        gain_beyond_local_interval=np.quantile(extra,[.025,.5,.975]).tolist()))

def evaluate(objs,columns,salt):
    outer=foldmap(range(139),salt,5); predictions=[None]*139
    for fold in range(5):
        train=[i for i in range(139) if outer[i]!=fold]; test=[i for i in range(139) if outer[i]==fold]
        inner=foldmap(train,salt+f'-inner-{fold}',3); losses=[]
        for penalty in registration['ridge']:
            loss=[]
            for j in range(3):
                tr=[i for i in train if inner[i]!=j]; va=[i for i in train if inner[i]==j]
                model=fit(objs,tr,columns,penalty)
                loss.extend(np.mean((objs[i]['target']-predict(objs[i],model,columns))**2) for i in va)
            losses.append(np.mean(loss))
        model=fit(objs,train,columns,registration['ridge'][int(np.argmin(losses))])
        for i in test: predictions[i]=predict(objs[i],model,columns)
    before=np.array([np.mean(o['target']**2) for o in objs])
    after=np.array([np.mean((o['target']-p)**2) for o,p in zip(objs,predictions)])
    return dict(gain=float(1-after.mean()/before.mean()),improved=int(np.sum(after<before)))

shape=[]
centered=[dict(o,target=o['target']-np.mean(o['target'])) for o in objects]
for salt in registration['salts']:
    for label,columns in [('acceleration',2),('local_structure',7),('nonlocal_structure',13)]:
        shape.append(dict(salt=salt,model=label,**evaluate(centered,columns,salt)))
shuffle=[]
for trial in range(12):
    objs=[]
    for o in objects:
        X=o['X'].copy(); X[:,7:]=X[rng.permutation(len(X)),7:]
        objs.append(dict(o,X=X))
    for salt in registration['salts']:
        shuffle.append(dict(trial=trial,salt=salt,**evaluate(objs,13,salt)))
print('Completed alignment controls',flush=True)
out=dict(registration=registration,variance_decomposition=variance,paired_bootstrap=bootstrap,
    centered_shape=shape,shuffled_nonlocal=shuffle,admitted_laws=0)
(dest/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(dict(variance=variance,bootstrap=bootstrap,shape=shape)))
for salt in registration['salts']:
    gains=[r['gain'] for r in shuffle if r['salt']==salt]
    actual=next(r['equal_galaxy_fractional_mse_gain'] for r in original['models'] if r['scenario']=='nominal' and r['salt']==salt and r['model']=='nonlocal_structure')
    print(salt,actual,min(gains),float(np.median(gains)),max(gains),'shuffles_ge_actual',sum(g>=actual for g in gains))
