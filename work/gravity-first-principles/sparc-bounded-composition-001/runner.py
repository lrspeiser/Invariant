"""Post-result descriptor repair: preserve signed gravity, bound composition magnitudes."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
base=Path(__file__).parent;root=base/'Invariant'
parent=root/'work/gravity-first-principles/sparc-pattern-analysis-001';robust=root/'work/gravity-first-principles/sparc-pattern-robustness-001'
dest=root/'work/gravity-first-principles/sparc-bounded-composition-001';dest.mkdir(exist_ok=False)
data=json.loads((parent/'result.json').read_text());registration=data['registration'];names=registration['names']
raw={g['name']:g for g in json.loads((root/'configs/sparc_rotation_curves_full_v1.json').read_text())['galaxies'] if g['name'] in names}
photo={g['galaxy']:g for g in json.loads((root/'configs/sparc_surface_brightness_exploration_v1.json').read_text())['galaxies']}
for path in [parent/'runner.py',robust/'runner.py']:
    t=ast.parse(path.read_text());exec(compile(ast.Module(body=[n for n in t.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'original_functions','exec'))
original_build=build
code=(parent/'runner.py').read_text()
old='fg=gas*abs(gas)/vb; fb=ml[1]*bulge**2/vb'
new='denom=gas**2+ml[0]*disk**2+ml[1]*bulge**2; fg=gas**2/denom; fb=ml[1]*bulge**2/denom'
assert code.count(old)==1
tree=ast.parse(code.replace(old,new))
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build'],type_ignores=[]),'bounded_build','exec'))
frozen=dict(parent_sha256=hashlib.sha256((parent/'result.json').read_bytes()).hexdigest(),
    reason='Signed gas-to-net-acceleration ratio becomes large near component cancellation; DDO064 maximum absolute ratio 18.87 in lighter-stellar case.',
    changed_definition=new,unchanged='Signed Vbar^2, RAR predictions, observed velocities, all galaxies, fold salts, ridge grid, local acceleration and slope.',
    scope='Data-motivated successor; bounded component-magnitude descriptors are not measured mass fractions. All earlier results retained.')
(dest/'registration.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8');(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
rows=[];checks=[]
for scenario,ml in registration['scenarios'].items():
    objects=build(ml);before=original_build(ml)
    assert all(np.array_equal(a['vrar'],b['vrar']) for a,b in zip(objects,before))
    assert all(np.all((o['X'][:,3]>=0)&(o['X'][:,3]<=1)) for o in objects)
    checks.append(dict(scenario=scenario,maximum_original_abs_gas_ratio=max(float(np.max(abs(o['X'][:,3]))) for o in before),
        maximum_bounded_gas_descriptor=max(float(np.max(o['X'][:,3])) for o in objects),rar_unchanged=True))
    for mode in ['full','shape_oracle_centered']:
        for model,extra in [('local',[]),('plus_broad_gas',[11,12]),('plus_all_nonlocal',list(range(7,13)))]:
            selected=list(range(7))+extra
            objs=[dict(o,X=o['X'][:,selected],target=o['target']-(np.mean(o['target']) if mode!='full' else 0)) for o in objects]
            for salt in registration['salts']:
                rows.append(dict(scenario=scenario,mode=mode,model=model,salt=salt,**evaluate(objs,len(selected),salt)))
    print('Completed',scenario,flush=True)
(dest/'result.json').write_text(json.dumps(dict(registration=frozen,rows=rows,checks=checks),indent=2),encoding='utf-8')
print(json.dumps(rows))
