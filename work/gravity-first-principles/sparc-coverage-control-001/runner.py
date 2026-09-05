"""Check if broad gas-profile signal is absorbed by observational radial coverage."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
base=Path(__file__).parent;root=base/'Invariant'
parent=root/'work/gravity-first-principles/sparc-pattern-analysis-001'
robust=root/'work/gravity-first-principles/sparc-pattern-robustness-001'
dest=root/'work/gravity-first-principles/sparc-coverage-control-001';dest.mkdir(exist_ok=False)
data=json.loads((parent/'result.json').read_text()); registration=data['registration']; names=registration['names']
raw={g['name']:g for g in json.loads((root/'configs/sparc_rotation_curves_full_v1.json').read_text())['galaxies'] if g['name'] in names}
photo={g['galaxy']:g for g in json.loads((root/'configs/sparc_surface_brightness_exploration_v1.json').read_text())['galaxies']}
for path in [parent/'runner.py',robust/'runner.py']:
    t=ast.parse(path.read_text());exec(compile(ast.Module(body=[n for n in t.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'frozen_functions','exec'))
frozen=dict(parent_sha256=hashlib.sha256((parent/'result.json').read_bytes()).hexdigest(),
    nuisance_features='logarithmic position within observed radial span, its square, and logarithmic observed radial span',
    scope='Post-result coverage diagnostic only; observational extent is not admitted as a physical gravity source.')
(dest/'registration.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8');(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
rows=[]
for scenario,ml in registration['scenarios'].items():
    objects=build(ml)
    for mode in ['full','shape_oracle_centered']:
        for model,extra in [('local_plus_coverage',[]),('plus_broad_gas',[11,12])]:
            objs=[]
            for o in objects:
                rr=np.log(o['r']);span=rr[-1]-rr[0];coord=(rr-rr[0])/span
                X=np.column_stack([o['X'][:,:7],coord,coord**2,np.full(len(rr),span),o['X'][:,extra]])
                objs.append(dict(o,X=X,target=o['target']-(np.mean(o['target']) if mode!='full' else 0)))
            for salt in registration['salts']:
                rows.append(dict(scenario=scenario,mode=mode,model=model,salt=salt,**evaluate(objs,10+len(extra),salt)))
(dest/'result.json').write_text(json.dumps(dict(registration=frozen,rows=rows),indent=2),encoding='utf-8')
print(json.dumps(rows))
