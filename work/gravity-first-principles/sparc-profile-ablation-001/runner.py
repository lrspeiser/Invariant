"""Post-result localization of the surrounding-profile development signal."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
base=Path(__file__).parent; root=base/'Invariant'
parent=root/'work/gravity-first-principles/sparc-pattern-analysis-001'
robust=root/'work/gravity-first-principles/sparc-pattern-robustness-001'
dest=root/'work/gravity-first-principles/sparc-profile-ablation-001'
dest.mkdir(exist_ok=False)
original=json.loads((parent/'result.json').read_text()); registration=original['registration']
names=registration['names']
raw={g['name']:g for g in json.loads((root/'configs/sparc_rotation_curves_full_v1.json').read_text())['galaxies'] if g['name'] in names}
photo={g['galaxy']:g for g in json.loads((root/'configs/sparc_surface_brightness_exploration_v1.json').read_text())['galaxies']}
for path in [parent/'runner.py',robust/'runner.py']:
    tree=ast.parse(path.read_text())
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'frozen_analysis_functions','exec'))
groups={'near_brightness':[7,8],'broad_brightness':[9,10],'broad_gas':[11,12],
        'broad_brightness_and_gas':[9,10,11,12],'interior_only':[7,9,11],'exterior_only':[8,10,12]}
frozen=dict(groups=groups,scope='Post-result feature ablation on exposed development sample; not new validation.',
    parent_sha256=hashlib.sha256((parent/'result.json').read_bytes()).hexdigest())
(dest/'registration.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
objects=build([.5,.7]); rows=[]
for mode in ['full_residual','shape_only_oracle_centered']:
    for label,extra in groups.items():
        selected=list(range(7))+extra
        objs=[dict(o,X=o['X'][:,selected],target=o['target']-(np.mean(o['target']) if mode!='full_residual' else 0)) for o in objects]
        for salt in registration['salts']:
            rows.append(dict(mode=mode,group=label,salt=salt,**evaluate(objs,len(selected),salt)))
(dest/'result.json').write_text(json.dumps(dict(registration=frozen,rows=rows),indent=2),encoding='utf-8')
print(json.dumps(rows))
