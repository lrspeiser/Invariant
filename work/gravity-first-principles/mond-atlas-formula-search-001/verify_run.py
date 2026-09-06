"""Read-only numerical replay; print receipt for a separately preserved output."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from mond_atlas_common import digest, read_json
from mond_atlas_formula_search import replay, nested
from mond_atlas_pattern_learning import galaxy_folds

p=Path(__file__).parent/'run-001'
binding=read_json(p/'bindings.json')
for name, expected in binding['bindings'].items():
    if digest(ROOT/name)!=expected: raise ValueError('Binding mismatch: '+name)
config=binding['config']
with (ROOT/'work/gravity-first-principles/mond-atlas-pattern-learning-001/sample.csv').open(newline='') as f:
    sample=list(csv.DictReader(f))
x=np.array([[float(r[k]) for k in config['features']] for r in sample]); y=np.array([float(r['target']) for r in sample])
names=[r['galaxy'] for r in sample]
with (p/'predictions.csv').open(newline='') as f: rows=list(csv.DictReader(f))
records=read_json(p/'selections.json'); maximum=0.
for record in records:
    fold=galaxy_folds(names,record['seed'],config['fold_count']); selected=fold==record['fold']
    for model in ('adaptive','baseline'):
        values=replay(x[selected],record['formulas'][model])
        expected=np.array([float(next(r[model] for r in rows if r['galaxy']==n and int(r['seed'])==record['seed'])) for n in np.array(names)[selected]])
        maximum=max(maximum,float(np.max(abs(values-expected))))
        assert np.allclose(values,expected,atol=1e-8,rtol=0)
seed=config['fold_seeds'][0]; folds=galaxy_folds(names,seed,config['fold_count'])
cpu, cpu_records=nested(x,y,folds,config,np)
for model in cpu:
    expected=np.array([float(next(r[model] for r in rows if r['galaxy']==n and int(r['seed'])==seed)) for n in names])
    maximum=max(maximum,float(np.max(abs(cpu[model]-expected))))
    assert np.allclose(cpu[model],expected,atol=1e-8,rtol=0)
for c,g in zip(cpu_records,[r for r in records if r['seed']==seed]):
    for model in ('adaptive','baseline'):
        assert c['formulas'][model]['columns']==g['formulas'][model]['columns']
        assert c['formulas'][model]['alpha']==g['formulas'][model]['alpha']
for i in range(config['null_replicates']):
    q=read_json(p/f'shuffle-{i:02d}.json'); sx=x.copy(); sx[:,4:]=q['shuffled_structure']
    for r in q['selection']:
        for model in ('adaptive','baseline'):
            actual=replay(sx[folds==r['fold']],r['formulas'][model])
            expected=np.array(q['predictions'][model])[folds==r['fold']]
            assert np.allclose(actual,expected,atol=1e-8,rtol=0)
    b=np.mean((np.array(q['predictions']['baseline'])-y)**2)
    e=np.mean((np.array(q['predictions']['adaptive'])-y)**2)
    assert abs(q['metric']['mse_gain_percent']-100*(b-e)/b)<1e-10
print(json.dumps(dict(status='PASS',bound_files=len(binding['bindings']),observed_formulas_replayed=30,
                     shuffled_formulas_replayed=config['null_replicates']*10,full_first_seed_cpu_selection_matches=True,
                     maximum_prediction_difference=maximum),indent=2))
