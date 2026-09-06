"""Read-only review of interrupted native-covariance delivery."""
import sys,json,csv
from pathlib import Path
import numpy as np
from scipy.stats import multivariate_normal
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'))
from mond_atlas_common import read_json,digest
from mond_atlas_native_covariance import fit_and_select_training
p=Path(__file__).parent/'run-001';c=read_json(ROOT/'configs/mond_atlas_native_covariance_v1.json')
bindings=read_json(p/'prospective-bindings.json')['bindings']; manifest=read_json(p/'run-manifest.json')['files']
for name,expected in {**bindings,**manifest}.items():
    assert digest(ROOT/name)==expected,name
for name,expected in read_json(p/'selection-frozen-before-east.json')['files'].items():
    assert digest(p/name)==expected,name
with (p/'block-geometry.csv').open(newline='') as f: geometry=list(csv.DictReader(f))
training_rows=[dict(r,fold=int(r['fold'])) for r in geometry if r['region']=='training']
with np.load(ROOT/'work/private/mond-atlas-native-covariance-001/run-001/background-blocks.npz') as z:
    training=z['training'];east=z['validation'];td=z['training_design'];ed=z['validation_design']
models,ranking,cv=fit_and_select_training(training,td,training_rows,c)
saved=read_json(p/'fitted-models.json'); oldrank=read_json(p/'training-selection.json')['ranking']
assert [r['model_id'] for r in ranking]==[r['model_id'] for r in oldrank]
maximum=0.
reports=read_json(p/'model-results.json')
for name,m in models.items():
    for k in ['beta','covariance']: np.testing.assert_allclose(m[k],saved[name][k],atol=1e-12,rtol=0)
    residual=east-ed[...,:len(m['beta'])]@m['beta']; flat=residual.reshape(-1,42)
    q=np.einsum('ni,ij,nj->n',flat,np.linalg.inv(m['covariance']),flat).mean()/42
    log=multivariate_normal.logpdf(flat,mean=np.zeros(42),cov=m['covariance']).mean()/42
    ref=reports[name]['validation']
    maximum=max(maximum,abs(q-ref['mean_q_over_n']),abs(log-ref['mean_logpdf_per_channel']))
    assert abs(q-ref['mean_q_over_n'])<1e-10 and abs(log-ref['mean_logpdf_per_channel'])<1e-10
print(json.dumps(dict(status='PASS',bound_and_manifest_files=len(set(bindings)|set(manifest)),
    training_models_refitted=8,training_ranking_matches=True,independent_eastern_scores_replayed=16,
    maximum_score_error=maximum,admitted_cube_likelihoods=0),indent=2))
