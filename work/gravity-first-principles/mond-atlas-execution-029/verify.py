"""Read-only coordinator verification of completed results and input bindings."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[3];P=Path(__file__).resolve().parent;B=R/'work/gravity-first-principles'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
checked=[]
for pkg,run in [('mond-atlas-local-calibration-001','run001'),('mond-atlas-local-gas-calibration-001','run001'),('mond-atlas-noise-reference-001','run001'),('mond-atlas-noise-transfer-001','run001'),('mond-atlas-static-susceptibility-001','run001'),('mond-atlas-log-susceptibility-001','run002')]:
    binding_name='pre-access-bindings.json' if pkg=='mond-atlas-local-calibration-001' else 'bindings.json'
    obj=read(B/pkg/run/binding_name);files=obj.get('files',obj)
    for name,value in files.items():
        if isinstance(value,str) and len(value)==64 and (R/name).is_file():
            assert hashlib.sha256((R/name).read_bytes()).hexdigest()==value,name;checked.append(name)
    completion=B/pkg/'completion-receipt.json'
    if completion.exists():
        for name,value in read(completion).get('files',{}).items():
            assert hashlib.sha256((R/name).read_bytes()).hexdigest()==value,name;checked.append(name)
receipts={}
for pkg,path in [('mond-atlas-local-calibration-001','independent-review.json'),('mond-atlas-local-gas-calibration-001','independent-review/results.json'),('mond-atlas-noise-reference-001','independent-review.json'),('mond-atlas-noise-transfer-001','independent-review/receipt.json'),('mond-atlas-static-susceptibility-001','independent-review/receipt.json'),('mond-atlas-log-susceptibility-001','independent-radial-review/receipt.json'),('mond-atlas-log-susceptibility-001','independent-analytic-review/receipt.json')]:
    obj=read(B/pkg/path);receipts[pkg+'/'+path]=obj.get('status',obj)
for pkg,run in [('mond-atlas-static-susceptibility-001','run001'),('mond-atlas-log-susceptibility-001','run002')]:
    obj=read(B/pkg/run/'results.json');assert obj['integrations']==16 and obj['new_observed_scores']==0
    for row in obj['results']:
        assert row['conservation_pass']
        assert row.get('exact_circular_pass') is not False
        assert row.get('trajectory_pass') is not False
noise=read(B/'mond-atlas-noise-transfer-001/run001/summary.json');assert noise['galaxy']=='NGC3198' and noise['observed_gravity_scores']==0 and noise['channels']==72
assert noise['mode_flag_count']==576 and any(g['q']>1.2 for g in noise['diagnostics'])
ref=read(B/'mond-atlas-noise-reference-001/run001/summary.json');assert ref['bonferroni_rejections']==0 and not ref['likelihood_admitted']
prior=read(B/'mond-atlas-execution-028/publication-manifest.json')
for e in prior['files']:assert hashlib.sha256((R/e['path']).read_bytes()).hexdigest()==e['sha256'],e['path']
out=dict(status='COORDINATOR_VERIFIED_WITH_RETAINED_LIMITS',prior_files=len(prior['files']),bound_files=len(checked),bindings_checked=checked,independent_receipts=receipts,cartesian_integrations=32,independent_radial_cases=16,source_likelihood_admitted=False,goal_complete=False)
(P/'coordinator-verification.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k not in ['bindings_checked','independent_receipts']}))
