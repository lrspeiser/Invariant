import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
import mond_atlas_external_program as external
from mond_atlas_selection_transfer import new_controls
assert new_controls()['passed']
controls=external.controls();logs=[]
for package,test in [('mond-atlas-noise-stationary-001','test_stationary.py'),('mond-atlas-noise-scale-channel-001','test_scale_channel.py')]:
    run=subprocess.run([sys.executable,'-m','unittest','discover','-s','work/gravity-first-principles/'+package,'-p',test,'-v'],cwd=ROOT,capture_output=True,text=True)
    logs.append(run.stdout+run.stderr);assert run.returncode==0,logs[-1]
receipt=json.loads((ROOT/'work/gravity-first-principles/mond-atlas-selection-transfer-001/completion-receipt.json').read_text(encoding='utf-8'))
for path,expected in receipt['files'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
(P/'parent-tests.log').write_text('\n'.join(logs),encoding='utf-8')
(P/'parent-verification.json').write_text(json.dumps(dict(status='PASS',selection_formula_controls=new_controls(),external_controls=controls,selection_completion_bindings_rehashed=len(receipt['files']),noise_test_suites=2,observed_targets_opened=False),indent=2)+'\n',encoding='utf-8')
print('Coordinator controls, noise tests and selection bindings passed')
