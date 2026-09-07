"""Coordinator hash and saved-result checks; never modifies experiment outputs."""
import hashlib,json,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3];P=Path(__file__).resolve().parent;B=R/'work/gravity-first-principles'
checked=[]
for pkg in ['mond-atlas-external-joint-001','mond-atlas-noise-dc-regularization-001','mond-atlas-foreground-metadata-001','mond-atlas-normalization-001']:
    f=B/pkg/'run001/bindings.json';obj=json.loads(f.read_text());obj=obj.get('files',obj)
    for name,value in obj.items():
        if isinstance(value,str) and len(value)==64 and (R/name).is_file():
            assert hashlib.sha256((R/name).read_bytes()).hexdigest()==value,name;checked.append(name)
joint=json.loads((B/'mond-atlas-external-joint-001/run001/summary.json').read_text())
assert joint['joint_endpoint_gates_passed'] and all(c['passed'] for c in joint['comparisons'])
assert joint['sampled_peak_rss_bytes']<joint['working_cap_bytes'] and joint['seconds']<300
noise=json.loads((B/'mond-atlas-noise-dc-regularization-001/independent-review/receipt.json').read_text());assert noise['passed']
normal=json.loads((B/'mond-atlas-normalization-001/run001/summary.json').read_text());assert normal['observational_tests']==0 and len(normal['constant_environment_cases'])==36
proc=subprocess.run([sys.executable,str(B/'mond-atlas-noise-dc-regularization-001/test_dc.py')],cwd=R,capture_output=True,text=True);assert proc.returncode==0,proc.stderr
(P/'coordinator-verification.json').write_text(json.dumps(dict(bound_files_verified=checked,external_endpoint_passed=True,noise_independent_replay_passed=True,normalization_pairs=36,tests=dict(returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr),goal_complete=False),indent=2)+'\n',encoding='utf-8')
print('Coordinator verified',len(checked),'bindings and retained numerical outcomes')
