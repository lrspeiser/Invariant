"""Coordinator checks without rerunning or replacing frozen experiment outputs."""
import csv,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];P=Path(__file__).resolve().parent
SP=ROOT/'work/gravity-first-principles/mond-atlas-spatial-program-001'
with (SP/'fields.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==3456
fields={}
for row in rows:
    key=tuple(row[k] for k in ['case','level','r','theta','z'])
    fields.setdefault(key,{})[row['component']]=np.array([float(row[k]) for k in ['phi','gx','gy','gz']])
err=0
for key,values in fields.items():
    assert set(values)=={'total','stellar_luminosity','atomic_helium','co21'}
    error=np.max(abs(values['total']-sum(values[k] for k in ['stellar_luminosity','atomic_helium','co21'])))
    err=max(err,error/max(abs(values['total'])))
assert err<1e-12
config=json.loads((SP/'source-bindings.json').read_text(encoding='utf-8'));hashes={}
for case in config['source_cases']:
    for c in case['components']:hashes[c['path']]=c['sha256']
for path,h in hashes.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
receipt=json.loads((ROOT/'work/gravity-first-principles/mond-atlas-ngc3198-recovery-001/completion-receipt.json').read_text(encoding='utf-8'))
for path,h in receipt['files'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
logs=[]
for folder,pattern in [('tests','test_mond_atlas_registered_source.py'),('tests','test_mond_atlas_ngc3198_source_v2.py'),('work/gravity-first-principles/mond-atlas-noise-extension-001','test_extension.py'),('work/gravity-first-principles/mond-atlas-noise-joint-program-001','test_joint.py')]:
    p=subprocess.run([sys.executable,'-m','unittest','discover','-s',folder,'-p',pattern,'-v'],cwd=ROOT,capture_output=True,text=True)
    logs.append(p.stdout+p.stderr);assert p.returncode==0,logs[-1]
(P/'parent-tests.log').write_text('\n'.join(logs),encoding='utf-8')
(P/'parent-verification.json').write_text(json.dumps(dict(status='PASS',spatial_rows=len(rows),spatial_component_sums=len(fields),maximum_sum_relative_error=err,source_packets_rehashed=len(hashes),recovery_bindings_rehashed=len(receipt['files']),test_suites=4,unit_tests=19,scope='Coordinator sums, exact bindings and meaningful source/noise tests; independent physics and source replay receipts are in each branch.'),indent=2)+'\n',encoding='utf-8')
print('Coordinator verification passed')
