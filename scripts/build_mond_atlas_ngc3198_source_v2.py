"""Fresh immutable NGC3198 source-only replay with frozen integer annuli fix."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import json,shutil,subprocess,sys,traceback
from pathlib import Path
from threadpoolctl import threadpool_limits
from mond_atlas_common import ROOT,read_json,write_json,digest
import build_mond_atlas_registered_source as legacy
import build_mond_atlas_ngc3198_source_checked as checked
from mond_atlas_ngc3198_source_v2 import rebin_tracer

P=ROOT/'work/gravity-first-principles/mond-atlas-ngc3198-recovery-001'
OLD=ROOT/'work/gravity-first-principles/mond-atlas-generic-source-002'
CORRECTION=OLD/'correction-preflight-001'
CONFIG=CORRECTION/'config-ngc3198-source-v2.json'
PRIVATE=ROOT/'work/private/mond-atlas-ngc3198-recovery-001'

def run():
    if (P/'run-001').exists() or PRIVATE.exists():raise FileExistsError('Immutable outputs exist')
    receipt=P/'attempt-002';receipt.mkdir(exist_ok=False)
    if shutil.disk_usage(ROOT).free<10*1024**3:raise RuntimeError('Need8GB reserve plus2GB budget')
    for path in [OLD/'freeze.json',CORRECTION/'freeze.json',legacy.REPORT/'freeze.json']:
        checked.verify_bindings(read_json(path)['bindings'])
    config=read_json(CONFIG);bindings={}
    for path in [Path(__file__),ROOT/'scripts/mond_atlas_ngc3198_source_v2.py',ROOT/'tests/test_mond_atlas_ngc3198_source_v2.py',P/'PREFLIGHT.md',CONFIG,CORRECTION/'PROTOCOL.md',CORRECTION/'freeze.json']:
        checked.bind(bindings,path)
    write_json(receipt/'pre-access-bindings.json',dict(disposition='SOURCE_BLOCKED',source_arrays_opened=0,observed_response_arrays_opened=0,bindings=bindings))
    try:
        logs=[]
        for pattern in ['test_mond_atlas_registered_source.py','test_mond_atlas_ngc3198_source_v2.py']:
            result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',pattern,'-v'],cwd=ROOT,capture_output=True,text=True)
            logs.append(result.stdout+result.stderr)
            if result.returncode:
                (receipt/'preconstruction-tests.log').write_text('\n'.join(logs),encoding='utf-8');raise RuntimeError('Pre-source tests failed')
        (receipt/'preconstruction-tests.log').write_text('\n'.join(logs),encoding='utf-8')
        headers,ph,transfers,audit=checked.audit_headers(config,bindings)
        write_json(receipt/'header-audit.json',audit)
        checks=checked.geometry_checks(config,headers,ph,transfers)
        write_json(receipt/'actual-header-checks.json',checks)
        if not all(c['passed'] for c in checks):raise RuntimeError('Actual-header geometry gate failed')
        write_json(receipt/'all-preconstruction-bindings.json',bindings)
        legacy.rebin_tracer=rebin_tracer
        with threadpool_limits(limits=1):legacy.execute(CONFIG,P/'run-001',PRIVATE/'run-001')
        used=sum(p.stat().st_size for p in PRIVATE.rglob('*') if p.is_file())
        if used>2*1024**3 or shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('Storage budget violation')
        write_json(P/'storage.json',dict(new_private_bytes=used,free_bytes=shutil.disk_usage(ROOT).free,limit_bytes=2*1024**3,observed_response_arrays_opened=0))
    except Exception:
        write_json(receipt/'failure.json',dict(traceback=traceback.format_exc(),preserve_all_outputs=True));raise

if __name__=='__main__':run()
