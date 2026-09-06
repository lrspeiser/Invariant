"""Parent review of frozen pressure/light deliverables; no observed data reads."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'tests'))


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, data):
    with p.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(data,f,indent=2,allow_nan=False)
        f.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label',default='parent-001')
    args=parser.parse_args()
    if not args.label.replace('-','').isalnum():
        raise ValueError('simple output label required')
    output=HERE/args.label
    output.mkdir(exist_ok=False)
    pressure=ROOT/'work/gravity-first-principles/mond-atlas-pressure-support-001'
    light=ROOT/'work/gravity-first-principles/mond-atlas-light-projection-001'
    manifests={}
    files={}
    pm=read(pressure/'manifest.json')
    for group in ('public','private'):
        for name,row in pm[group].items():
            files[name]=row['sha256']
    lm=read(light/'delivery-manifest.json')
    files.update(lm['files'])
    for name,expected in files.items():
        if digest(ROOT/name)!=expected:
            raise ValueError('delivery hash mismatch: '+name)
    manifests['pressure']=dict(public_entries=len(pm['public']),private_entries=len(pm['private']),
        manifest_sha256=digest(pressure/'manifest.json'))
    manifests['light']=dict(entries=len(lm['files']),
        manifest_sha256=digest(light/'delivery-manifest.json'))
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name)
        for name in ('test_mond_atlas_pressure_support','test_mond_atlas_light_projection'))
    with (output/'unit-tests.log').open('w',encoding='utf-8') as stream:
        with threadpool_limits(limits=1):
            tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    if not tests.wasSuccessful() or tests.skipped:
        raise ValueError('parent unit tests failed')
    pressure_out='verification-execution018-'+args.label+'.json'
    for name,command in [
        ('pressure-replay',[sys.executable,'-B',str(pressure/'verify_replay.py'),'--run-id','run-002','--output-name',pressure_out]),
        ('light-receipts',[sys.executable,'-B',str(light/'verify_receipts.py')])]:
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=120)
        (output/(name+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8',newline='\n')
        if result.returncode:
            raise ValueError(name+' failed')
    import run_mond_atlas_light_projection as runner
    checks=runner.Checks()
    with threadpool_limits(limits=1):
        derived=runner.run_benchmarks(read(ROOT/'configs/mond_atlas_light_projection_v1.json'),checks)
    old={r['id']:r for r in read(light/'run-001/checks.json')}
    for row in checks.rows:
        if row!=old[row['id']]:
            raise ValueError('light numerical replay differs: '+row['id'])
    if any(r['required'] and not r['passed'] for r in checks.rows):
        raise ValueError('light required numerical gate failed')
    if derived!=read(light/'run-001/summary.json')['derived_synthetic_results']:
        raise ValueError('light image/magnification replay differs')
    pressure_replay=read(pressure/pressure_out)
    write(output/'verification.json',dict(status='PASS',tests=tests.testsRun,
        failures=len(tests.failures),errors=len(tests.errors),skipped=len(tests.skipped),
        delivery_files_rehashed=len(files),manifests=manifests,
        pressure_replay=pressure_replay,
        light_numerical_check_records_replayed_exactly=len(checks.rows),
        light_required_numerical_checks=sum(r['required'] for r in checks.rows),
        light_diagnostic_failures_retained=sum(not r['passed'] for r in checks.rows if not r['required']),
        light_derived_image_and_magnification_results_exact=True,
        script_sha256=digest(Path(__file__)),observational_source_or_response_arrays_opened=0,
        pressure_and_light_disposition='THEORY_BENCHMARK_ONLY',goal_complete=False))
    print(json.dumps({'status':'PASS','tests':tests.testsRun,'delivery_files_rehashed':len(files),
        'light_numerical_records_exact':len(checks.rows),'pressure_predictions_exact':pressure_replay['prediction_arrays_exact']}))


if __name__=='__main__':
    main()
