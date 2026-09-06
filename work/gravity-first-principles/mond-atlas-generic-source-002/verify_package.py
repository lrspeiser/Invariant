"""Read-only scoped verification; retains rather than suppresses failed gates."""
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent
ROOT=P.parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from mond_atlas_common import read_json,digest


def verify():
    manifest=read_json(P/'publication-manifest.json')
    for r in manifest['files']:
        p=ROOT/r['path']
        if p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:
            raise ValueError('changed publication file: '+r['path'])
    freeze=read_json(P/'freeze.json')
    for path,value in freeze['bindings'].items():
        if digest(ROOT/path)!=value:raise ValueError('changed frozen input: '+path)
    for path,value in read_json(ROOT/'work/gravity-first-principles/mond-atlas-generic-source-001/freeze.json')['bindings'].items():
        if digest(ROOT/path)!=value:raise ValueError('changed inherited freeze: '+path)
    run=read_json(P/'run-001/summary.json')
    for row in run['cases']:
        if digest(ROOT/row['packet'])!=row['packet_sha256']:raise ValueError('changed source packet')
    for path,value in read_json(P/'run-001/artifact-hashes.json').items():
        if digest(ROOT/path)!=value:raise ValueError('changed execution receipt')
    gate=read_json(P/'preconstruction-numerical-gate.json')
    if not gate['passed'] or gate['tests']!=9:raise ValueError('preconstruction test gate absent')
    actual=read_json(P/'preconstruction-actual-header-checks.json')
    if len(actual['checks'])!=30 or not all(r['passed'] for r in actual['checks']):raise ValueError('actual header gate absent')
    if not gate['completed_utc']<freeze['created_utc']<read_json(P/'run-001/execution-start.json')['started_utc']:
        raise ValueError('freeze chronology failed')
    final=read_json(P/'findings-002/summary.json')
    failed=read_json(P/'findings-002/distance-scaling-counterexample.json')
    if final['disposition']!='SOURCE_BLOCKED' or final['numerical_disposition']!='BENCHMARK_FAILED' or final['actual_distance_scaling_passed']:
        raise ValueError('failed scientific gate was suppressed')
    if failed['status']!='BENCHMARK_FAILED' or not (P/'findings-001/failure.json').exists():raise ValueError('failure receipt absent')
    if len(run['cases'])!=12 or len(run['mass_cases'])!=72 or run['observed_response_arrays_opened']!=0:
        raise ValueError('scope/case count changed')
    return dict(status='PACKAGE_INTEGRITY_VERIFIED_WITH_RETAINED_BENCHMARK_FAILURE',publication_files=len(manifest['files']),frozen_bindings=len(freeze['bindings']),private_packets=12,mass_rows=72,numerical_disposition='BENCHMARK_FAILED',admission_disposition='SOURCE_BLOCKED',observed_response_arrays_opened=0)


if __name__=='__main__':print(json.dumps(verify(),indent=2))
