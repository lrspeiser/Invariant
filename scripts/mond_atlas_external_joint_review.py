"""Saved-vector and resource replay independent of the large-grid solver."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'work/gravity-first-principles/mond-atlas-external-joint-001'


def run():
    out=P/'run001';summary=json.loads((out/'summary.json').read_text(encoding='utf-8'));bindings=json.loads((out/'bindings.json').read_text(encoding='utf-8'))
    for p,h in bindings.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    rows=list(csv.DictReader((out/'vectors.csv').open(encoding='utf-8')));sets={}
    for row in rows:sets.setdefault(row['grid'],[]).append(row)
    g={k:np.array([[float(r[v]) for v in ['gx','gy','gz']] for r in rr]) for k,rr in sets.items()};height=np.array([float(r['z']) for r in sets['fine24'][:-1]]);error=0.;replay=[]
    for ref in summary['comparisons']:
        old,_=ref['comparison'].split('_to_');a=g[old].copy();b=g['fine24'].copy()
        if ref['center_relative']:a-=a[-1];b-=b[-1]
        a=a[:-1];b=b[:-1]
        rms=lambda aa,bb:float(np.sqrt(np.mean(np.sum((aa-bb)**2,axis=1)))/max(np.sqrt(np.mean(np.sum(bb*bb,axis=1))),1e-8))
        value=rms(a,b);groups=[rms(a[height==h],b[height==h]) for h in [0,.2,.5,1]];passed=value<.05 and max(groups)<.08
        error=max(error,abs(value-ref['rms']),max(abs(v-r['relative']) for v,r in zip(groups,ref['groups'])));assert passed==ref['passed']
        replay.append(dict(comparison=ref['comparison'],center_relative=ref['center_relative'],rms=value,height_relative=groups,passed=passed))
    memory=json.loads((out/'memory-samples.json').read_text(encoding='utf-8'));peak=max(v[1] for v in memory['samples']);assert peak==memory['sampled_peak_rss_bytes'] and peak<16_000_000_000 and summary['seconds']<300 and error<1e-12
    d=g['fine24'][:-1]-g['fine24'][-1];effects=dict(center_vector=g['fine24'][-1].tolist(),center_relative_rms_per_unit_applied_field=float(np.sqrt(np.mean(np.sum(d*d,axis=1)))),height_rms=[float(np.sqrt(np.mean(np.sum(d[height==h]**2,axis=1)))) for h in [0,.2,.5,1]])
    result=dict(status='PASS_SAVED_VECTOR_AND_RESOURCE_REPLAY',vector_rows=len(rows),all_bound_hashes_verified=True,maximum_comparison_error=error,comparisons=replay,unit_response=effects,sampled_peak_rss_bytes=peak,memory_sample_count=len(memory['samples']),runtime_seconds=summary['seconds'],observational_validation=False)
    (P/'independent-review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':run()
