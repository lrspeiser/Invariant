"""Separate saved-vector arithmetic audit for the fixed external-domain extension."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'work/gravity-first-principles/mond-atlas-external-boundary-001'


def run():
    out=P/'run001';summary=json.loads((out/'summary.json').read_text(encoding='utf-8'));bindings=json.loads((out/'bindings.json').read_text(encoding='utf-8'))
    for p,sha in bindings.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha
    rows=list(csv.DictReader((out/'vectors.csv').open(encoding='utf-8')));sets={}
    for row in rows:sets.setdefault((float(row['ell']),row['grid']),[]).append(row)
    vectors={k:np.array([[float(r[q]) for q in ['gx','gy','gz']] for r in rr]) for k,rr in sets.items()};heights=np.array([float(r['z']) for r in next(iter(sets.values()))[:-1]])
    error=0.;gates=[]
    for ref in summary['comparisons']:
        first,second=ref['comparison'].split('_to_');a=vectors[(ref['ell'],first)].copy();b=vectors[(ref['ell'],second)].copy()
        if ref['center_relative']:a=a-a[-1];b=b-b[-1]
        a=a[:-1];b=b[:-1]
        def metric(aa,bb):return float(np.sqrt(np.mean(np.sum((aa-bb)**2,axis=1)))/max(np.sqrt(np.mean(np.sum(bb*bb,axis=1))),1e-8))
        rms=metric(a,b);groups=[metric(a[heights==h],b[heights==h]) for h in [0,.2,.5,1]];passed=rms<.05 and max(groups)<.08
        error=max(error,abs(rms-ref['rms']),max(abs(v-g['relative']) for v,g in zip(groups,ref['groups'])));assert passed==ref['passed']
        gates.append(dict(ell=ref['ell'],comparison=ref['comparison'],center_relative=ref['center_relative'],rms=rms,groups=groups,passed=passed))
    effects=[]
    for ell in [.25,.5]:
        for grid in ['base12','base18','base24','fine18']:
            g=vectors[(ell,grid)];d=g[:-1]-g[-1];effects.append(dict(ell=ell,grid=grid,center_vector=g[-1].tolist(),center_relative_rms_per_unit_applied_field=float(np.sqrt(np.mean(np.sum(d*d,axis=1)))),height_rms=[float(np.sqrt(np.mean(np.sum(d[heights==h]**2,axis=1)))) for h in [0,.2,.5,1]]))
    assert error<1e-12
    result=dict(status='PASS_SAVED_VECTOR_REPLAY',all_binding_hashes_verified=True,vector_rows=len(rows),comparison_rows=len(gates),maximum_comparison_error=error,gates=gates,unit_response=effects,joint_24_fine_endpoint_validated=False)
    (P/'independent-vector-review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':run()
