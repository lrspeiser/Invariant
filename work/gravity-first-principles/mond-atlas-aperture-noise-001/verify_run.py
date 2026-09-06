"""Independent explicit-loop aggregation and inverse scoring of all saved scales."""
import sys,csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'))
from mond_atlas_common import read_json,digest
p=Path(__file__).parent/'run-001';c=read_json(ROOT/'configs/mond_atlas_aperture_noise_v1.json')
for name,expected in read_json(p/'bindings.json')['files'].items(): assert digest(ROOT/name)==expected,name
with np.load(ROOT/c['input']) as d: west=d['training'];east=d['validation']
saved=read_json(p/'models-before-east.json');mean=west.mean(axis=(0,1,2))
np.testing.assert_allclose(mean,saved['mean'],rtol=0,atol=1e-14)
with (p/'core-scores.csv').open(newline='') as f: rows=list(csv.DictReader(f))
maximum=0.; models={}
def loop(data,side):
    return np.array([[data[b,y:y+side,x:x+side].mean(axis=(0,1)) for y in range(0,24,side) for x in range(0,24,side)] for b in range(len(data))])
for side in c['sides_native_pixels']:
    tr=loop(west-mean,side).reshape(-1,42); raw=tr.T@tr/len(tr)
    diag=np.diag(raw); floor=max(1e-12,1e-8*np.median(diag));raw+=np.diag(np.maximum(diag,floor)-diag)
    covariance=.9*raw+.1*np.diag(np.diag(raw));models[side]=covariance
    np.testing.assert_allclose(covariance,saved['covariances'][str(side)],rtol=0,atol=1e-12)
    data=loop(east-mean,side)
    for name,cov in [('independent_pixels',models[1]/side**2),('empirical_aperture',covariance)]:
        q=np.einsum('bni,ij,bnj->bn',data,np.linalg.inv(cov),data)
        log=-.5*(q+np.linalg.slogdet(cov)[1]+42*np.log(2*np.pi))
        for b in range(len(east)):
            row=next(r for r in rows if int(r['side_pixels'])==side and r['model']==name and int(r['core'])==b)
            for k,v in [('q_over_n',q[b].mean()/42),('logpdf_per_channel',log[b].mean()/42),('trace_second_moment',np.sum(data[b]**2,axis=-1).mean())]:
                err=abs(v-float(row[k]));maximum=max(maximum,err);assert err<1e-10
print(json.dumps(dict(status='PASS',independent_aggregation=True,covariances_replayed=6,core_score_rows_replayed=len(rows),maximum_error=maximum),indent=2))
