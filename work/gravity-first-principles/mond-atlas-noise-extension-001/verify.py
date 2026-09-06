"""Independent aggregate/covariance and inverse-score audit of immutable run001."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,csv,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'AGENTS.md').exists());RUN=P/'run001'

def readcsv(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def aggregate(a,side):
    return np.array([[a[b,y:y+side,x:x+side].mean(axis=(0,1)) for y in range(0,24,side) for x in range(0,24,side)] for b in range(len(a))])

def shrink(a,alpha):
    raw=np.einsum('ni,nj->ij',a,a)/len(a);d=np.diag(raw).copy();floor=max(1e-12,1e-8*np.median(d));raw+=np.diag(np.maximum(d,floor)-d)
    return (1-alpha)*raw+alpha*np.diag(np.diag(raw))

def main():
    cfg=json.loads((ROOT/'configs/mond_atlas_aperture_noise_v1.json').read_text(encoding='utf-8'))
    source=ROOT/cfg['input']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==cfg['input_sha256']
    fitted=json.loads((RUN/'models-before-east.json').read_text(encoding='utf-8'))
    with np.load(source) as z:west=z['training'];east=z['validation']
    mean=west.mean(axis=(0,1,2));res=west-mean;pixel=shrink(res.reshape(-1,42),.1);corr=pixel/np.sqrt(np.outer(pixel.diagonal(),pixel.diagonal()))
    selected_matrix_error=0.;means_error=0.;predictions={}
    for side_str,item in fitted['models'].items():
        side=int(side_str);name=fitted['chosen'][side_str];sample=aggregate(res,side).reshape(-1,42)
        if name.startswith('full'):cov=shrink(sample,float(name.split('_')[1]))
        else:
            samplecov=shrink(sample,.1);target=corr*np.sqrt(np.outer(samplecov.diagonal(),samplecov.diagonal()));weight=float(name.split('_')[-1]);cov=weight*samplecov+(1-weight)*target
        selected_matrix_error=max(selected_matrix_error,float(np.max(abs(cov-np.array(item['covariances'][name])))))
        means_error=max(means_error,float(np.max(abs(mean-np.array(item['mean'])))))
        e=aggregate(east-mean,side)
        for model,c in item['covariances'].items():
            c=np.array(c);inv=np.linalg.inv(c);logdet=np.linalg.slogdet(c)[1]
            q=np.einsum('bti,ij,btj->bt',e,inv,e).mean(axis=1)/42
            lp=-.5*(42*q+logdet+42*np.log(2*np.pi))/42
            trace=np.sum(e*e,axis=-1).mean(axis=1)
            predictions[(side,model)]=(q,lp,trace)
    geometry=readcsv(ROOT/'work/gravity-first-principles/mond-atlas-native-covariance-001/run-001/block-geometry.csv')
    eastrows=[r for r in geometry if r['region']=='validation'];index={r['block_id']:i for i,r in enumerate(eastrows)}
    error=0.;score_rows=readcsv(RUN/'all-east-core-scores.csv')
    for row in score_rows:
        q,lp,tr=predictions[(int(row['side']),row['model'])];i=index[row['core']]
        error=max(error,abs(q[i]-float(row['q_over_n'])),abs(lp[i]-float(row['logpdf_per_channel'])),abs(tr[i]-float(row['trace_second_moment'])))
    cv=readcsv(RUN/'western-cv.csv');selection_mismatch=0
    for side,selected in fitted['chosen'].items():
        names=list(fitted['models'][side]['covariances']);values={name:np.mean([float(r['logpdf']) for r in cv if r['side']==side and r['model']==name]) for name in names}
        winner=max(names,key=lambda n:values[n]);selection_mismatch+=winner!=selected
    supportfile=ROOT/'work/private/mond-atlas-native-covariance-001/run-001/geometry-supports.npz'
    support_hash=hashlib.sha256(supportfile.read_bytes()).hexdigest()
    support_pass=True
    with np.load(supportfile) as packet:
        for row in geometry:
            y0,y1,x0,x1=[int(row[k]) for k in ('y0','y1','x0','x1')]
            support_pass=bool(support_pass and packet[row['region']][y0:y1,x0:x1].all())
        disjoint=not bool(np.any(packet['training']&packet['validation']))
    receipt=dict(independent_loop_aggregation=True,selected_covariances_verified=6,all_east_core_scores=len(score_rows),selected_covariance_max_abs=selected_matrix_error,mean_max_abs=means_error,score_max_abs=float(error),selection_mismatches=selection_mismatch,inherited_support_contains_every_core=support_pass,inherited_training_validation_support_disjoint=disjoint,support_file_sha256=support_hash,support_cleans_noise_or_emission_admission=False,passed=bool(max(selected_matrix_error,means_error,error)<1e-10 and selection_mismatch==0 and support_pass and disjoint))
    out=P/'independent-review';out.mkdir(exist_ok=False);(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt,indent=2))
    if not receipt['passed']:raise RuntimeError('Independent verification failed')

if __name__=='__main__':main()
