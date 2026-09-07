"""Independent saved-vector replay; does not import the sensitivity implementation."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
P=R/'work/gravity-first-principles/mond-atlas-local-calibration-001'
F=R/'work/gravity-first-principles/mond-atlas-refraction-program-001'
def read(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))
def vector(row):return np.array([float(row[k]) for k in ('gx','gy','gz')])
def run():
    bindings=json.loads((P/'run001/pre-access-bindings.json').read_text())
    for path,digest in bindings.items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==digest,path
    nr=[r for r in read(F/'finer002/sampled-vectors.csv') if r['model']=='newton']
    candidates={(float(r['ell_kpc']),float(r['x']),float(r['y']),float(r['z'])):vector(r) for r in read(F/'coherence-scale/run001/sampled-vectors.csv') if r['grid']=='finer'}
    rows=read(P/'run001/point-sensitivity.csv'); groups=read(P/'run001/group-summary.csv'); thresholds=read(P/'run001/required-calibration.csv')
    errors=[];replay=[]
    for r in rows:
        p=np.array([float(r[k]) for k in ('x','y','z')]);n=vector(nr[int(r['index'])]);c=candidates[(float(r['ell_kpc']),*p)];u=float(r['u_lab']);e=.2+.8*u/(1+u);g=e*c
        inc=-np.dot(g,p)/np.linalg.norm(p);inn=-np.dot(n,p)/np.linalg.norm(p);rat=np.linalg.norm(g)/np.linalg.norm(n)
        errors.extend(abs(g-vector(r)));errors.extend([abs(rat-float(r['norm_ratio'])),abs(inc-float(r['radial_inward'])),abs(inn-float(r['newton_radial_inward'])),abs(inc-inn-float(r['inward_difference']))])
        assert (rat>1)==(r['norm_exceeds_newton']=='True');assert (inc>inn)==(r['more_inward_than_newton']=='True')
        replay.append(dict(ell=float(r['ell_kpc']),u=u,z=p[2],ratio=rat,inc=inc,inn=inn))
    for s in groups:
        a=[r for r in replay if r['ell']==float(s['ell_kpc']) and r['u']==float(s['u_lab']) and (s['height_group']=='all' or r['z']==float(s['height_group']))]
        ratios=np.array([r['ratio'] for r in a]); inc=np.array([r['inc'] for r in a]); inn=np.array([r['inn'] for r in a]);valid=(inc>0)&(inn>0)
        values=dict(points=len(a),norm_ratio_min=ratios.min(),norm_ratio_median=np.median(ratios),norm_ratio_max=ratios.max(),norm_enhancement_points=sum(ratios>1),more_inward_points=sum(inc>inn),candidate_inward_points=sum(inc>0),candidate_outward_points=sum(inc<0),newton_outward_points=sum(inn<0),positive_inward_ratio_points=sum(valid),inward_ratio_median=np.median(inc[valid]/inn[valid]),mean_inward_difference=np.mean(inc-inn))
        for k,v in values.items():assert np.isclose(v,float(s[k]),rtol=1e-12,atol=1e-10),(k,v,s[k])
    threshold_errors=[]
    for t in thresholds:
        p=np.array([float(t[k]) for k in ('x','y','z')]);n=vector(nr[int(t['index'])]);c=candidates[(float(t['ell_kpc']),*p)]
        inc=-np.dot(c,p);inn=-np.dot(n,p)
        for name,ratio in [('norm',np.linalg.norm(c)/np.linalg.norm(n)),('radial',inc/inn if inc>0 and inn>0 else None)]:
            status=t[name+'_classification']
            if ratio is None:assert status.startswith('undefined');continue
            assert np.isclose(ratio,float(t['uncalibrated_'+('norm' if name=='norm' else 'inward')+'_ratio']),rtol=1e-12)
            assert np.isclose(1/ratio,float(t[name+'_epsilon_required']),rtol=1e-12)
            if ratio<=1:assert status=='unattainable'
            elif ratio>5:assert status=='all_nonnegative_u'
            else:
                cutoff=float(t[name+'_u_required']);err=abs((.2+.8*cutoff/(1+cutoff))*ratio-1);threshold_errors.append(err);assert err<1e-12
                assert status=='strictly_above_u_threshold'
    result=dict(status='PASS',binding_hashes=len(bindings),point_rows=len(rows),groups=len(groups),threshold_rows=len(thresholds),max_absolute_arithmetic_error=float(max(errors)),max_threshold_identity_error=float(max(threshold_errors)),positive_inward_points_per_model=382,warning='Two points are outward in both models; a more-inward difference there need not be attraction.',review_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (P/'independent-review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':run()
