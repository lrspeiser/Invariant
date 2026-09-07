"""Sensitivity to manufactured local G calibration, self-generated fields only."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import mond_atlas_refraction_program as m
ROOT=m.ROOT;P=ROOT/'work/gravity-first-principles/mond-atlas-local-calibration-001';REF=ROOT/'work/gravity-first-principles/mond-atlas-refraction-program-001'
U=[0,.01,.1,1,10,100]


def epsilon(u):return (u+.2)/(u+1.)
def write_csv(p,rows):
    with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)


def threshold(ratio):
    if ratio is None:return dict(epsilon_required=None,u_required=None,classification='undefined_nonpositive_inward_reference_or_candidate')
    value=1./ratio
    if ratio<=1:return dict(epsilon_required=value,u_required=None,classification='unattainable')
    if ratio>5:return dict(epsilon_required=value,u_required=None,classification='all_nonnegative_u')
    return dict(epsilon_required=value,u_required=(value-.2)/(1-value),classification='strictly_above_u_threshold')


def controls():
    axis=np.linspace(-1,1,9);x,y,z=np.meshgrid(axis,axis,axis,indexing='ij');eps=1+.2*x;rhs=6+1.6*x;boundary=x*x+y*y+z*z
    reference,check=m.solve(rhs,eps,[.25]*3,boundary);assert check['passed'];results=[]
    for u in U:
        scale=epsilon(u);phi,c=m.solve(scale*rhs,eps,[.25]*3,scale*boundary);error=float(np.max(abs(phi-scale*reference)));assert c['passed'] and error<1e-9
        assert abs(scale/scale-1)<1e-12
        results.append(dict(u=u,epsilon=scale,linear_potential_error=error,same_medium_measured_newton_ratio=scale/scale))
    # Closed-form thresholds verified without actual source fields.
    for ratio in [.5,1.,1.1,2.,4.,5.,6.]:
        t=threshold(ratio)
        if t['u_required'] is not None:assert abs(epsilon(t['u_required'])*ratio-1)<1e-12
    return results


def run():
    out=P/'run001';out.mkdir(exist_ok=False)
    smooth=REF/'coherence-scale/run001';newton=REF/'finer002';normal=ROOT/'work/gravity-first-principles/mond-atlas-normalization-001'
    paths=[Path(__file__),P/'PREFLIGHT.md',smooth/'sampled-vectors.csv',smooth/'summary.json',smooth/'pre-access-bindings.json',newton/'sampled-vectors.csv',newton/'summary.json',normal/'run001/summary.json',normal/'README.md',m.MANIFEST,ROOT/'scripts/mond_atlas_refraction_program.py',ROOT/'scripts/mond_atlas_refraction_program_coherence.py']
    bindings={p.relative_to(ROOT).as_posix():m.digest(p) for p in paths}
    manifest=json.loads(m.MANIFEST.read_text(encoding='utf-8'));case=next(c for c in manifest['source_cases'] if c['id']=='f4-stars-h0p4')
    for c in case['components']:assert m.digest(ROOT/c['path'])==c['sha256'];bindings[c['path']]=c['sha256']
    m.save(out/'pre-access-bindings.json',bindings);m.save(out/'pre-field-controls.json',controls())
    sd=json.loads((smooth/'summary.json').read_text(encoding='utf-8'));nd=json.loads((newton/'summary.json').read_text(encoding='utf-8'))
    assert sd['all_gates_passed'] and next(r for r in nd['records'] if r['model']=='newton')['field_convergence_passed']
    with (smooth/'sampled-vectors.csv').open(encoding='utf-8') as f:sr=list(csv.DictReader(f))
    with (newton/'sampled-vectors.csv').open(encoding='utf-8') as f:nr=[r for r in csv.DictReader(f) if r['model']=='newton']
    points=np.array([[float(r[k]) for k in ['x','y','z']] for r in nr]);N=np.array([[float(r[k]) for k in ['gx','gy','gz']] for r in nr]);assert len(N)==384
    radius=np.linalg.norm(points,axis=1);inward=-points/radius[:,None];nn=np.linalg.norm(N,axis=1);nrcomponent=np.sum(N*inward,axis=1);height=points[:,2]
    pointrows=[];summaries=[];thresholds=[];inherited=[]
    for ell in [.25,.5]:
        rr=[r for r in sr if float(r['ell_kpc'])==ell and r['grid']=='finer'];assert np.array_equal(points,[[float(r[k]) for k in ['x','y','z']] for r in rr]);C=np.array([[float(r[k]) for k in ['gx','gy','gz']] for r in rr]);cn=np.linalg.norm(C,axis=1);cr=np.sum(C*inward,axis=1)
        normratio=cn/nn;valid=(cr>0)&(nrcomponent>0);radialratio=np.divide(cr,nrcomponent,out=np.full_like(cr,np.nan),where=valid)
        for j,p in enumerate(points):
            nt=threshold(float(normratio[j]));rt=threshold(float(radialratio[j]) if valid[j] else None)
            thresholds.append(dict(ell_kpc=ell,index=j,x=float(p[0]),y=float(p[1]),z=float(p[2]),uncalibrated_norm_ratio=float(normratio[j]),uncalibrated_inward_ratio=float(radialratio[j]) if valid[j] else None,**{'norm_'+k:v for k,v in nt.items()},**{'radial_'+k:v for k,v in rt.items()}))
        for u in U:
            elab=epsilon(u);g=elab*C;norms=np.linalg.norm(g,axis=1);radials=np.sum(g*inward,axis=1);delta=radials-nrcomponent;ratio=norms/nn
            for j,p in enumerate(points):pointrows.append(dict(ell_kpc=ell,u_lab=u,epsilon_lab=elab,index=j,x=float(p[0]),y=float(p[1]),z=float(p[2]),gx=float(g[j,0]),gy=float(g[j,1]),gz=float(g[j,2]),norm_ratio=float(ratio[j]),radial_inward=float(radials[j]),newton_radial_inward=float(nrcomponent[j]),inward_difference=float(delta[j]),norm_exceeds_newton=bool(ratio[j]>1),more_inward_than_newton=bool(delta[j]>0)))
            for label,mask in [('all',np.ones(len(points),bool))]+[(str(z),height==z) for z in [0,.2,.5,1]]:
                vm=mask&valid
                summaries.append(dict(ell_kpc=ell,u_lab=u,epsilon_lab=elab,height_group=label,points=int(mask.sum()),norm_ratio_min=float(ratio[mask].min()),norm_ratio_median=float(np.median(ratio[mask])),norm_ratio_max=float(ratio[mask].max()),norm_enhancement_points=int(np.sum(ratio[mask]>1)),more_inward_points=int(np.sum(delta[mask]>0)),candidate_inward_points=int(np.sum(radials[mask]>0)),candidate_outward_points=int(np.sum(radials[mask]<0)),newton_outward_points=int(np.sum(nrcomponent[mask]<0)),positive_inward_ratio_points=int(vm.sum()),inward_ratio_median=float(np.median(elab*radialratio[vm])) if vm.any() else None,mean_inward_difference=float(np.mean(delta[mask]))))
        inherited.extend([r for r in sd['comparisons'] if r['ell_kpc']==ell])
    write_csv(out/'point-sensitivity.csv',pointrows);write_csv(out/'group-summary.csv',summaries);write_csv(out/'required-calibration.csv',thresholds)
    m.save(out/'summary.json',dict(status='CONDITIONAL_LOCAL_CALIBRATION_SENSITIVITY',new_real_source_solves=0,scenarios=U,points_per_source=384,point_scenarios=len(pointrows),threshold_rows=len(thresholds),group_rows=len(summaries),inherited_source_gates=inherited,observed_lab_environment_used=False,observed_response_scored=False,imposed_external_field_scaled=False))
    print(json.dumps([r for r in summaries if r['height_group']=='all'],indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
