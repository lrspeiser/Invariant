"""One bounded source-preserving refraction refinement; no stored large fields."""
import csv,json,time,traceback
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from threadpoolctl import threadpool_limits
import mond_atlas_refraction_program as m
P=m.P

def plummer():
    records=[]
    points=np.array([[.5,.2,.3],[1.,.3,.5],[1.5,.4,.8]])
    expected=-points/(1+(points*points).sum(axis=1))[:,None]**1.5
    for n in [33,65,129]:
        axis=np.linspace(-4,4,n);x,y,z=np.meshgrid(axis,axis,axis,indexing='ij');r2=x*x+y*y+z*z
        phi=-1/np.sqrt(1+r2);rhs=3/(1+r2)**2.5
        solved,check=m.solve(rhs,np.ones_like(rhs),[8/(n-1)]*3,phi)
        force=-np.stack(np.gradient(solved,8/(n-1),edge_order=2),axis=-1)
        sampled=RegularGridInterpolator([axis]*3,force)(points)
        error=float(np.linalg.norm(sampled-expected)/np.linalg.norm(expected))
        records.append(dict(n=n,force_relative_rms=error,**check))
    assert records[-1]['force_relative_rms']<.01 and records[-1]['force_relative_rms']<records[0]['force_relative_rms']
    return records

def run():
    out=P/'finer002';out.mkdir(exist_ok=False)
    m.save(out/'pre-access-bindings.json',{p.relative_to(m.ROOT).as_posix():m.digest(p) for p in [Path(__file__),m.ROOT/'scripts/mond_atlas_refraction_program.py',m.ROOT/'scripts/mond_atlas_source_resolution.py',P/'REFINEMENT_ADDENDUM.md',P/'PLUMMER_CONTROL_REFINEMENT.md',P/'run001/summary.json',m.MANIFEST]})
    try:
        m.save(out/'plummer-control.json',plummer())
        start=time.perf_counter();deadline=start+180
        originalcg=m.cg
        def timedcg(*args,**kwargs):
            previous=kwargs['callback']
            def callback(x):
                previous(x)
                if time.perf_counter()>deadline:raise TimeoutError('Declared180second numerical budget reached')
            kwargs['callback']=callback;return originalcg(*args,**kwargs)
        m.cg=timedcg
        manifest=json.loads(m.MANIFEST.read_text(encoding='utf-8'));case=next(c for c in manifest['source_cases'] if c['id']=='f4-stars-h0p4')
        spacing=[.0625,.0625,.03125];axes=[np.arange(-128,129)*h for h in spacing]
        rho,mass=m.source(case,axes);fullmass=sum(c['full_source_mass'] for c in mass)
        rr=np.sqrt(axes[0][:,None,None]**2+axes[1][None,:,None]**2+axes[2][None,None,:]**2);rr=np.maximum(rr,1e-30)
        old=list(csv.DictReader((P/'run001/sampled-vectors.csv').open(encoding='utf-8')))
        records=[];rows=[]
        for model in ['newton','refraction']:
            selected=[r for r in old if r['case']==case['id'] and r['grid']=='fine' and r['model']==model]
            points=np.array([[float(r[k]) for k in ['x','y','z']] for r in selected]);reference=np.array([[float(r[k]) for k in ['gx','gy','gz']] for r in selected])
            eps=np.ones_like(rho) if model=='newton' else .2+.8*rho/(rho+1e7)
            bc=-m.G*fullmass/rr/(1 if model=='newton' else .2)
            phi,check=m.solve(4*np.pi*m.G*rho,eps,spacing,bc)
            # Interpolate one gradient component at a time to limit peak RAM.
            vectors=[]
            for axis,h in enumerate(spacing):
                component=-np.gradient(phi,h,axis=axis,edge_order=2)
                vectors.append(RegularGridInterpolator(axes,component)(points))
            sampled=np.array(vectors).T
            rms=float(np.linalg.norm(sampled-reference)/np.linalg.norm(sampled));groups=[]
            for z in [0,.2,.5,1.]:
                mask=points[:,2]==z;groups.append(dict(height=z,relative_rms=float(np.linalg.norm(sampled[mask]-reference[mask])/np.linalg.norm(sampled[mask]))))
            records.append(dict(model=model,shape=list(rho.shape),mass=mass,elapsed_seconds=time.perf_counter()-start,vector_relative_rms_vs_fine=rms,groups=groups,field_convergence_passed=rms<.05 and all(g['relative_rms']<.08 for g in groups),**check))
            rows.extend(dict(model=model,x=float(p[0]),y=float(p[1]),z=float(p[2]),gx=float(v[0]),gy=float(v[1]),gz=float(v[2])) for p,v in zip(points,sampled))
            m.save(out/'progress.json',records);print(model,records[-1],flush=True)
            if not check['passed']:raise RuntimeError('PDE gate failed')
            del phi,eps,bc,component
        with (out/'sampled-vectors.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        m.save(out/'summary.json',dict(status='SOURCE_PRESERVING_NUMERICAL_REFINEMENT',records=records,all_field_gates_passed=all(r['field_convergence_passed'] for r in records),observed_response_arrays_opened=0,new_private_bytes=0))
    except Exception:m.save(out/'failure.json',dict(traceback=traceback.format_exc(),retain_failed_attempt=True));raise

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
