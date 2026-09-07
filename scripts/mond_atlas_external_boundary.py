"""Larger domains, unchanged conditional normal-field physics, bounded CPU job."""
import csv,json,time,traceback
from pathlib import Path
import numpy as np
import psutil
from scipy.ndimage import gaussian_filter
from threadpoolctl import threadpool_limits
import mond_atlas_external_program as external
m=external.m
P=m.ROOT/'work/gravity-first-principles/mond-atlas-external-boundary-001'
OLD=external.P/'run001'


def run():
    out=P/'run001';out.mkdir(exist_ok=False);start=time.perf_counter();records=[];rows=[];samples={}
    try:
        bindings={p.relative_to(m.ROOT).as_posix():m.digest(p) for p in [Path(__file__),P/'PREFLIGHT.md',OLD/'vectors.csv',OLD/'summary.json',OLD/'bindings.json',m.MANIFEST,m.ROOT/'scripts/mond_atlas_external_program.py',m.ROOT/'scripts/mond_atlas_refraction_program.py',m.ROOT/'scripts/mond_atlas_source_resolution.py',external.P/'analytic-review/field-review-operator.json']}
        for path,sha in json.loads((OLD/'bindings.json').read_text(encoding='utf-8')).items():assert m.digest(m.ROOT/path)==sha
        operator=json.loads((external.P/'analytic-review/field-review-operator.json').read_text(encoding='utf-8'));assert operator['status']=='PASS' and operator['independent_sparse_max_potential_error']<1e-9
        manifest=json.loads(m.MANIFEST.read_text(encoding='utf-8'));case=next(c for c in manifest['source_cases'] if c['id']=='f4-stars-h0p4')
        for component in case['components']:
            path=m.ROOT/component['path'];assert m.digest(path)==component['sha256'];bindings[component['path']]=component['sha256']
        m.save(out/'bindings.json',bindings);m.save(out/'pre-source-controls.json',dict(analytic_controls=external.controls(),independent_sparse_review_reused=True,available_bytes=psutil.virtual_memory().available))
        oldrows=list(csv.DictReader((OLD/'vectors.csv').open(encoding='utf-8')))
        probes=None
        for ell in [.25,.5]:
            selected=[r for r in oldrows if r['grid']=='box' and float(r['ell'])==ell and int(r['direction'])==2]
            pp=np.array([[float(r[k]) for k in ['x','y','z']] for r in selected]);gg=np.array([[float(r[k]) for k in ['gx','gy','gz']] for r in selected]);assert len(pp)==385 and np.array_equal(pp[-1],[0,0,0])
            if probes is None:probes=pp
            else:assert np.array_equal(pp,probes)
            samples[(ell,'base12')]=gg
            rows.extend(dict(grid='base12',ell=ell,x=float(p[0]),y=float(p[1]),z=float(p[2]),gx=float(v[0]),gy=float(v[1]),gz=float(v[2]),is_origin=i==384,reused=True) for i,(p,v) in enumerate(zip(pp,gg)))
        original=m.cg
        def bounded(*args,**kw):
            previous=kw['callback']
            def callback(x):
                previous(x)
                if time.perf_counter()-start>300:raise TimeoutError('Frozen 300 second budget')
            kw['callback']=callback;return original(*args,**kw)
        m.cg=bounded
        for grid,half,spacing in [('base18',[18,18,9],[.25,.25,.125]),('base24',[24,24,12],[.25,.25,.125]),('fine18',[18,18,9],[.125,.125,.0625])]:
            available=psutil.virtual_memory().available
            if available<30*1024**3:raise MemoryError('Frozen available RAM floor30GiB')
            if time.perf_counter()-start>300:raise TimeoutError('Frozen budget before grid')
            axes=[np.arange(-round(b/h),round(b/h)+1)*h for b,h in zip(half,spacing)];rho,masses=m.source(case,axes)
            for ell in [.25,.5]:
                smooth=gaussian_filter(rho,[ell/h for h in spacing],mode='constant',cval=0,truncate=4);eps=.2+.8*smooth/(smooth+1e7);del smooth
                bc=np.broadcast_to(-axes[2][None,None,:],rho.shape).copy();phi,check=external.solve(eps,spacing,bc);g=external.sample(phi,axes,probes);samples[(ell,grid)]=g
                records.append(dict(grid=grid,ell=ell,shape=list(rho.shape),halfwidths=half,spacing=spacing,seconds=time.perf_counter()-start,available_bytes_before_grid=available,process_rss_bytes=psutil.Process().memory_info().rss,masses=masses,**check))
                rows.extend(dict(grid=grid,ell=ell,x=float(p[0]),y=float(p[1]),z=float(p[2]),gx=float(v[0]),gy=float(v[1]),gz=float(v[2]),is_origin=i==384,reused=False) for i,(p,v) in enumerate(zip(probes,g)))
                m.save(out/'progress.json',records)
                with (out/'vectors.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
                print(grid,ell,round(time.perf_counter()-start,1),flush=True);del phi,bc,eps
            del rho
        comparisons=[]
        for ell in [.25,.5]:
            for old,new in [('base12','base18'),('base18','base24'),('base18','fine18')]:
                for centered in [False,True]:
                    a=samples[(ell,old)].copy();b=samples[(ell,new)].copy()
                    if centered:a-=a[-1];b-=b[-1]
                    a=a[:-1];b=b[:-1]
                    rel=lambda aa,bb:float(np.linalg.norm(aa-bb)/max(np.linalg.norm(bb),1e-8*np.sqrt(len(bb))))
                    rms=rel(a,b);groups=[dict(height=z,relative=rel(a[probes[:-1,2]==z],b[probes[:-1,2]==z])) for z in [0,.2,.5,1]]
                    comparisons.append(dict(ell=ell,comparison=old+'_to_'+new,center_relative=centered,rms=rms,groups=groups,passed=rms<.05 and all(v['relative']<.08 for v in groups)))
        m.save(out/'summary.json',dict(status='CONDITIONAL_BOUNDARY_EXTENSION',new_solves=len(records),reused_solves=2,seconds=time.perf_counter()-start,records=records,comparisons=comparisons,all_separate_gates_passed=all(r['passed'] for r in comparisons),joint_24_fine_endpoint_tested=False,observed_external_field_used=False,observed_motion_scored=False))
    except Exception:m.save(out/'failure.json',dict(traceback=traceback.format_exc(),completed_solves=len(records),seconds=time.perf_counter()-start));raise


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
