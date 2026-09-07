"""Joint24kpc finest-grid normal-field endpoint, fixed physics and bounded resources."""
import csv,json,threading,time,traceback
from pathlib import Path
import numpy as np
import psutil
from scipy.ndimage import gaussian_filter
from threadpoolctl import threadpool_limits
import mond_atlas_external_program as external
m=external.m
P=m.ROOT/'work/gravity-first-principles/mond-atlas-external-joint-001'
OLD=m.ROOT/'work/gravity-first-principles/mond-atlas-external-boundary-001/run001'


def run():
    out=P/'run001';out.mkdir(exist_ok=False);start=time.perf_counter();stop=threading.Event();rss=[psutil.Process().memory_info().rss];samples=[]
    def monitor():
        proc=psutil.Process()
        while not stop.is_set():
            value=proc.memory_info().rss;rss[0]=max(rss[0],value);samples.append([time.perf_counter()-start,value]);stop.wait(.1)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    def budget():
        if time.perf_counter()-start>300:raise TimeoutError('Frozen300second budget')
        if rss[0]>16_000_000_000:raise MemoryError('Revised16GB sampled working cap')
    try:
        paths=[Path(__file__),P/'PREFLIGHT.md',P/'RESOURCE_ADDENDUM.md',OLD/'vectors.csv',OLD/'summary.json',OLD/'bindings.json',m.MANIFEST,m.ROOT/'scripts/mond_atlas_external_program.py',m.ROOT/'scripts/mond_atlas_refraction_program.py',m.ROOT/'scripts/mond_atlas_source_resolution.py',external.P/'analytic-review/field-review-operator.json']
        bindings={p.relative_to(m.ROOT).as_posix():m.digest(p) for p in paths}
        for path,h in json.loads((OLD/'bindings.json').read_text(encoding='utf-8')).items():assert m.digest(m.ROOT/path)==h
        case=next(c for c in json.loads(m.MANIFEST.read_text(encoding='utf-8'))['source_cases'] if c['id']=='f4-stars-h0p4')
        for comp in case['components']:assert m.digest(m.ROOT/comp['path'])==comp['sha256'];bindings[comp['path']]=comp['sha256']
        m.save(out/'bindings.json',bindings);controls=external.controls();available=psutil.virtual_memory().available
        assert available>=32_000_000_000
        operator=json.loads((external.P/'analytic-review/field-review-operator.json').read_text(encoding='utf-8'));assert operator['status']=='PASS'
        m.save(out/'pre-source-controls.json',dict(analytic_controls=controls,independent_sparse_review_reused=True,available_memory_before_bytes=available,shape=[385,385,385],working_cap_bytes=16_000_000_000))
        oldrows=list(csv.DictReader((OLD/'vectors.csv').open(encoding='utf-8')));vectors={};rows=[];probes=None
        for grid in ['fine18','base24']:
            selected=[r for r in oldrows if r['grid']==grid and float(r['ell'])==.5];pp=np.array([[float(r[k]) for k in ['x','y','z']] for r in selected]);gg=np.array([[float(r[k]) for k in ['gx','gy','gz']] for r in selected]);assert len(pp)==385
            if probes is None:probes=pp
            else:assert np.array_equal(pp,probes)
            vectors[grid]=gg;rows.extend(dict(grid=grid,ell=.5,x=float(p[0]),y=float(p[1]),z=float(p[2]),gx=float(v[0]),gy=float(v[1]),gz=float(v[2]),is_origin=i==384,reused=True) for i,(p,v) in enumerate(zip(pp,gg)))
        original=m.cg
        def bounded(*args,**kw):
            previous=kw['callback']
            def callback(x):previous(x);budget()
            kw['callback']=callback;return original(*args,**kw)
        m.cg=bounded;spacing=[.125,.125,.0625];half=[24,24,12];axes=[np.arange(-round(b/h),round(b/h)+1)*h for b,h in zip(half,spacing)]
        rho,masses=m.source(case,axes);budget();smooth=gaussian_filter(rho,[.5/h for h in spacing],mode='constant',cval=0,truncate=4);eps=.2+.8*smooth/(smooth+1e7);del smooth;budget()
        bc=np.broadcast_to(-axes[2][None,None,:],rho.shape).copy();phi,check=external.solve(eps,spacing,bc);budget();gg=external.sample(phi,axes,probes);vectors['fine24']=gg;budget()
        rows.extend(dict(grid='fine24',ell=.5,x=float(p[0]),y=float(p[1]),z=float(p[2]),gx=float(v[0]),gy=float(v[1]),gz=float(v[2]),is_origin=i==384,reused=False) for i,(p,v) in enumerate(zip(probes,gg)))
        with (out/'vectors.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        comparisons=[]
        for old in ['fine18','base24']:
            for centered in [False,True]:
                a=vectors[old].copy();b=vectors['fine24'].copy()
                if centered:a-=a[-1];b-=b[-1]
                a=a[:-1];b=b[:-1]
                metric=lambda aa,bb:float(np.linalg.norm(aa-bb)/max(np.linalg.norm(bb),1e-8*np.sqrt(len(bb))))
                rms=metric(a,b);groups=[dict(height=z,relative=metric(a[probes[:-1,2]==z],b[probes[:-1,2]==z])) for z in [0,.2,.5,1]]
                comparisons.append(dict(comparison=old+'_to_fine24',center_relative=centered,rms=rms,groups=groups,passed=rms<.05 and max(g['relative'] for g in groups)<.08))
        m.save(out/'summary.json',dict(status='CONDITIONAL_JOINT_ENDPOINT',ell=.5,new_solves=1,reused_solves=2,shape=list(rho.shape),seconds=time.perf_counter()-start,sampled_peak_rss_bytes=rss[0],working_cap_bytes=16_000_000_000,masses=masses,solver=check,comparisons=comparisons,joint_endpoint_gates_passed=all(r['passed'] for r in comparisons),observed_external_amplitude_used=False,observed_motion_scored=False,alternative21_executed=False))
        print(json.dumps(json.loads((out/'summary.json').read_text(encoding='utf-8')),indent=2))
    except Exception:m.save(out/'failure.json',dict(traceback=traceback.format_exc(),seconds=time.perf_counter()-start,sampled_peak_rss_bytes=rss[0]));raise
    finally:
        stop.set();thread.join();m.save(out/'memory-samples.json',dict(sample_interval_seconds=.1,samples=samples,sampled_peak_rss_bytes=rss[0]))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
