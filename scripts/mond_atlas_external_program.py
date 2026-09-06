"""Conditional response to external Dirichlet fields, no observed motion score."""
import csv,json,time,traceback
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from threadpoolctl import threadpool_limits
import mond_atlas_refraction_program as m
P=m.ROOT/'work/gravity-first-principles/mond-atlas-external-program-001'

def divergence(phi,eps,spacing):
    total=np.zeros(tuple(n-2 for n in phi.shape))
    for axis,h in enumerate(spacing):
        flux=np.diff(phi,axis=axis)/h
        left=[slice(None)]*3;right=left.copy();left[axis]=slice(0,-1);right[axis]=slice(1,None)
        a=eps[tuple(left)];b=eps[tuple(right)];flux*=2*a*b/(a+b)
        div=np.diff(flux,axis=axis)/h
        keep=[slice(1,-1)]*3;keep[axis]=slice(None);total+=div[tuple(keep)]
    return total

def solve(eps,spacing,bc):
    phi,original=m.solve(np.zeros_like(eps),eps,spacing,bc)
    onlybc=bc.copy();onlybc[1:-1,1:-1,1:-1]=0
    scale=np.linalg.norm(divergence(onlybc,eps,spacing))
    residual=float(np.linalg.norm(divergence(phi,eps,spacing))/scale)
    assert original['cg_info']==0 and residual<1e-8,(original['cg_info'],residual)
    return phi,dict(cg_info=original['cg_info'],iterations=original['iterations'],boundary_forcing_relative_residual=residual,zero_rhs_relative_diagnostic_applicable=False)

def sample(phi,axes,points):
    out=[]
    for j,a in enumerate(axes):
        g=-np.gradient(phi,a[1]-a[0],axis=j,edge_order=2)
        out.append(RegularGridInterpolator(axes,g)(points))
    return np.array(out).T

def controls():
    points=np.array([[.1,.2,.3],[-.2,.3,-.4],[0,0,0]])
    rows=[]
    for n in [17,33]:
        axis=np.linspace(-1,1,n);axes=[axis]*3;x,y,z=np.meshgrid(*axes,indexing='ij');h=[2/(n-1)]*3
        eps=1+.2*z;exact=-5*np.log1p(.2*z)
        phi,check=solve(eps,h,exact);g=sample(phi,axes,points)
        truth=np.column_stack((np.zeros((len(points),2)),1/(1+.2*points[:,2])))
        err=float(np.linalg.norm(g-truth)/np.linalg.norm(truth));rows.append(dict(n=n,slab_force_relative=err,**check))
        uni,_=solve(np.full_like(z,.2),h,-x);uf=sample(uni,axes,points)
        assert np.max(abs(uf-[1,0,0]))<1e-10
        reverse,_=solve(eps,h,-2*exact);assert np.max(abs(reverse+2*phi))<1e-10
    assert rows[-1]['slab_force_relative']<.002 and rows[-1]['slab_force_relative']<rows[0]['slab_force_relative']
    return rows

def run():
    out=P/'run001';out.mkdir(exist_ok=False)
    try:
        m.save(out/'bindings.json',{p.relative_to(m.ROOT).as_posix():m.digest(p) for p in [Path(__file__),P/'PREFLIGHT.md',m.MANIFEST,m.ROOT/'scripts/mond_atlas_refraction_program.py',m.ROOT/'scripts/mond_atlas_source_resolution.py']})
        m.save(out/'pre-source-controls.json',controls())
        case=next(c for c in json.loads(m.MANIFEST.read_text(encoding='utf-8'))['source_cases'] if c['id']=='f4-stars-h0p4')
        probes=np.array([[r*np.cos(t),r*np.sin(t),z] for z in [0,.2,.5,1] for r in [.5,1,1.5,2,2.5,3] for t in np.arange(16)*2*np.pi/16]+[[0,0,0]])
        grids={'base':([8,8,4],[.25,.25,.125]),'box':([12,12,6],[.25,.25,.125]),'fine':([8,8,4],[.125,.125,.0625]),'finer':([8,8,4],[.0625,.0625,.03125])}
        start=time.perf_counter();original=m.cg
        def timed(*args,**kw):
            previous=kw['callback']
            def callback(x):
                previous(x)
                if time.perf_counter()-start>360:raise TimeoutError('Frozen360second budget')
            kw['callback']=callback;return original(*args,**kw)
        m.cg=timed
        rows=[];records=[];samples={}
        for name,(half,spacing) in grids.items():
            axes=[np.arange(-round(b/h),round(b/h)+1)*h for b,h in zip(half,spacing)]
            rho,masses=m.source(case,axes)
            for ell in [.25,.5]:
                smooth=gaussian_filter(rho,[ell/h for h in spacing],mode='constant',cval=0,truncate=4)
                eps=.2+.8*smooth/(smooth+1e7);del smooth
                for direction in [0,2]:
                    dims=[1,1,1];dims[direction]=len(axes[direction]);bc=np.broadcast_to(-axes[direction].reshape(dims),rho.shape).copy()
                    phi,check=solve(eps,spacing,bc);g=sample(phi,axes,probes);samples[(ell,name,direction)]=g
                    records.append(dict(grid=name,ell=ell,direction=direction,seconds=time.perf_counter()-start,masses=masses,**check))
                    rows.extend(dict(grid=name,ell=ell,direction=direction,x=p[0],y=p[1],z=p[2],gx=v[0],gy=v[1],gz=v[2],is_origin=i==len(probes)-1) for i,(p,v) in enumerate(zip(probes,g)))
                    m.save(out/'progress.json',records);print(name,ell,direction,round(time.perf_counter()-start,1),flush=True)
                    del phi,bc
                del eps
        comparisons=[]
        for ell in [.25,.5]:
            for direction in [0,2]:
                for old,new in [('base','box'),('fine','finer')]:
                    for centered in [False,True]:
                        a=samples[(ell,old,direction)].copy();b=samples[(ell,new,direction)].copy()
                        if centered:a-=a[-1];b-=b[-1]
                        a=a[:-1];b=b[:-1]
                        rel=lambda a,b:float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-8*np.sqrt(len(b))))
                        rms=rel(a,b);groups=[dict(height=z,relative=rel(a[probes[:-1,2]==z],b[probes[:-1,2]==z])) for z in [0,.2,.5,1]]
                        comparisons.append(dict(ell=ell,direction=direction,comparison=old+'_to_'+new,center_relative=centered,rms=rms,groups=groups,passed=rms<.05 and all(v['relative']<.08 for v in groups)))
        with (out/'vectors.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        m.save(out/'summary.json',dict(status='CONDITIONAL_EXTERNAL_BOUNDARY_RESPONSE',solves=len(records),records=records,comparisons=comparisons,all_field_gates_passed=all(v['passed'] for v in comparisons),observed_external_field_measured=False,observed_motion_scored=False))
    except Exception:m.save(out/'failure.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
