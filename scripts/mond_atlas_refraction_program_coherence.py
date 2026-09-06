"""Frozen finite Gaussian density scale affects epsilon only, not matter RHS."""
import csv,json,time,traceback
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter,gaussian_filter1d
from scipy.interpolate import RegularGridInterpolator
from threadpoolctl import threadpool_limits
import mond_atlas_refraction_program as m
P=m.P/'coherence-scale'

def controls():
    uniform=np.ones((33,33,33));smoothed=gaussian_filter(uniform,2.,mode='constant',cval=0,truncate=4)
    error=float(np.max(abs(smoothed[8:-8,8:-8,8:-8]-1)));assert error<1e-12
    impulse=np.zeros_like(uniform);impulse[16,16,16]=1;out=gaussian_filter(impulse,2.,mode='constant',cval=0,truncate=4)
    assert out.min()>=0 and abs(out.sum()-1)<1e-12
    axis=np.arange(-8,9);kernel=np.exp(-axis**2/8);kernel/=kernel.sum();v=np.linspace(0,1,51)**2
    direct=np.convolve(v,kernel,mode='same');production=gaussian_filter1d(v,2.,mode='constant',truncate=4)
    difference=float(np.max(abs(direct-production)));assert difference<1e-12
    eps=.2+.8*out/(out+1e7);assert eps.min()>=.2 and eps.max()<=1
    return dict(constant_interior_error=error,impulse_mass_error=float(abs(out.sum()-1)),independent_1d_convolution_error=difference,passed=True)

def run():
    out=P/'run001';out.mkdir(exist_ok=False)
    dependencies=[Path(__file__),m.ROOT/'scripts/mond_atlas_refraction_program.py',m.ROOT/'scripts/mond_atlas_source_resolution.py',P/'PREFLIGHT.md',m.MANIFEST]
    m.save(out/'pre-access-bindings.json',{p.relative_to(m.ROOT).as_posix():m.digest(p) for p in dependencies})
    try:
        m.save(out/'gaussian-controls.json',controls());started=time.perf_counter();deadline=started+180
        original=m.cg
        def timed(*args,**kwargs):
            before=kwargs['callback']
            def callback(x):
                before(x)
                if time.perf_counter()>deadline:raise TimeoutError('Frozen180second solver budget reached')
            kwargs['callback']=callback;return original(*args,**kwargs)
        m.cg=timed
        cases=json.loads(m.MANIFEST.read_text(encoding='utf-8'))['source_cases'];case=next(c for c in cases if c['id']=='f4-stars-h0p4')
        points=np.array([[r*np.cos(t),r*np.sin(t),z] for z in [0,.2,.5,1.] for r in [.5,1,1.5,2,2.5,3] for t in np.arange(16)*2*np.pi/16])
        grids={'base':([8,8,4],[.25,.25,.125]),'box':([12,12,6],[.25,.25,.125]),'fine':([8,8,4],[.125,.125,.0625]),'finer':([8,8,4],[.0625,.0625,.03125])}
        records=[];rows=[];samples={}
        for name,(half,spacing) in grids.items():
            axes=[np.arange(-round(b/h),round(b/h)+1)*h for b,h in zip(half,spacing)]
            rho,masses=m.source(case,axes);total=sum(v['full_source_mass'] for v in masses);volume=np.prod(spacing)
            gridmass=float(rho.sum()*volume);interiormass=float(rho[1:-1,1:-1,1:-1].sum()*volume)
            rr=np.sqrt(axes[0][:,None,None]**2+axes[1][None,:,None]**2+axes[2][None,None,:]**2);rr=np.maximum(rr,1e-30)
            for ell in [.25,.5]:
                smooth=gaussian_filter(rho,np.array([ell/h for h in spacing]),mode='constant',cval=0,truncate=4)
                lost=float(1-smooth.sum()/rho.sum());assert -.0000000001<lost<.001
                eps=.2+.8*smooth/(smooth+1e7);del smooth
                assert .2<=eps.min()<=eps.max()<=1
                bc=-m.G*total/rr/.2
                potential,check=m.solve(4*np.pi*m.G*rho,eps,spacing,bc)
                vectors=[]
                for axis,h in enumerate(spacing):
                    grad=-np.gradient(potential,h,axis=axis,edge_order=2);vectors.append(RegularGridInterpolator(axes,grad)(points))
                sample=np.array(vectors).T;samples[(ell,name)]=sample
                row=dict(grid=name,ell_kpc=ell,shape=list(rho.shape),full_source_mass=total,grid_mass=gridmass,interior_rhs_mass=interiormass,excluded_boundary_node_mass=gridmass-interiormass,smoothing_grid_mass_loss_fraction=lost,elapsed_seconds=time.perf_counter()-started,**check)
                records.append(row);rows.extend(dict(ell_kpc=ell,grid=name,x=float(p[0]),y=float(p[1]),z=float(p[2]),gx=float(v[0]),gy=float(v[1]),gz=float(v[2])) for p,v in zip(points,sample))
                m.save(out/'progress.json',records);print(ell,name,check,flush=True)
                if not check['passed']:raise RuntimeError('PDE failure')
                del eps,potential,bc,grad
        comparisons=[]
        for ell in [.25,.5]:
            for old,new in [('fine','finer'),('base','box')]:
                a=samples[(ell,old)];b=samples[(ell,new)];rms=float(np.linalg.norm(a-b)/np.linalg.norm(b));groups=[]
                for z in [0,.2,.5,1.]:
                    mask=points[:,2]==z;groups.append(dict(height=z,relative_rms=float(np.linalg.norm(a[mask]-b[mask])/np.linalg.norm(b[mask]))))
                comparisons.append(dict(ell_kpc=ell,comparison=old+'_to_'+new,relative_rms=rms,groups=groups,passed=rms<.05 and all(g['relative_rms']<.08 for g in groups)))
        with (out/'sampled-vectors.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        m.save(out/'summary.json',dict(status='NEW_FINITE_DENSITY_SCALE_CONDITIONAL_LAW',records=records,comparisons=comparisons,all_gates_passed=all(c['passed'] for c in comparisons),observed_response_arrays_opened=0,new_private_bytes=0))
    except Exception:m.save(out/'failure.json',dict(traceback=traceback.format_exc(),retained=True));raise

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
