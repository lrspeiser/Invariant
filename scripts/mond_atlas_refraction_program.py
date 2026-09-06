"""Harmonic-face variable-permittivity conditional gravity, CPU finite volume."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.fft import dstn,idstn
from scipy.sparse.linalg import LinearOperator,cg
from scipy.interpolate import RegularGridInterpolator
from threadpoolctl import threadpool_limits
from mond_atlas_source_resolution import cell_projection_matrix
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'work/gravity-first-principles/mond-atlas-refraction-program-001'
PRIVATE=ROOT/'work/private/mond-atlas-refraction-program-001'
MANIFEST=ROOT/'work/gravity-first-principles/mond-atlas-spatial-program-001/source-bindings.json'
G=4.30091727003628e-6
def save(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def solve(rhs,epsilon,spacing,boundary):
    shape=rhs.shape;inner=tuple(n-2 for n in shape);mid=(slice(1,-1),)*3
    faces=[]
    for axis,h in enumerate(spacing):
        a=[slice(None)]*3;b=a.copy();a[axis]=slice(0,-1);b[axis]=slice(1,None)
        low=epsilon[tuple(a)];high=epsilon[tuple(b)];face=2*low*high/(low+high)/h**2
        plus=[slice(1,-1)]*3;minus=plus.copy();plus[axis]=slice(1,None);minus[axis]=slice(0,-1)
        faces.append((face[tuple(minus)],face[tuple(plus)]))
    def apply(full):
        result=np.zeros(inner)
        for axis,(minus,plus) in enumerate(faces):
            lo=[slice(1,-1)]*3;hi=lo.copy();lo[axis]=slice(0,-2);hi[axis]=slice(2,None)
            result+=minus*(full[mid]-full[tuple(lo)])+plus*(full[mid]-full[tuple(hi)])
        return result
    bc=boundary.copy();bc[mid]=0
    b=-rhs[mid]-apply(bc)
    eig=np.zeros(inner)
    for axis,(n,h) in enumerate(zip(inner,spacing)):
        vals=4*np.sin(np.pi*np.arange(1,n+1)/(2*(n+1)))**2/h**2
        dims=[1]*3;dims[axis]=n;eig+=vals.reshape(dims)
    def mat(v):
        full=np.zeros(shape);full[mid]=v.reshape(inner);return apply(full).ravel()
    def pre(v):return idstn(dstn(v.reshape(inner),type=1,norm='ortho')/eig,type=1,norm='ortho').ravel()
    size=b.size;iterations=[0]
    def callback(x):iterations[0]+=1
    x,info=cg(LinearOperator((size,size),matvec=mat,dtype=float),b.ravel(),M=LinearOperator((size,size),matvec=pre,dtype=float),rtol=1e-10,atol=0,maxiter=500,callback=callback)
    phi=bc;phi[mid]=x.reshape(inner)
    residual=apply(phi)+rhs[mid]
    relative=float(np.linalg.norm(residual)/max(np.linalg.norm(rhs[mid]),1e-30))
    fluxerror=float(abs(residual.sum())/max(abs(rhs[mid].sum()),1e-30))
    return phi,dict(iterations=iterations[0],cg_info=int(info),physical_relative_residual=relative,discrete_flux_relative_error=fluxerror,passed=bool(info==0 and relative<1e-8 and fluxerror<1e-8))

def controls():
    results=[];variable=[]
    for n in [17,33]:
        axis=np.linspace(-1,1,n);x,y,z=np.meshgrid(axis,axis,axis,indexing='ij');truth=x*x+y*y+z*z
        for varying in [False,True]:
            eps=1+.2*x if varying else np.ones_like(x)*.4
            rhs=6+1.6*x if varying else np.ones_like(x)*2.4
            phi,r=solve(rhs,eps,[2/(n-1)]*3,truth)
            err=float(np.linalg.norm(phi-truth)/np.linalg.norm(truth));assert err<.002 and r['passed']
            results.append(dict(n=n,variable=varying,potential_relative_error=err,**r))
            if varying:variable.append(err)
    assert variable[1]<variable[0]*.4
    # Positive harmonic interface and conservation of equal/opposite internal flux.
    a=np.array([.2,.3,.8,1.]);b=a[::-1];h=2*a*b/(a+b)
    assert np.array_equal(h,2*b*a/(b+a)) and np.all(h>0)
    delta=np.array([2.,-1.,.3,7.]);assert np.sum(h*delta**2)>0
    dz=.125;z=np.arange(-32,33)*dz;height=.2
    F=lambda t:.5*(1+np.sign(t)*(-np.expm1(-abs(t)/height)))
    frac=(F(z+dz/2)-F(z-dz/2)).sum();expected=1-np.exp(-(z[-1]+dz/2)/height)
    assert abs(frac-expected)<1e-10
    return results

def source(case,axes):
    rho=np.zeros(tuple(len(a) for a in axes));records=[];dx,dy,dz=[a[1]-a[0] for a in axes]
    for component in case['components']:
        path=ROOT/component['path'];assert digest(path)==component['sha256']
        with np.load(path,allow_pickle=False) as packet:nodes=packet['latent_axis'];surface=packet['intrinsic_effective_surface']
        assert np.isfinite(surface).all() and surface.min()>=0 and not np.any(surface[[0,-1]]) and not np.any(surface[:,[0,-1]])
        h=nodes[1]-nodes[0];left=cell_projection_matrix(axes[0],dx,nodes,h,0);right=cell_projection_matrix(axes[1],dy,nodes,h,0)
        conversion=component['conversion_to_msun_pc2']*1e6;plane=(left@surface@right.T)*conversion
        vertical=np.zeros(len(axes[2]));z=axes[2]
        for weight,height in component['vertical_layers']:
            F=lambda t:.5*(1+np.sign(t)*(-np.expm1(-abs(t)/height)))
            vertical+=weight*(F(z+dz/2)-F(z-dz/2))/dz
        added=plane[:,:,None]*vertical[None,None,:];rho+=added
        full=float(surface.sum()*h*h*conversion);finite=float(added.sum()*dx*dy*dz)
        expected=float(plane.sum()*dx*dy*vertical.sum()*dz)
        assert abs(finite/expected-1)<1e-10 and finite<=full*(1+1e-10)
        records.append(dict(component=component['id'],full_source_mass=full,finite_grid_mass=finite,finite_fraction=finite/full,relative_mass_arithmetic_error=abs(finite/expected-1)))
    return rho,records

def run():
    out=P/'run001';out.mkdir(exist_ok=False);PRIVATE.mkdir(exist_ok=False)
    deps=[Path(__file__),ROOT/'scripts/mond_atlas_source_resolution.py',P/'PREFLIGHT.md',MANIFEST]
    save(out/'pre-access-bindings.json',{p.relative_to(ROOT).as_posix():digest(p) for p in deps})
    save(out/'manufactured-controls.json',controls())
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'));cases={c['id']:c for c in manifest['source_cases']}
    radii=np.array([.5,1,1.5,2,2.5,3]);phiangles=np.arange(16)*2*np.pi/16;heights=[0,.2,.5,1.]
    points=np.array([[r*np.cos(t),r*np.sin(t),z] for z in heights for r in radii for t in phiangles]);samples={};runs=[];allrows=[]
    grids={'base':([8,8,4],[.25,.25,.125]),'fine':([8,8,4],[.125,.125,.0625]),'box':([12,12,6],[.25,.25,.125])}
    for caseid,gridname in [('f4-stars-h0p4',g) for g in grids]+[('f4-stars-h0p1','base')]:
        half,spacing=grids[gridname];axes=[np.arange(-round(b/h),round(b/h)+1)*h for b,h in zip(half,spacing)]
        rho,mass=source(cases[caseid],axes);fullmass=sum(m['full_source_mass'] for m in mass)
        rr=np.sqrt(sum(a.reshape(tuple(len(a) if j==i else 1 for j in range(3)))**2 for i,a in enumerate(axes)));rr=np.maximum(rr,1e-30)
        for model in ['newton','refraction']:
            start=time.perf_counter();epsilon=np.ones_like(rho) if model=='newton' else .2+.8*rho/(rho+1e7)
            boundary=-G*fullmass/rr/(1 if model=='newton' else .2)
            potential,check=solve(4*np.pi*G*rho,epsilon,spacing,boundary)
            force=-np.stack(np.gradient(potential,*spacing,edge_order=2),axis=-1)
            sampled=RegularGridInterpolator(axes,force)(points);samples[(caseid,gridname,model)]=sampled
            filename=f'{caseid}-{gridname}-{model}.npz';np.savez_compressed(PRIVATE/filename,potential=potential,axes_x=axes[0],axes_y=axes[1],axes_z=axes[2])
            record=dict(case=caseid,grid=gridname,model=model,shape=list(rho.shape),mass=mass,epsilon_min=float(epsilon.min()),epsilon_max=float(epsilon.max()),seconds=time.perf_counter()-start,packet_path=(PRIVATE/filename).relative_to(ROOT).as_posix(),packet_sha256=digest(PRIVATE/filename),**check)
            runs.append(record);save(out/'progress.json',runs)
            allrows.extend(dict(case=caseid,grid=gridname,model=model,x=float(p[0]),y=float(p[1]),z=float(p[2]),gx=float(v[0]),gy=float(v[1]),gz=float(v[2])) for p,v in zip(points,sampled))
            print(caseid,gridname,model,check,flush=True)
            if not check['passed']:raise RuntimeError('PDE failure retained')
    comparisons=[]
    for model in ['newton','refraction']:
        for alternative in ['fine','box']:
            a=samples[('f4-stars-h0p4','base',model)];b=samples[('f4-stars-h0p4',alternative,model)]
            relative=float(np.linalg.norm(a-b)/np.linalg.norm(b));groups=[]
            for z in heights:
                mask=points[:,2]==z;groups.append(dict(height=z,relative_rms=float(np.linalg.norm(a[mask]-b[mask])/np.linalg.norm(b[mask]))))
            comparisons.append(dict(model=model,comparison=alternative+'_vs_base',relative_rms=relative,groups=groups,passed=relative<.05 and all(g['relative_rms']<.08 for g in groups)))
    effects=[]
    for caseid,grid in [('f4-stars-h0p4',g) for g in grids]+[('f4-stars-h0p1','base')]:
        n=samples[(caseid,grid,'newton')];r=samples[(caseid,grid,'refraction')]
        ratio=np.linalg.norm(r,axis=1)/np.linalg.norm(n,axis=1)
        angle=np.degrees(np.arccos(np.clip(np.sum(r*n,axis=1)/(np.linalg.norm(r,axis=1)*np.linalg.norm(n,axis=1)),-1,1)))
        effects.append(dict(case=caseid,grid=grid,median_strength_ratio=float(np.median(ratio)),minimum_strength_ratio=float(ratio.min()),maximum_strength_ratio=float(ratio.max()),median_direction_change_deg=float(np.median(angle)),maximum_direction_change_deg=float(angle.max())))
    with (out/'sampled-vectors.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    size=sum(p.stat().st_size for p in PRIVATE.iterdir());assert size<1024**3
    save(out/'summary.json',dict(status='CONDITIONAL_SPATIAL_PDE_NOT_OBSERVATION',runs=runs,convergence=comparisons,effects=effects,all_numerical_field_gates_passed=all(r['passed'] for r in comparisons),private_bytes=size,observed_response_arrays_opened=0,source_3d_truth=False,parameters_fitted=False))

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
