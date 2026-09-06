"""Full source readback and independent CPU quadrature; no observed targets."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,json,hashlib,sys
from pathlib import Path
import numpy as np
from scipy.special import roots_laguerre
from scipy.integrate import quad
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());P=Path(__file__).resolve().parent;BASE=P.parent
sys.path.insert(0,str(ROOT/'scripts'))
from mond_atlas_spatial_program import pair
G=4.30091727003628e-6

def cellmass(nodes,surface,conversion,width):
    # Independently integrate the rising/falling linear pieces of each tent.
    h=float(nodes[1]-nodes[0]);centers=np.arange(nodes[0]+width/2,nodes[-1],width)
    a=centers[:,None]-width/2;b=a+width;n=nodes[None,:]
    u=np.maximum(a,n-h);v=np.minimum(b,n)
    left=np.where(v>u,((v-n+h)**2-(u-n+h)**2)/(2*h),0)
    u=np.maximum(a,n);v=np.minimum(b,n+h)
    right=np.where(v>u,(v-u)-((v-n)**2-(u-n)**2)/(2*h),0)
    integral=left+right
    mass=integral@surface@integral.T*conversion*1e6
    x,y=np.meshgrid(centers,centers,indexing='ij');keep=mass>0
    return np.column_stack([x[keep],y[keep]]),mass[keep]

def evaluate(point,xy,masses,layers,order):
    nodes,weights=roots_laguerre(order);total=np.zeros(4)
    for fraction,height in layers:
        for node,weight in zip(nodes,weights):
            for sign in (-1,1):
                d=np.column_stack([point[0]-xy[:,0],point[1]-xy[:,1],np.full(len(xy),point[2]-sign*node*height)])
                radius=np.linalg.norm(d,axis=1);x=radius/4;u=np.minimum(x,10.)
                enclosed=np.log1p(u)-1+1/(1+u)
                small=u<1e-3
                enclosed[small]=sum((-1)**k*(k-1)/k*u[small]**k for k in range(2,12))
                # Enclosed/source-shell decomposition of the spherical potential.
                outer=np.where(x<10,(1/(1+x)-1/11)/4,0)
                potential=-G*np.sum(masses*(enclosed/radius+outer))
                accel=np.sum((-G*masses*enclosed/radius**2)[:,None]*(d/radius[:,None]),axis=0)
                total+=fraction*weight/2*np.r_[potential,accel]
    return total

def main():
    cfg=json.loads((BASE/'source-bindings.json').read_text(encoding='utf-8'))
    unique={c['path']:c for case in cfg['source_cases'] for c in case['components']}
    controls={}
    for x in (1e-5,.1,1,9.999,10,10.001,100):
        m=quad(lambda t:t/(1+t)**2,0,min(x,10),epsabs=1e-13)[0]
        expected=-G*2*m/(4*x)**2
        actual=pair([[4*x,0,0]],[[0,0,0]],[2])[0,1]
        controls[str(x)]=abs(actual/expected-1)
    assert max(controls.values())<1e-9
    manifest=[];massrows=[];subset=[]
    # Read every key of every source packet, not only arrays used by the evaluator.
    for path,c in unique.items():
        raw=ROOT/path;assert hashlib.sha256(raw.read_bytes()).hexdigest()==c['sha256']
        with np.load(raw) as z:packet={k:z[k] for k in z.files}
        nonfinite={}
        for k,v in packet.items():
            if np.issubdtype(v.dtype,np.number) and not np.isfinite(v).all():
                bad=~np.isfinite(v);nonfinite[k]=int(bad.sum())
                if k!='source_mean' or np.any(packet['fit_weight'][bad]) or np.any(packet['evaluation_weight'][bad]):raise ValueError('Nonfinite admitted packet values '+path+' '+k)
        nodes=packet['latent_axis'];surface=packet['intrinsic_effective_surface'];layers=packet['vertical_layers']
        h=np.diff(nodes);assert np.allclose(h,h[0],atol=1e-13,rtol=0)
        assert surface.shape==(len(nodes),len(nodes)) and surface.min()>=0
        assert not np.any(surface[[0,-1]]) and not np.any(surface[:,[0,-1]])
        assert np.allclose(layers,c['vertical_layers']) and abs(layers[:,0].sum()-1)<1e-12
        exact=float(surface.sum()*h[0]**2*c['conversion_to_msun_pc2']*1e6)
        manifest.append(dict(path=path,sha256=c['sha256'],keys={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in packet.items()},excluded_nonfinite_source_mean=nonfinite,mass_msun=exact,axis='major X first, deprojected minor Y second'))
        for level,(spacing,order) in enumerate(((.125,12),(.0625,24),(.03125,48))):
            xy,mass=cellmass(nodes,surface,c['conversion_to_msun_pc2'],spacing)
            error=abs(mass.sum()/exact-1);assert error<1e-10
            massrows.append(dict(path=path,level=level,mass_msun=float(mass.sum()),relative_mass_error=float(error),cells=len(mass)))
            # Every unique component gets two independent coarse CPU field checks.
            # One fine point on first source also checks the largest quadrature.
            if level==0 or (level==2 and path==next(iter(unique))):
                positions=[(1.,0.,0.),(3.,np.pi/3,.4)] if level==0 else [(3.,np.pi/3,.4)]
                for r,t,z in positions:
                    val=evaluate([r*np.cos(t),r*np.sin(t),z],xy,mass,layers,order)
                    subset.append(dict(path=path,level=level,r=r,theta=t,z=z,phi=float(val[0]),gx=float(val[1]),gy=float(val[2]),gz=float(val[3])))
    (P/'source-readback.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (P/'independent-masses.json').write_text(json.dumps(massrows,indent=2)+'\n',encoding='utf-8')
    (P/'cpu-subset.json').write_text(json.dumps(subset,indent=2)+'\n',encoding='utf-8')
    (P/'kernel-controls.json').write_text(json.dumps(controls,indent=2)+'\n',encoding='utf-8')
    if not (BASE/'run.json').exists():
        print('Source readback and CPU subset complete; parent run.json pending.');return
    compare()

def compare():
    cfg=json.loads((BASE/'source-bindings.json').read_text(encoding='utf-8'));lookup={(case['id'],c['id']):c['path'] for case in cfg['source_cases'] for c in case['components']}
    with (BASE/'fields.csv').open(encoding='utf-8',newline='') as f:fields=list(csv.DictReader(f))
    subset=json.loads((P/'cpu-subset.json').read_text(encoding='utf-8'));comparisons=[]
    for ref in subset:
        matches=[r for r in fields if r['component']!='total' and lookup[(r['case'],r['component'])]==ref['path'] and int(r['level'])==ref['level'] and abs(float(r['r'])-ref['r'])<1e-10 and abs(float(r['theta'])-ref['theta'])<1e-10 and abs(float(r['z'])-ref['z'])<1e-10]
        assert matches
        for row in matches:
            a=np.array([float(row[k]) for k in ('phi','gx','gy','gz')]);b=np.array([ref[k] for k in ('phi','gx','gy','gz')]);err=float(np.max(abs(a-b))/np.max(abs(b)))
            comparisons.append(dict(case=row['case'],component=row['component'],level=ref['level'],relative_max_error=err,force_relative_error=float(np.linalg.norm(a[1:]-b[1:])/np.linalg.norm(b[1:]))))
    convergence=[]
    for case in cfg['source_cases']:
        for component in [c['id'] for c in case['components']]+['total']:
            bylevel={level:[r for r in fields if r['case']==case['id'] and r['component']==component and int(r['level'])==level] for level in (1,2)}
            a=np.array([[float(r[k]) for k in ('gx','gy','gz')] for r in bylevel[1]]);b=np.array([[float(r[k]) for k in ('gx','gy','gz')] for r in bylevel[2]])
            assert a.shape==b.shape==(72,3)
            rms=float(np.linalg.norm(a-b)/np.linalg.norm(b));maximum=float(np.max(np.linalg.norm(a-b,axis=1)/np.linalg.norm(b,axis=1)))
            convergence.append(dict(case=case['id'],component=component,relative_rms=rms,max_point_relative=maximum,passes=bool(rms<.01 and maximum<.03)))
    receipt=dict(comparisons=comparisons,maximum_cpu_gpu_relative=max(r['relative_max_error'] for r in comparisons),independent_subset_pass=all(r['relative_max_error']<1e-10 for r in comparisons),convergence=convergence,observed_targets_opened=False,source_readback_complete=True)
    (P/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    if '--compare-only' in sys.argv:compare()
    else:main()
