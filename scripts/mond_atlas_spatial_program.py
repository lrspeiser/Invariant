"""Finite distributed secondary kernel on conditional observed tracer sources."""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
from mond_atlas_source_resolution import cell_projection_matrix

G = 4.30091727003628e-6
BASE = Path('work/gravity-first-principles/mond-atlas-spatial-program-001')

def pair(points, sources, masses, length=4., cutoff=10., xp=np):
    """Return potential and Cartesian acceleration; finite kernel, eta=1."""
    points, sources, masses = (xp.asarray(a, dtype=xp.float64) for a in (points,sources,masses))
    out=[]
    for p in points:
        d=p-sources
        s=xp.sqrt(xp.sum(d*d,axis=1))
        safe=xp.maximum(s,1e-30)
        x=safe/length
        y=xp.minimum(x,cutoff)
        # Stable enclosed mass for small separation.
        m=xp.where(y<1e-4, y*y*(.5-y*(2/3-y*.75)), xp.log1p(y)-y/(1+y))
        f=xp.where(x<cutoff,xp.log1p(x)/safe-1/(length*(1+cutoff)),m/safe)
        a=-G*xp.sum((m*masses/safe**3)[:,None]*d,axis=0)
        out.append(xp.concatenate((xp.asarray([-G*xp.sum(masses*f)]),a)))
    return xp.stack(out)

def planar(packet, conversion, spacing):
    nodes=packet['latent_axis']; surface=packet['intrinsic_effective_surface']
    h=float(nodes[1]-nodes[0])
    if np.any(surface[[0,-1]]) or np.any(surface[:,[0,-1]]) or np.any(surface<0):
        raise ValueError('nonnegative compact bilinear source required')
    centers=np.arange(nodes[0]+spacing/2,nodes[-1],spacing)
    mat=cell_projection_matrix(centers,spacing,nodes,h,0)
    mass=(mat@surface@mat.T)*(conversion*1e6*spacing**2)
    x,y=np.meshgrid(centers,centers,indexing='ij')
    keep=mass>0
    expected=float(surface.sum()*h*h*conversion*1e6)
    assert abs(mass.sum()/expected-1)<1e-10
    return np.column_stack([x[keep],y[keep]]),mass[keep],expected

def integrate(points, xy, masses, layers, order, xp=np):
    nodes,weights=np.polynomial.laguerre.laggauss(order)
    pts=xp.asarray(points); xy=xp.asarray(xy); masses=xp.asarray(masses)
    total=xp.zeros((len(points),4))
    for fraction,height in layers:
        for z,w in zip(nodes,weights):
            for sign in [-1,1]:
                src=xp.column_stack((xy,xp.full(len(xy),sign*z*height)))
                total+=pair(pts,src,masses*(fraction*w/2),xp=xp)
    return total if xp is np else xp.asnumpy(total)

def controls():
    src=np.array([[.1,.3,-.2],[-.6,.5,.8]])
    p=np.array([[1.,2.,3.],[50.,4.,3.]])
    mass=np.array([2e8,3e8]); a=pair(p,src,mass)
    errors={}
    errors['translation']=float(np.max(np.abs(pair(p+7,src+7,mass)-a))/np.max(abs(a)))
    rot=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    b=pair(p@rot.T,src@rot.T,mass)
    errors['rotation']=float(np.max(abs(b[:,1:]-a[:,1:]@rot.T))/np.max(abs(a[:,1:])))
    delta=1e-4; grad=[]
    for j in range(3):
        step=np.eye(3)[j]*delta
        grad.append(-(pair(p+step,src,mass)[:,0]-pair(p-step,src,mass)[:,0])/(2*delta))
    errors['potential_gradient']=float(np.max(abs(np.array(grad).T-a[:,1:]))/np.max(abs(a[:,1:])))
    f1=pair(src[:1],src[1:],mass[1:])[0,1:]*mass[0]
    f2=pair(src[1:],src[:1],mass[:1])[0,1:]*mass[1]
    errors['reciprocity']=float(np.linalg.norm(f1+f2)/np.linalg.norm(f1))
    m10=np.log(11)-10/11
    expected=-G*mass[0]*m10/100**2
    errors['outer_inverse_square']=float(abs(pair([[100,0,0]],[[0,0,0]],mass[:1])[0,1]/expected-1))
    # Independent spherical radial integral of q yields m(x).
    from scipy.integrate import quad
    errors['enclosed_kernel']=max(abs(quad(lambda t:t/(1+t)**2,0,x)[0]-(np.log1p(x)-x/(1+x))) for x in [.1,1,10])
    import cupy as cp
    errors['cpu_gpu']=float(np.max(abs(cp.asnumpy(pair(p,src,mass,xp=cp))-a))/np.max(abs(a)))
    for key,value in errors.items():
        assert value < (1e-6 if key=='potential_gradient' else 1e-10),(key,value)
    return errors

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--controls-only',action='store_true'); args=parser.parse_args()
    BASE.mkdir(parents=True,exist_ok=True)
    checks=controls()
    (BASE/'controls.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    if args.controls_only: print(json.dumps(checks)); return
    import cupy as cp
    config=json.loads((BASE/'source-bindings.json').read_text(encoding='utf-8'))
    params=[(r,theta,z) for r in [1.,3.,6.] for z in [0.,.4] for theta in np.arange(12)*2*np.pi/12]
    points=np.array([[r*np.cos(t),r*np.sin(t),z] for r,t,z in params])
    rows=[]; masslog=[]; start=time.time(); cache={}
    for case in config['source_cases']:
        for level,(spacing,order) in enumerate([(.125,12),(.0625,24),(.03125,48)]):
            total=np.zeros((len(points),4))
            for comp in case['components']:
                path=Path(comp['path']); key=(str(path),spacing,order)
                if key not in cache:
                    assert hashlib.sha256(path.read_bytes()).hexdigest()==comp['sha256']
                    with np.load(path) as pkt:
                        xy,mass,expected=planar(pkt,comp['conversion_to_msun_pc2'],spacing)
                        assert np.allclose(pkt['vertical_layers'],comp['vertical_layers'])
                    result=integrate(points,xy,mass,comp['vertical_layers'],order,xp=cp)
                    cache[key]=result
                    masslog.append(dict(path=str(path),spacing=spacing,order=order,mass_msun=expected,quadrature_mass_msun=float(mass.sum()),cells=len(mass)))
                result=cache[key]; total+=result
                for i,(r,t,z) in enumerate(params):
                    rows.append(dict(case=case['id'],level=level,component=comp['id'],r=r,theta=t,z=z,phi=result[i,0],gx=result[i,1],gy=result[i,2],gz=result[i,3]))
            for i,(r,t,z) in enumerate(params):
                rows.append(dict(case=case['id'],level=level,component='total',r=r,theta=t,z=z,phi=total[i,0],gx=total[i,1],gy=total[i,2],gz=total[i,3]))
            with (BASE/'fields.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
            print(case['id'],level,'elapsed',round(time.time()-start,1),flush=True)
    (BASE/'mass-checks.json').write_text(json.dumps(masslog,indent=2)+'\n',encoding='utf-8')
    (BASE/'run.json').write_text(json.dumps(dict(status='CONDITIONAL_SOURCE_DIAGNOSTICS',observed_response_scored=False,seconds=time.time()-start,rows=len(rows),controls=checks,device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode()),indent=2)+'\n',encoding='utf-8')

if __name__=='__main__': main()
