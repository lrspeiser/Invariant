"""Fixed logarithmic pair response on inherited conditional source quadrature."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
import mond_atlas_spatial_program as old
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-log-source-001';G=old.G;L=4.;B=.05
def save(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def pair(points,src,mass,xp=np):
    pts,src,mass=[xp.asarray(a,dtype=xp.float64) for a in (points,src,mass)];out=[]
    for p in pts:
        d=p-src;r2=xp.sum(d*d,axis=1);s=xp.sqrt(r2+B*B);r=xp.sqrt(r2)
        a=-G*xp.sum((mass/(s*s*(s+L)))[:,None]*d,axis=0);phi=-G/L*xp.sum(mass*xp.log1p(L/s))
        n=-G*xp.sum((mass/xp.maximum(r,1e-30)**3)[:,None]*d,axis=0);pn=-G*xp.sum(mass/xp.maximum(r,1e-30))
        out.append(xp.stack([xp.concatenate((xp.asarray([phi]),a)),xp.concatenate((xp.asarray([pn]),n))]))
    return xp.stack(out)
def integrate(points,xy,mass,layers,order,xp):
    nodes,weights=np.polynomial.laguerre.laggauss(order);xy=xp.asarray(xy);mass=xp.asarray(mass);p=xp.asarray(points);total=xp.zeros((len(points),2,4))
    for fraction,height in layers:
        for z,w in zip(nodes,weights):
            for sign in [-1,1]:
                src=xp.column_stack((xy,xp.full(len(xy),sign*z*height)));total+=pair(p,src,mass*fraction*w/2,xp)
    return total if xp is np else xp.asnumpy(total)
def controls(cp):
    from scipy.integrate import quad
    p=np.array([[1.,2.,3.],[8.,1.,2.]]);src=np.array([[.2,-.3,.1],[-1.,.4,.5]]);mass=np.array([2e8,3e8]);a=pair(p,src,mass);norm=np.max(abs(a));rot=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    err=dict(translation=float(np.max(abs(pair(p+4,src+4,mass)-a))/norm),mass_split=float(np.max(abs(pair(p,np.repeat(src,2,axis=0),np.repeat(mass/2,2))-a))/norm))
    ar=pair(p@rot.T,src@rot.T,mass);err['rotation']=float(np.max(abs(ar[:,:,1:]-a[:,:,1:]@rot.T))/norm)
    h=1e-4;grad=np.stack([-(pair(p+h*np.eye(3)[j],src,mass)[:,:,0]-pair(p-h*np.eye(3)[j],src,mass)[:,:,0])/(2*h) for j in range(3)],axis=-1)
    err['gradient']=float(np.max(abs(grad-a[:,:,1:]))/np.max(abs(a[:,:,1:])))
    f=mass[0]*pair(src[:1],src[1:],mass[1:])[0,0,1:]+mass[1]*pair(src[1:],src[:1],mass[:1])[0,0,1:];err['reciprocity']=float(np.linalg.norm(f)/(G*mass.prod()))
    err['cpu_gpu']=float(np.max(abs(cp.asnumpy(pair(p,src,mass,cp))-a))/norm)
    v=pair([[1e8,0,0]],[[0,0,0]],[1])[0,0,1];err['outer_ratio']=float(abs(v/(-G/1e16)-1))
    for radius in [.01,.1,1,10]:
        val=quad(lambda t:t/((t*t+B*B)*(np.sqrt(t*t+B*B)+L)),radius,np.inf,epsabs=1e-11)[0]
        expected=np.log1p(L/np.sqrt(radius*radius+B*B))/L
        assert abs(val/expected-1)<1e-9
    r=np.geomspace(1e-6,1e6,1000);s=np.sqrt(r*r+B*B)
    # Enclosed effective mass fraction r^3/[s^2(s+L)] increases monotonically.
    assert np.all(np.diff(r**3/(s*s*(s+L)))>0);assert np.all(pair([[0,0,0]],[[0,0,0]],[1])[0,0,1:]==0)
    for k,v in err.items():assert v<(1e-6 if k in ['gradient','outer_ratio'] else 1e-10),(k,v)
    return err
def main():
    import cupy as cp
    out=P/'run001';out.mkdir(exist_ok=False);manifest=R/old.BASE/'source-bindings.json';config=json.loads(manifest.read_text());paths=[Path(__file__),P/'PREFLIGHT.md',manifest,R/old.BASE/'fields.csv',R/'scripts/mond_atlas_spatial_program.py',R/'scripts/mond_atlas_source_resolution.py']
    bindings={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for case in config['source_cases']:
        for comp in case['components']:
            assert hashlib.sha256((R/comp['path']).read_bytes()).hexdigest()==comp['sha256'];bindings[comp['path']]=comp['sha256']
    save(out/'bindings.json',bindings);save(out/'controls.json',controls(cp))
    params=[(r,t,z) for r in [1.,3.,6.] for z in [0.,.4] for t in np.arange(12)*2*np.pi/12];pts=np.array([[r*np.cos(t),r*np.sin(t),z] for r,t,z in params]);cache={};rows=[];masslog=[];start=time.monotonic();audit=[]
    for case in config['source_cases']:
        for level,(spacing,order) in enumerate([(.125,12),(.0625,24),(.03125,48)]):
            total=np.zeros((72,2,4))
            for comp in case['components']:
                key=(comp['path'],level)
                if key not in cache:
                    with np.load(R/comp['path']) as packet:xy,mass,expected=old.planar(packet,comp['conversion_to_msun_pc2'],spacing)
                    cache[key]=integrate(pts,xy,mass,comp['vertical_layers'],order,cp);masslog.append(dict(path=comp['path'],level=level,mass=float(mass.sum()),expected=expected,cells=len(mass)))
                    if len(audit)==0:
                        cpu=integrate(pts[[0,37]],xy,mass,comp['vertical_layers'],order,np);err=float(np.max(abs(cpu-cache[key][[0,37]]))/np.max(abs(cpu)));assert err<1e-10
                        rot=np.array([[0,-1],[1,0]]);rp=pts[[0]].copy();rp[:,:2]=rp[:,:2]@rot.T;rv=integrate(rp,xy@rot.T,mass,comp['vertical_layers'],order,cp);expectedv=cache[key][[0]].copy();expectedv[:,:,1:3]=expectedv[:,:,1:3]@rot.T;er=float(np.max(abs(rv-expectedv))/np.max(abs(expectedv)));assert er<1e-10;audit.append(dict(cpu_gpu=err,actual_source_rotation=er))
                result=cache[key];total+=result
                for model,mi in [('log_extra',0),('newton',1),('newton_plus_log',2)]:
                    values=result[:,mi] if mi<2 else result.sum(axis=1)
                    for i,(r,t,z) in enumerate(params):rows.append(dict(case=case['id'],level=level,component=comp['id'],model=model,r=r,theta=t,z=z,phi=float(values[i,0]),gx=float(values[i,1]),gy=float(values[i,2]),gz=float(values[i,3])))
            for model,mi in [('log_extra',0),('newton',1),('newton_plus_log',2)]:
                values=total[:,mi] if mi<2 else total.sum(axis=1)
                for i,(r,t,z) in enumerate(params):rows.append(dict(case=case['id'],level=level,component='total',model=model,r=r,theta=t,z=z,phi=float(values[i,0]),gx=float(values[i,1]),gy=float(values[i,2]),gz=float(values[i,3])))
            with (out/'fields.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
            elapsed=time.monotonic()-start;print(case['id'],level,elapsed,flush=True)
            if elapsed>300:save(out/'timeout.json',dict(elapsed=elapsed));raise RuntimeError('Frozen 300-second budget exceeded')
    save(out/'mass-checks.json',masslog);save(out/'run.json',dict(seconds=time.monotonic()-start,rows=len(rows),audit=audit,device=str(cp.cuda.runtime.getDeviceProperties(0)['name']),observed_response_scored=False))
if __name__=='__main__':main()
