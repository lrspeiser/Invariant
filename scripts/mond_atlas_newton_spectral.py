"""Exact bilinear-source Fourier coefficients, analytic exponential vertical field."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.fft import fft2,ifft2,fftshift,ifftshift,fftfreq
from scipy.ndimage import map_coordinates
from scipy.integrate import quad
from scipy.special import j0,j1
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-newton-spectral-001';G=4.30091727003628e-6
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def vertical(k,z,h):
    a=abs(z);q=k*h;den=1-q*q;near=abs(q-1)<1e-7;safe=np.where(near,1,den);ea=np.exp(-k*a);eb=np.exp(-a/h)
    v=np.where(near,.5*(1+a/h)*eb,(ea-q*eb)/safe)
    d=np.where(near,-a*eb/(2*h*h),k*(eb-ea)/safe)*np.sign(z)
    return v,d
def field(coeff,width,points,layers):
    n=len(coeff);freq=2*np.pi*fftfreq(n,d=width/n);kx=freq[:,None];ky=freq[None,:];k=np.hypot(kx,ky);safe=np.where(k==0,1,k);out=np.zeros((len(points),3))
    for z in sorted(set(points[:,2])):
        v=np.zeros_like(k);d=np.zeros_like(k)
        for f,h in layers:
            vv,dd=vertical(k,z,h);v+=f*vv;d+=f*dd
        phi=-2*np.pi*G*coeff*v/safe;phi[0,0]=0
        gz=2*np.pi*G*coeff*d/safe;gz[0,0]=-2*np.pi*G*coeff[0,0]*sum(f*np.sign(z)*(1-np.exp(-abs(z)/h)) for f,h in layers)
        mask=points[:,2]==z;coord=(points[mask,:2].T*n/width)%n
        for j,spectral in enumerate([-1j*kx*phi,-1j*ky*phi,gz]):
            grid=ifft2(spectral,workers=1).real*n*n
            out[mask,j]=map_coordinates(grid,coord,order=3,mode='grid-wrap')
    return out
def coefficients(packet,conversion,width,dx):
    nodes=packet['latent_axis'];surf=packet['intrinsic_effective_surface'];h=nodes[1]-nodes[0];native=.03125;n=round(width/native);nf=round(width/dx)
    deposit=np.zeros((n,n));indices=np.rint(nodes/native).astype(int)%n;deposit[np.ix_(indices,indices)]=surf*conversion*1e6*h*h/width**2
    c=fft2(deposit,workers=1);del deposit
    if nf<n:
        center=fftshift(c);start=(n-nf)//2;c=ifftshift(center[start:start+nf,start:start+nf]).copy();del center
    freq=fftfreq(nf,d=dx);tent=np.sinc(freq*h)**2;c*=tent[:,None]*tent[None,:]
    expected=float(surf.sum()*conversion*1e6*h*h);assert abs(c[0,0].real*width**2/expected-1)<1e-10
    return c,expected
def controls():
    ve=[]
    for h in [.1,.4]:
        for k in [.1,1/h,3/h]:
            for z in [0.,.4]:
                num=quad(lambda t:np.exp(-abs(t)/h-k*abs(z-t))/(2*h),-np.inf,0,epsabs=1e-11)[0]+quad(lambda t:np.exp(-abs(t)/h-k*abs(z-t))/(2*h),0,z,epsabs=1e-11)[0]+quad(lambda t:np.exp(-abs(t)/h-k*abs(z-t))/(2*h),z,np.inf,epsabs=1e-11)[0]
                v,d=vertical(np.array(k),z,h);ve.append(abs(float(v)/num-1));assert ve[-1]<1e-8
    # Away from the exponential cusp, V''-k²V=-k/h exp(-|z|/h).
    k=.8;z=.7;h=.4;step=1e-4;v,d=vertical(np.array(k),z,h);vp=vertical(np.array(k),z+step,h)[0];vm=vertical(np.array(k),z-step,h)[0];res=float(abs(((vp-2*v+vm)/step**2-k*k*v)/(-k/h*np.exp(-z/h))-1));assert res<1e-5
    width=128.;n=1024;freq=2*np.pi*fftfreq(n,d=width/n);kk=np.hypot(freq[:,None],freq[None,:]);mass=1e9;coeff=mass/width**2*np.exp(-kk*kk/2);points=np.array([[r,0,z] for z in [0.,.4] for r in [.5,2,6]]);pred=field(coeff,width,points,[[1,.2]]);checks=[]
    for p,g in zip(points,pred):
        r,z=p[0],p[2];gr=-G*mass*quad(lambda k:k*np.exp(-k*k/2)*vertical(np.array(k),z,.2)[0]*j1(k*r),0,14,epsabs=1e-10)[0];gz=G*mass*quad(lambda k:np.exp(-k*k/2)*vertical(np.array(k),z,.2)[1]*j0(k*r),0,14,epsabs=1e-10)[0];truth=np.array([gr,0,gz]);err=float(np.linalg.norm(g-truth)/np.linalg.norm(truth));assert err<.002;checks.append(dict(r=r,z=z,relative_error=err))
    # Fourier source mode under rotation and reflection; no observational source.
    cc=np.zeros((64,64),complex);cc[1,0]=cc[-1,0]=.5;pts=np.array([[1.,2.,.4],[-1.,2.,.4]]);mode=field(cc,16,pts,[[1,.2]]);assert abs(mode[0,0]+mode[1,0])<1e-10 and abs(mode[0,2]-mode[1,2])<1e-10
    rotated=field(cc.T,16,pts[:,[1,0,2]],[[1,.2]]);assert np.max(abs(rotated-mode[:,[1,0,2]]))<1e-10
    return dict(vertical_convolution_max_relative=max(ve),mode_poisson_relative=res,gaussian_hankel=checks,rotation_reflection=True)
def main():
    out=P/'run001';out.mkdir(exist_ok=False);manifest=R/'work/gravity-first-principles/mond-atlas-spatial-program-001/source-bindings.json';config=json.loads(manifest.read_text());paths=[Path(__file__),P/'PREFLIGHT.md',manifest];bind={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for case in config['source_cases']:
        for c in case['components']:
            assert hashlib.sha256((R/c['path']).read_bytes()).hexdigest()==c['sha256'];bind[c['path']]=c['sha256']
    save(out/'bindings.json',bind);save(out/'controls.json',controls())
    params=[(r,t,z) for r in [1.,3.,6.] for z in [0.,.4] for t in np.arange(12)*2*np.pi/12];pts=np.array([[r*np.cos(t),r*np.sin(t),z] for r,t,z in params]);rows=[];cache={};masses=[];start=time.monotonic()
    for label,width,dx in [('box32',32.,.0625),('box64',64.,.0625),('box96',96.,.0625),('fine96',96.,.03125)]:
        for case in config['source_cases']:
            total=np.zeros((72,3))
            for c in case['components']:
                key=(c['path'],label)
                if key not in cache:
                    with np.load(R/c['path']) as packet:
                        assert np.allclose(packet['vertical_layers'],c['vertical_layers']);coeff,mass=coefficients(packet,c['conversion_to_msun_pc2'],width,dx)
                    cache[key]=field(coeff,width,pts,c['vertical_layers']);del coeff;masses.append(dict(path=c['path'],grid=label,mass_msun=mass))
                result=cache[key];total+=result
                for (r,t,z),v in zip(params,result):rows.append(dict(case=case['id'],grid=label,component=c['id'],r=r,theta=t,z=z,gx=float(v[0]),gy=float(v[1]),gz=float(v[2])))
            for (r,t,z),v in zip(params,total):rows.append(dict(case=case['id'],grid=label,component='total',r=r,theta=t,z=z,gx=float(v[0]),gy=float(v[1]),gz=float(v[2])))
        with (out/'fields.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        elapsed=time.monotonic()-start;print(label,elapsed,flush=True)
        if elapsed>300:save(out/'timeout.json',dict(seconds=elapsed));raise RuntimeError('Source-field time cap exceeded')
    save(out/'run.json',dict(seconds=time.monotonic()-start,rows=len(rows),masses=masses,observed_response_scored=False))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
