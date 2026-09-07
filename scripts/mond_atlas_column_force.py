"""Actual source-map HI-weighted column forces; no observed spectra."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.fft import fft2,ifft2,fftfreq
from scipy.ndimage import map_coordinates
from scipy.special import roots_laguerre,j1
from scipy.integrate import quad
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-column-force-001';G=4.30091727003628e-6;HT=.2
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def difference(z,h,t=HT):
    if h==t:return (1+z/h)*np.exp(-z/h)/(2*h)
    return (h*np.exp(-z/h)-t*np.exp(-z/t))/(h*h-t*t)
def W(k,h,t=HT):return (1+k*h*t/(h+t))/((1+k*h)*(1+k*t))
def radial_log(r,h,order):
    nodes,weights=roots_laguerre(order);out=np.zeros_like(np.asarray(r,dtype=float))
    terms=[(h,weights*(1+nodes)/2)] if h==HT else [(h,weights*h*h/(h*h-HT*HT)),(HT,-weights*HT*HT/(h*h-HT*HT))]
    for scale,ww in terms:
        for z,w in zip(scale*nodes,ww):
            s=np.sqrt(r*r+z*z+.05**2);out+=w/(s*s*(s+4))
    return out
def source_coeff(packet,conversion,width,dx):
    nodes=packet['latent_axis'];surf=packet['intrinsic_effective_surface'];h=nodes[1]-nodes[0];n=round(width/dx)
    assert np.max(abs(nodes/dx-np.rint(nodes/dx)))<1e-10
    dep=np.zeros((n,n));idx=np.rint(nodes/dx).astype(int)%n;dep[np.ix_(idx,idx)]=surf*conversion*1e6*h*h/width**2;c=fft2(dep,workers=1);f=fftfreq(n,d=dx);t=np.sinc(f*h)**2;c*=t[:,None]*t[None,:];mass=float(surf.sum()*conversion*1e6*h*h);assert abs(c[0,0].real*width**2/mass-1)<1e-10
    return c,mass
def kernel(width,dx,h,model,order=0):
    n=round(width/dx);fq=2*np.pi*fftfreq(n,d=dx)
    if model=='newton':
        k=np.hypot(fq[:,None],fq[None,:]);factor=2*np.pi*G*W(k,h)/np.where(k==0,1,k);factor[0,0]=0
        return [1j*fq[:,None]*factor,1j*fq[None,:]*factor]
    x=fftfreq(n)*width;r=np.hypot(x[:,None],x[None,:]);grid=np.arange(0,float(r.max())+dx/8,dx/8);v=np.interp(r.ravel(),grid,radial_log(grid,h,order)).reshape(r.shape)
    a=-G*x[:,None]*v;b=-G*x[None,:]*v;a[n//2,:]=0;b[:,n//2]=0
    return [fft2(a,workers=1)*dx*dx,fft2(b,workers=1)*dx*dx]
def grids(c,ker):return [ifft2(c*k,workers=1).real*len(c)**2 for k in ker]
def controls():
    checks=[]
    for h in [.1,.2,.4]:
        norm=quad(lambda z:difference(z,h),0,np.inf,epsabs=1e-12)[0];assert abs(norm-1)<1e-10
        for k in [.1,1.,10.,100.]:
            num=quad(lambda z:difference(z,h)*np.exp(-k*z),0,np.inf,epsabs=1e-12)[0];assert abs(num/W(k,h)-1)<1e-10
        for r in [0.,.01,.05,.2,1.,6.]:
            exact=quad(lambda z:difference(z,h)/((r*r+z*z+.05**2)*(np.sqrt(r*r+z*z+.05**2)+4)),0,np.inf,epsabs=1e-11)[0];approx=float(radial_log(np.array(r),h,128));err=abs(approx/exact-1);assert err<.002;checks.append(dict(kind='log_vertical',h=h,r=r,relative_error=err))
    width=128.;dx=.125;n=round(width/dx);f=2*np.pi*fftfreq(n,d=dx);kk=np.hypot(f[:,None],f[None,:]);mass=1e9;c=mass/width**2*np.exp(-kk*kk/2);gg=grids(c,kernel(width,dx,.1,'newton'))
    for r in [.5,2.,6.]:
        val=map_coordinates(gg[0],np.array([[r/dx],[0.]]),order=3,mode='grid-wrap')[0];exact=-G*mass*quad(lambda k:k*np.exp(-k*k/2)*W(k,.1)*j1(k*r),0,14,epsabs=1e-11)[0];err=abs(val/exact-1);assert err<.002;checks.append(dict(kind='newton_hankel',r=r,relative_error=err))
    width=16.;dx=.03125;n=512;deposit=np.zeros((n,n));sources=[(-.5,.25,1e8),(.5,-.25,2e8)]
    for x,y,m in sources:deposit[round(x/dx)%n,round(y/dx)%n]+=m/width**2
    c=fft2(deposit);gg=grids(c,kernel(width,dx,.2,'log',128))
    for x,y in [(1.,0.),(2.,1.)]:
        pred=np.array([map_coordinates(a,np.array([[x/dx],[y/dx]]),order=3,mode='grid-wrap')[0] for a in gg]);truth=np.zeros(2)
        for sx,sy,m in sources:
            d=np.array([x-sx,y-sy]);r=np.linalg.norm(d);v=quad(lambda z:difference(z,.2)/((r*r+z*z+.05**2)*(np.sqrt(r*r+z*z+.05**2)+4)),0,np.inf,epsabs=1e-11)[0];truth-=G*m*d*v
        err=float(np.linalg.norm(pred-truth)/np.linalg.norm(truth));assert err<.002;checks.append(dict(kind='direct_compact_log',x=x,y=y,relative_error=err))
    assert np.max(abs(gg[0].T-grids(c.T,kernel(width,dx,.2,'log',128))[1]))<1e-9
    return checks
def main():
    out=P/'run001';out.mkdir(exist_ok=False);manifest=R/'work/gravity-first-principles/mond-atlas-spatial-program-001/source-bindings.json';cfg=json.loads(manifest.read_text());cases=[c for c in cfg['source_cases'] if c['id'].startswith('f4')];paths=[Path(__file__),P/'PREFLIGHT.md',manifest,R/'work/gravity-first-principles/mond-atlas-observation-admission-001/first-score-protocol.json'];bindings={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};comps={c['path']:c for ca in cases for c in ca['components']}
    for path,c in comps.items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==c['sha256'];bindings[path]=c['sha256']
    save(out/'bindings.json',bindings);save(out/'controls.json',controls())
    hi=next(c for c in comps.values() if c['id']=='atomic_helium');hp=np.load(R/hi['path']);nodes=hp['latent_axis'];h=nodes[1]-nodes[0];surface=hp['intrinsic_effective_surface']/1.36;radii=np.arange(1,241)*.025;theta=np.arange(2048)*2*np.pi/2048;x=radii[:,None]*np.cos(theta);y=radii[:,None]*np.sin(theta);coords=np.array([(x-nodes[0])/h,(y-nodes[0])/h]);weight=map_coordinates(surface,coords,order=1,mode='constant',cval=0,prefilter=False);assert np.all(weight.mean(axis=1)>0);hp.close()
    configs=[('n64',64.,.0625,'newton',0),('n96',96.,.0625,'newton',0),('nf96',96.,.03125,'newton',0),('l32early',32.,.0625,'log',32),('l32mid',32.,.03125,'log',64),('l32vertical',32.,.03125,'log',128),('l32fine',32.,.015625,'log',128),('l48box',48.,.03125,'log',128)]
    rows=[];massrows=[];start=time.monotonic()
    for label,width,dx,model,order in configs:
        ker_cache={};component_cache={}
        for path,c in comps.items():
            hs=c['vertical_layers'][0][1];assert len(c['vertical_layers'])==1 and c['vertical_layers'][0][0]==1
            if hs not in ker_cache:ker_cache[hs]=kernel(width,dx,hs,model,order)
            # Coarse field spacing may not contain every f4 source node; construct at native then crop exact modes.
            native=min(dx,.03125)
            with np.load(R/path) as packet:coeff,mass=source_coeff(packet,c['conversion_to_msun_pc2'],width,native)
            if native<dx:
                from scipy.fft import fftshift,ifftshift
                n=round(width/dx);big=len(coeff);s=(big-n)//2;coeff=ifftshift(fftshift(coeff)[s:s+n,s:s+n]).copy()
            gg=grids(coeff,ker_cache[hs]);n=len(coeff);sample=[map_coordinates(a,np.array([x*n/width%n,y*n/width%n]),order=3,mode='grid-wrap') for a in gg];radial=-(sample[0]*np.cos(theta)+sample[1]*np.sin(theta));tangent=-sample[0]*np.sin(theta)+sample[1]*np.cos(theta);component_cache[path]=(radial,tangent);massrows.append(dict(path=path,grid=label,mass_msun=mass));del coeff,gg,sample
        for ca in cases:
            total_r=sum(component_cache[c['path']][0] for c in ca['components']);total_t=sum(component_cache[c['path']][1] for c in ca['components'])
            for comp,rad,tan in [(c['id'],*component_cache[c['path']]) for c in ca['components']]+[('total',total_r,total_t)]:
                for nt in [512,1024,2048]:
                    step=2048//nt;w=weight[:,::step];rr=rad[:,::step];tt=tan[:,::step];den=w.sum(axis=1);avg=(w*rr).sum(axis=1)/den;spread=np.sqrt((w*(rr-avg[:,None])**2).sum(axis=1)/den);tang=np.sqrt((w*tt*tt).sum(axis=1)/den)
                    for i,r in enumerate(radii):rows.append(dict(case=ca['id'],grid=label,model=model,component=comp,azimuth_nodes=nt,r_kpc=float(r),sigma_hi_msun_pc2=float(w[i].mean()),gbar_inward=float(avg[i]),radial_rms_deviation=float(spread[i]),tangential_rms=float(tang[i])))
        with (out/'radial-fields.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        elapsed=time.monotonic()-start;print(label,elapsed,flush=True)
        if elapsed>300:save(out/'timeout.json',dict(seconds=elapsed));raise RuntimeError('Frozen resource cap exceeded')
    save(out/'run.json',dict(seconds=time.monotonic()-start,rows=len(rows),mass_checks=massrows,observed_spectra_accessed=False,force_units='(km/s)^2/kpc',vertical_target_height_kpc=.2))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
