"""Read-only node alignment and Nyquist-strip audit of the frozen spectral run."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.fft import fftfreq,ifft2
from scipy.ndimage import map_coordinates
from threadpoolctl import threadpool_limits
from mond_atlas_newton_spectral import coefficients,vertical,G
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-newton-spectral-001';O=P/'nyquist-review'
def main():
    O.mkdir(exist_ok=False);config=json.loads((R/'work/gravity-first-principles/mond-atlas-spatial-program-001/source-bindings.json').read_text());unique={c['path']:c for ca in config['source_cases'] for c in ca['components']}
    with (P/'run001/fields.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    params=[(r,t,z) for r in [1.,3.,6.] for z in [0.,.4] for t in np.arange(12)*2*np.pi/12];pts=np.array([[r*np.cos(t),r*np.sin(t),z] for r,t,z in params]);alignment=[];records=[];changes={}
    for path,c in unique.items():
        with np.load(R/path) as packet:
            nodes=packet['latent_axis'];res=float(np.max(abs(nodes/.03125-np.rint(nodes/.03125))));assert res==0;alignment.append(dict(path=path,max_lattice_residual=res))
            for label,dx in [('box96',.0625),('fine96',.03125)]:
                coeff,mass=coefficients(packet,c['conversion_to_msun_pc2'],96.,dx);n=len(coeff);fq=2*np.pi*fftfreq(n,d=dx);kx=fq[:,None];ky=fq[None,:];k=np.hypot(kx,ky);safe=np.where(k==0,1,k);reverse=(-np.arange(n))%n
                correction=np.zeros((72,3));leak=[]
                for z in [0.,.4]:
                    v=np.zeros_like(k);d=np.zeros_like(k)
                    for f,h in c['vertical_layers']:
                        vv,dd=vertical(k,z,h);v+=f*vv;d+=f*dd
                    phi=-2*np.pi*G*coeff*v/safe;phi[0,0]=0;gz=2*np.pi*G*coeff*d/safe;gz[0,0]=-2*np.pi*G*coeff[0,0]*sum(f*np.sign(z)*(1-np.exp(-abs(z)/h)) for f,h in c['vertical_layers']);mask=pts[:,2]==z;coord=(pts[mask,:2].T*n/96)%n
                    for j,s in enumerate([-1j*kx*phi,-1j*ky*phi,gz]):
                        conjugate=np.conj(s[np.ix_(reverse,reverse)]);realpart=(s+conjugate)/2;imagpart=(s-conjugate)/2;den=np.linalg.norm(realpart);leak.append(dict(z=z,axis=j,imaginary_to_real_rms=float(np.linalg.norm(imagpart)/den) if den else 0.))
                        strip=np.zeros_like(s);strip[n//2,:]=s[n//2,:];strip[:,n//2]=s[:,n//2]
                        grid=ifft2(strip,workers=1).real*n*n;correction[mask,j]=map_coordinates(grid,coord,order=3,mode='grid-wrap')
                # Removing boundary strips is a diagnostic of cutoff convention, not a physical refit.
                changes[(path,label)]=correction
                case=next(ca for ca in config['source_cases'] if any(cc['path']==path for cc in ca['components']));saved=np.array([[float(r[x]) for x in ['gx','gy','gz']] for r in rows if r['case']==case['id'] and r['component']==c['id'] and r['grid']==label]);rms=float(np.linalg.norm(correction)/np.linalg.norm(saved));worst=float(np.max(np.linalg.norm(correction,axis=1)/np.linalg.norm(saved,axis=1)))
                records.append(dict(path=path,grid=label,leakage=leak,strip_removal_sample_rms=rms,strip_removal_sample_max=worst));print(path.split('/')[-1],label,rms,flush=True)
    totals=[]
    for ca in config['source_cases']:
        for label in ['box96','fine96']:
            change=sum(changes[(c['path'],label)] for c in ca['components']);saved=np.array([[float(r[x]) for x in ['gx','gy','gz']] for r in rows if r['case']==ca['id'] and r['component']=='total' and r['grid']==label]);totals.append(dict(case=ca['id'],grid=label,strip_removal_rms=float(np.linalg.norm(change)/np.linalg.norm(saved)),strip_removal_max=float(np.max(np.linalg.norm(change,axis=1)/np.linalg.norm(saved,axis=1)))))
    # Algebraic projection identity checked independently on a non-Hermitian small spectrum.
    rng=np.random.default_rng(41);s=rng.normal(size=(16,16))+1j*rng.normal(size=(16,16));i=(-np.arange(16))%16;hs=(s+np.conj(s[np.ix_(i,i)]))/2;identity=float(np.max(abs(ifft2(hs).real-ifft2(s).real)));assert identity<1e-14
    result=dict(status='AUDIT_COMPLETE_NO_FROZEN_OUTPUT_CHANGED',alignment=alignment,records=records,totals=totals,hermitian_projection_real_field_identity_error=identity,max_imaginary_to_real_rms=max(q['imaginary_to_real_rms'] for r in records for q in r['leakage']),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (O/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(totals))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
