"""Raw alternative HI columns and the prescribed pressure closure."""
import csv,gzip,json,hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates,gaussian_filter1d
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-source-sensitivity-001'
def main():
    out=P/'hi001';out.mkdir(exist_ok=False);assets=json.loads((P/'run001/summary.json').read_text())['assets'];assets=[a for a in assets if a['component']=='atomic_helium' and a.get('quadrature_factor',1)==1];binding={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),P/'HI_PRESSURE_PREFLIGHT.md']}
    for a in assets:assert hashlib.sha256((R/a['path']).read_bytes()).hexdigest()==a['sha256'];binding[a['path']]=a['sha256']
    (out/'bindings.json').write_text(json.dumps(binding,indent=2)+'\n');rows=[];gates=[];tails=[]
    for a in assets:
        with np.load(R/a['path']) as p:nodes=p['latent_axis'];s=p['intrinsic_effective_surface']/1.36
        profiles=[]
        for dr in [.0125,.00625]:
            r=np.arange(round(12/dr)+1)*dr;theta=np.arange(2048)*2*np.pi/2048;raw=np.zeros(len(r))
            for start in range(0,len(r),128):
                rr=r[start:start+128,None];x=rr*np.cos(theta);y=rr*np.sin(theta);sample=map_coordinates(s,[(x-nodes[0])/(nodes[1]-nodes[0]),(y-nodes[0])/(nodes[1]-nodes[0])],order=1,mode='constant',cval=0,prefilter=False);raw[start:start+len(rr)]=sample.mean(axis=1)
            mirrored=np.r_[raw[:0:-1],raw];sm=gaussian_filter1d(mirrored,.25/dr,mode='constant',cval=0,truncate=8)[len(raw)-1:];der=np.gradient(sm,dr,edge_order=2);pressure=np.divide(100*der,raw,out=np.zeros_like(raw),where=raw>0);profiles.append((r,raw,sm,der,pressure));support=(r>=.05)&(r<=6.025+1e-9)
            for j in np.flatnonzero(support):rows.append(dict(branch=a['branch'],radial_spacing_kpc=dr,r_kpc=float(r[j]),raw_sigma_hi=float(raw[j]),smoothed_sigma_hi=float(sm[j]),smoothed_gradient=float(der[j]),pressure_acceleration_sigma10=float(pressure[j]) if raw[j]>0 else '',positive_raw_column=bool(raw[j]>0)))
            if dr==.00625:
                mass=np.trapezoid(2*np.pi*r*raw,r)*1e6;tail=np.trapezoid((2*np.pi*r*raw)[r>=6],r[r>=6])*1e6;tails.append(dict(branch=a['branch'],radial_integral_hi_mass=mass,radial_integral_fraction_outside6=tail/mass,interpretation='Numerical radial integral, not a delivered-aperture flux bound'))
        r,raw,_,_,co=profiles[0];fine=profiles[1][4][::2]
        for scope,mask in [('full',(r>=.05)&(r<=6.025)&(raw>0)),('aperture',(r>=.75)&(r<=2.5)&(raw>0))]:
            delta=co[mask]-fine[mask];rms=float(np.linalg.norm(delta)/np.linalg.norm(fine[mask]));point=float(np.max(abs(delta/fine[mask])));gates.append(dict(branch=a['branch'],scope=scope,rms=rms,max_point=point,passed=rms<.01 and point<.03))
    with gzip.open(out/'pressure-profiles.csv.gz','wt',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(dict(gates=gates,tails=tails,observed_spectra_accessed=False,closure='Pi=sigma_reference²*Gaussian.25(rawHI); pressure acceleration=dPi/dR/rawHI; scale sigma10 column by.25 or2.25 for5/15',emission_transport_completed=False),indent=2)+'\n');print(json.dumps(gates));print(json.dumps(tails))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
