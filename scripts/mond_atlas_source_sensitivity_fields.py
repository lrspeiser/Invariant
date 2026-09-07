"""Conservative column-force replay using each new physical HI source weight."""
import csv,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.fft import fftshift,ifftshift
from scipy.ndimage import map_coordinates
from threadpoolctl import threadpool_limits
import mond_atlas_column_force_v2 as m
R=m.R;P=R/'work/gravity-first-principles/mond-atlas-source-sensitivity-001'
def main():
    out=P/'fields001';out.mkdir(exist_ok=False);summary=json.loads((P/'run001/summary.json').read_text());assets=summary['assets'];cases={}
    for name,branch,hs in [('common30','common30_gaussian_approximation',.1),('common30_stars_h0.4','common30_gaussian_approximation',.4),('missing_zero','missing_zero',.1),('missing_annular','missing_annular',.1)]:
        cases[name]=[next(a for a in assets if a['branch']==branch and a['component']==c and a['height_kpc']==(hs if c=='stellar_luminosity' else .2) and a.get('quadrature_factor',1)==1) for c in ['stellar_luminosity','atomic_helium','co21']]
    paths=[Path(__file__),P/'FIELD_PREFLIGHT.md',P/'run001/summary.json',P/'run001/independent-review.json',R/'scripts/mond_atlas_column_force_v2.py'];bindings={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};unique={a['path']:a for aa in cases.values() for a in aa}
    for p,a in unique.items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==a['sha256'];bindings[p]=a['sha256']
    m.save(out/'bindings.json',bindings);radii=np.arange(2,242)*.025;phi=np.arange(2048)*2*np.pi/2048;x=radii[:,None]*np.cos(phi);y=radii[:,None]*np.sin(phi);weights={};profiles=[]
    for name,aa in cases.items():
        hi=next(a for a in aa if a['component']=='atomic_helium')
        with np.load(R/hi['path']) as p:nodes=p['latent_axis'];s=p['intrinsic_effective_surface']/1.36
        w=map_coordinates(s,[(x-nodes[0])/(nodes[1]-nodes[0]),(y-nodes[0])/(nodes[1]-nodes[0])],order=1,mode='constant',cval=0,prefilter=False);weights[name]=w
        for rr,mean in zip(radii,w.mean(axis=1)):profiles.append(dict(case=name,r_kpc=float(rr),sigma_hi_msun_pc2=float(mean),positive_column=bool(mean>0)))
    rows=[];start=time.monotonic()
    for label,dx,model in [('nc',.0625,'newton'),('nf',.03125,'newton'),('lc',.03125,'log_extra'),('lf',.015625,'log_extra')]:
        width=96. if model=='newton' else 32.;ker={};fields={}
        for path,a in unique.items():
            hs=a['height_kpc']
            if hs not in ker:ker[hs]=m.kernel(width,dx,hs,'newton' if model=='newton' else 'log',128)
            with np.load(R/path) as p:c,mass=m.source_coeff(p,a['conversion_to_msun_pc2'],width,min(dx,.03125))
            if dx>.03125:
                n=round(width/dx);starti=(len(c)-n)//2;c=ifftshift(fftshift(c)[starti:starti+n,starti:starti+n]).copy()
            grids=m.grids(c,ker[hs]);sample=[map_coordinates(v,[x/dx%len(v),y/dx%len(v)],order=3,mode='grid-wrap') for v in grids];fields[path]=(-(sample[0]*np.cos(phi)+sample[1]*np.sin(phi)),-sample[0]*np.sin(phi)+sample[1]*np.cos(phi));del grids,c,sample
        for name,aa in cases.items():
            values=[(a['component'],*fields[a['path']]) for a in aa]+[('total',sum(fields[a['path']][0] for a in aa),sum(fields[a['path']][1] for a in aa))]
            for component,rad,tan in values:
                for nt in [1024,2048]:
                    step=2048//nt;w=weights[name][:,::step];rr=rad[:,::step];tt=tan[:,::step];den=w.sum(axis=1);valid=den>0;mean=np.divide((w*rr).sum(axis=1),den,out=np.zeros(len(radii)),where=valid);spread=np.sqrt(np.divide((w*(rr-mean[:,None])**2).sum(axis=1),den,out=np.zeros(len(radii)),where=valid));tang=np.sqrt(np.divide((w*tt*tt).sum(axis=1),den,out=np.zeros(len(radii)),where=valid))
                    for j,r in enumerate(radii):rows.append(dict(case=name,grid=label,model=model,component=component,azimuth_nodes=nt,r_kpc=float(r),gbar_inward=float(mean[j]) if valid[j] else '',radial_rms_deviation=float(spread[j]) if valid[j] else '',tangential_rms=float(tang[j]) if valid[j] else '',positive_hi_column=bool(valid[j])))
        print(label,time.monotonic()-start,flush=True)
        if time.monotonic()-start>300:raise RuntimeError('Frozen time budget exceeded')
    with gzip.open(out/'radial-fields.csv.gz','wt',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    with (out/'hi-column-profiles.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=profiles[0]);w.writeheader();w.writerows(profiles)
    m.save(out/'source-contract.json',dict(cases=cases,pressure='Recompute Pi from this alternative raw HI column with frozen.25kpc radial smoothing and rawHI denominator; do not reuse baselinePi.',emission='Recompute physical projected source cells/centroids and native beam transport from this alternative HI packet; retain expanded tails.',source_effective_beam_warning='Additional30arcsec Gaussian-source approximation does not remove native beam already embedded in baseline; delivered cube beam remains a separate operator. No exact deconvolution or exact common non-Gaussian PSF claim.',observed_spectra_accessed=False))
    m.save(out/'run.json',dict(seconds=time.monotonic()-start,rows=len(rows),observed_spectra_accessed=False))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
