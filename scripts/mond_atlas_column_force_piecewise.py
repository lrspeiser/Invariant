"""HI-grid-aware angular integration and nonuniform radial source-force table."""
import csv,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.fft import fftshift,ifftshift
from scipy.ndimage import map_coordinates,spline_filter
from scipy.interpolate import PchipInterpolator
from threadpoolctl import threadpool_limits
import mond_atlas_column_force_v2 as m
R=m.R;P=m.P
def rule(radii,nodes,surface,order):
    q,w=np.polynomial.legendre.leggauss(order);angles=[];weights=[];sizes=[]
    for r in radii:
        t=nodes[abs(nodes)<r]/r;a=np.arccos(t);b=np.arcsin(t);events=np.unique(np.concatenate(([0.,2*np.pi],a,2*np.pi-a,b%(2*np.pi),(np.pi-b)%(2*np.pi))));lo,hi=events[:-1],events[1:];phi=((hi+lo)[:,None]/2+(hi-lo)[:,None]/2*q).ravel();ww=((hi-lo)[:,None]/2*np.broadcast_to(w,(len(lo),order))).ravel();angles.append(phi);weights.append(ww);sizes.append(len(phi))
    phi=np.concatenate(angles);weights=np.concatenate(weights);starts=np.r_[0,np.cumsum(sizes)[:-1]];rr=np.repeat(radii,sizes);x=rr*np.cos(phi);y=rr*np.sin(phi);h=nodes[1]-nodes[0];hi=map_coordinates(surface,[(x-nodes[0])/h,(y-nodes[0])/h],order=1,mode='constant',cval=0,prefilter=False);weights*=hi;den=np.add.reduceat(weights,starts)
    return dict(phi=phi,x=x,y=y,w=weights,starts=starts,den=den)
def main():
    out=P/'run004';out.mkdir(exist_ok=False);manifest=R/'work/gravity-first-principles/mond-atlas-spatial-program-001/source-bindings.json';cfg=json.loads(manifest.read_text());cases=[ca for ca in cfg['source_cases'] if ca['id'].startswith('f4')];comps={c['path']:c for ca in cases for c in ca['components']};paths=[Path(__file__),R/'scripts/mond_atlas_column_force_v2.py',P/'ANGULAR_SUPPORT_ADDENDUM.md',P/'PREFLIGHT.md',manifest];binding={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for p,c in comps.items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==c['sha256'];binding[p]=c['sha256']
    m.save(out/'bindings.json',binding);radii=np.unique(np.round(np.r_[np.arange(.05,.25001,.00025),np.arange(.25,5.95001,.00625),np.arange(5.95,6.02501,.00025)],8));hi=next(c for c in comps.values() if c['id']=='atomic_helium')
    with np.load(R/hi['path']) as p:nodes=p['latent_axis'];surface=p['intrinsic_effective_surface']/1.36
    start=time.monotonic();rules={order:rule(radii,nodes,surface,order) for order in [4,8]};zero=[float(r) for r,d in zip(radii,rules[8]['den']) if d<=0];m.save(out/'support.json',dict(radii=len(radii),zero_column_radii=zero,minimum_radius=float(radii[0]),maximum_radius=float(radii[-1])));assert not zero
    fields={};sig=rules[8]['den']/(2*np.pi)
    for label,dx,model in [('nc',.0625,'newton'),('nf',.03125,'newton'),('lc',.03125,'log'),('lf',.015625,'log')]:
        width=96. if model=='newton' else 32.;kernels={}
        for path,c in comps.items():
            hs=c['vertical_layers'][0][1]
            if hs not in kernels:kernels[hs]=m.kernel(width,dx,hs,model,128)
            with np.load(R/path) as packet:coeff,mass=m.source_coeff(packet,c['conversion_to_msun_pc2'],width,min(dx,.03125))
            if dx>.03125:
                n=round(width/dx);s=(len(coeff)-n)//2;coeff=ifftshift(fftshift(coeff)[s:s+n,s:s+n]).copy()
            grids=m.grids(coeff,kernels[hs]);splines=[spline_filter(a,order=3,mode='grid-wrap') for a in grids];del grids,coeff
            for order,ruledata in rules.items():
                d=ruledata;coord=[d['x']/dx%len(splines[0]),d['y']/dx%len(splines[0])];gx,gy=[map_coordinates(a,coord,order=3,mode='grid-wrap',prefilter=False) for a in splines];rad=-(gx*np.cos(d['phi'])+gy*np.sin(d['phi']));tan=-gx*np.sin(d['phi'])+gy*np.cos(d['phi']);avg=np.add.reduceat(d['w']*rad,d['starts'])/d['den'];variance=np.add.reduceat(d['w']*rad*rad,d['starts'])/d['den']-avg*avg;tang=np.add.reduceat(d['w']*tan*tan,d['starts'])/d['den'];fields[(path,label,order)]=(avg,np.sqrt(np.maximum(variance,0)),np.sqrt(np.maximum(tang,0)))
            del splines
        print(label,time.monotonic()-start,flush=True)
        if time.monotonic()-start>300:raise RuntimeError('Frozen300seconds cap exceeded')
    gates=[];rows=[]
    def gate(case,comp,name,a,b,rr):
        for scope,mask in [('full',np.ones(len(rr),bool)),('central',rr<=.25),('main',(rr>=.25)&(rr<=5.95)),('outer',rr>=5.95),('aperture',(rr>=.75)&(rr<=2.5))]:
            delta=a[mask]-b[mask];rms=float(np.linalg.norm(delta)/np.linalg.norm(b[mask]));worst=float(np.max(abs(delta/b[mask])));gates.append(dict(case=case,component=comp,comparison=name,scope=scope,rms=rms,max_point=worst,max_absolute=float(np.max(abs(delta))),passed=rms<.01 and worst<.03))
    for ca in cases:
        for component in [c['id'] for c in ca['components']]+['total']:
            paths=[c['path'] for c in ca['components'] if component=='total' or c['id']==component];get=lambda label,order:sum(fields[(p,label,order)][0] for p in paths)
            for label,co in [('nf','nc'),('lf','lc')]:
                fine=get(label,8);gate(ca['id'],component,label+'_grid',get(co,8),fine,radii);gate(ca['id'],component,label+'_angular',get(label,4),fine,radii)
                for lo,hi,step in [(.05,.25,.0005),(.25,5.95,.0125),(5.95,6.025,.0005)]:
                    mask=(radii>=lo)&(radii<=hi);rr=radii[mask];vv=fine[mask];coarse=np.isclose((rr-lo)/step,np.rint((rr-lo)/step));mid=~coarse;pred=PchipInterpolator(rr[coarse],vv[coarse])(rr[mid]);
                    # Piece-specific interpolation receipt, avoid empty scope masks.
                    delta=pred-vv[mid];rms=float(np.linalg.norm(delta)/np.linalg.norm(vv[mid]));worst=float(np.max(abs(delta/vv[mid])));gates.append(dict(case=ca['id'],component=component,comparison=label+'_radial',scope=f'{lo}:{hi}',rms=rms,max_point=worst,max_absolute=float(np.max(abs(delta))),passed=rms<.01 and worst<.03))
            n=get('nf',8);l=get('lf',8)
            for j,r in enumerate(radii):rows.append(dict(case=ca['id'],component=component,r_kpc=float(r),sigma_hi_msun_pc2=float(sig[j]),newton_gbar=float(n[j]),log_extra_gbar=float(l[j]),newton_plus_log_gbar=float(n[j]+l[j]),newton_grid_delta=float(n[j]-get('nc',8)[j]),log_grid_delta=float(l[j]-get('lc',8)[j]),newton_angular_delta=float(n[j]-get('nf',4)[j]),log_angular_delta=float(l[j]-get('lf',4)[j])))
    with gzip.open(out/'column-force-table.csv.gz','wt',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    m.save(out/'review.json',dict(gates=gates,passed=sum(g['passed'] for g in gates),total=len(gates),seconds=time.monotonic()-start,table_rows=len(rows),observed_spectra_accessed=False));print('gates',sum(g['passed'] for g in gates),len(gates))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
