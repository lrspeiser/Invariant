import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,csv
import numpy as np
R=Path(__file__).resolve().parents[4];P=Path(__file__).resolve().parent;OLD=R/'work/gravity-first-principles/mond-atlas-pressure-estimator-001';S=R/'work/gravity-first-principles/mond-atlas-source-sensitivity-001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def derivative(r,raw,targets):
    ell=.25;dr=r[1]-r[0];w=np.ones(len(r));w[1:-1:2]=4;w[2:-1:2]=2;w*=dr/3
    d=targets[:,None]-r;p=targets[:,None]+r
    k=lambda a:np.exp(-.5*(a/ell)**2)/(ell*np.sqrt(2*np.pi))
    return ((-d*k(d)-p*k(p))/ell**2)@(raw*w)
def main():
    dr=.00625;r=np.arange(1921)*dr;targets=np.array([0,.75,1.,2.5]);raw=np.exp(-r*r/(2*.7**2));width=np.sqrt(.7**2+.25**2);truth=-targets/width**2*.7/width*np.exp(-targets**2/(2*width**2));err=float(abs(derivative(r,raw,targets)-truth).max());assert err<1e-10
    assets=[a for a in json.loads((S/'run001/summary.json').read_text())['assets'] if a['component']=='atomic_helium' and a.get('quadrature_factor',1)==1]
    rows=list(csv.DictReader((OLD/'run001/comparison.csv').open()));results=[];theta=np.arange(2048)*2*np.pi/2048
    for a in assets:
        path=R/a['path'];assert sha(path)==a['sha256']
        with np.load(path) as z:axis=z['latent_axis'];surface=z['intrinsic_effective_surface']/1.36
        raw=np.zeros(len(r));n=len(axis)
        for start in range(0,len(r),64):
            rr=r[start:start+64,None];xx=(rr*np.cos(theta)-axis[0])/(axis[1]-axis[0]);yy=(rr*np.sin(theta)-axis[0])/(axis[1]-axis[0]);inside=(xx>=0)&(xx<=n-1)&(yy>=0)&(yy<=n-1);ix=np.clip(np.floor(xx).astype(int),0,n-2);iy=np.clip(np.floor(yy).astype(int),0,n-2);u=xx-ix;v=yy-iy
            values=(1-u)*(1-v)*surface[ix,iy]+u*(1-v)*surface[ix+1,iy]+(1-u)*v*surface[ix,iy+1]+u*v*surface[ix+1,iy+1];raw[start:start+len(rr)]=np.where(inside,values,0).mean(axis=1)
        offsets=np.arange(-320,321);weights=np.exp(-.5*(offsets*dr/.25)**2);weights/=weights.sum();extended=np.r_[raw[:0:-1],raw];smoothed=np.convolve(extended,weights,'same')[len(raw)-1:];fd=np.empty(len(raw));fd[1:-1]=(smoothed[2:]-smoothed[:-2])/(2*dr);fd[0]=(-3*smoothed[0]+4*smoothed[1]-smoothed[2])/(2*dr);fd[-1]=(3*smoothed[-1]-4*smoothed[-2]+smoothed[-3])/(2*dr)
        select=[x for x in rows if x['branch']==a['branch']];target=np.array([float(x['r_kpc']) for x in select]);index=np.rint(target/dr).astype(int);integrated=np.concatenate([derivative(r,raw,t) for t in np.array_split(target,8)]);den=raw[index];assert np.all(den>0)
        independent_fd=100*fd[index]/den;independent_integrated=100*integrated/den
        fd_error=float(abs(independent_fd-[float(x['sampled_pressure_acceleration']) for x in select]).max());ad_error=float(abs(independent_integrated-[float(x['integrated_pressure_acceleration']) for x in select]).max());assert fd_error<1e-7 and ad_error<1e-7
        delta=independent_fd-independent_integrated;worst=int(np.argmax(abs(delta)));results.append(dict(branch=a['branch'],rows=len(select),finite_difference_replay_error=fd_error,integrated_derivative_replay_error=ad_error,largest_difference_radius_kpc=float(target[worst]),raw_HI_at_largest_difference_msun_pc2=float(den[worst]),largest_difference_acceleration=float(abs(delta[worst])),units='(km/s)^2/kpc'))
    paths=[Path(__file__),P/'SCOPE.md',OLD/'PREFLIGHT.md',OLD/'run001/comparison.csv',OLD/'run001/summary.json',R/'scripts/mond_atlas_pressure_estimator.py',R/'scripts/mond_atlas_hi_column.py']+[R/a['path'] for a in assets]
    result=dict(status='INDEPENDENT_ALGEBRA_REPLAY_PASS',manufactured_derivative_error=err,results=results,observed_spectra_read=0,estimator_selected=False,bindings={str(x.relative_to(R)):sha(x) for x in paths})
    with (P/'receipt.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result['results']))
if __name__=='__main__':main()
