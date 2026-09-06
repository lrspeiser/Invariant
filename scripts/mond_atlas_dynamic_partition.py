"""Fixed-normalization kinetic coupling and strict source-partition controls."""
import csv,hashlib,itertools,json,time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from mond_atlas_dynamic_program import ROOT,setup

OUT=ROOT/'work/gravity-first-principles/mond-atlas-dynamic-program-001/partition-repair'


def mechanics(x,v,m,kind='corrected'):
    n=len(m);i,j=np.triu_indices(n,1);d=x[i]-x[j];dv=v[i]-v[j];r2=np.sum(d*d,axis=1);mm=m[i]*m[j]
    fp=-mm[:,None]*d/(r2+.05**2)[:,None]**1.5;V=-np.sum(mm/np.sqrt(r2+.05**2))
    mat=np.diag(m.copy());T=.5*np.sum(m[:,None]*v*v);Tk=0.
    if kind!='newton':
        coefficient=mm if kind=='corrected' else mm/(m[i]+m[j]);k=.5*coefficient*np.exp(-r2/2);grad=-k[:,None]*d
        fp+=.5*grad*np.sum(dv*dv,axis=1)[:,None]-np.sum(grad*dv,axis=1)[:,None]*dv
        for ii,jj,kk in zip(i,j,k):mat[ii,ii]+=kk;mat[jj,jj]+=kk;mat[ii,jj]-=kk;mat[jj,ii]-=kk
        Tk=.5*np.sum(k*np.sum(dv*dv,axis=1));T+=Tk
    F=np.zeros_like(x);np.add.at(F,i,fp);np.add.at(F,j,-fp);a=np.linalg.solve(mat,F);p=mat@v
    return a,float(T+V),p,np.sum(np.cross(x,p),axis=0),mat,float(T),float(Tk)


def controls():
    rng=np.random.default_rng(913);x=rng.normal(size=(4,3));v=.3*rng.normal(size=(4,3));m=np.array([.7,1,1.4,2.]);rows=[]
    for kind in ['corrected','old_reduced_mass']:
        a,E,p,J,mat,T,Tk=mechanics(x,v,m,kind)
        for fractions in [[.5,.5],[.3,.7]]:
            xx=np.repeat(x,2,axis=0);vv=np.repeat(v,2,axis=0);mm=(m[:,None]*fractions).ravel()
            aa,ee,pp,jj,mt,tt,tkk=mechanics(xx,vv,mm,kind)
            # The introduced within-parent soft pair energy is a constant at coincidence.
            offset=-np.sum(m*m*np.prod(fractions)/.05)
            row=dict(kind=kind,fractions=fractions,kinetic_relative=abs(tt/T-1),extra_kinetic_ratio=tkk/Tk,canonical_momentum_max_abs=float(np.max(abs(pp.reshape(4,2,3).sum(axis=1)-p))),child_acceleration_max_abs=float(np.max(abs(aa-np.repeat(a,2,axis=0)))),aggregate_acceleration_max_abs=float(np.max(abs(np.sum(aa.reshape(4,2,3)*np.array(fractions)[None,:,None],axis=1)-a))),self_energy_adjusted_error=abs(ee-offset-E))
            if kind=='corrected':assert max(row['kinetic_relative'],row['canonical_momentum_max_abs'],row['child_acceleration_max_abs'],row['self_energy_adjusted_error'])<1e-10
            rows.append(row)
    # Independent canonical momentum finite difference and coordinate L gradient.
    a,E,p,J,mat,T,Tk=mechanics(x,v,m);step=1e-5
    def direct(xx,vv):
        t=.5*np.sum(m[:,None]*vv*vv);pot=0.;mom=m[:,None]*vv
        for i,j in itertools.combinations(range(4),2):
            d=xx[i]-xx[j];dv=vv[i]-vv[j];k=.5*m[i]*m[j]*np.exp(-d@d/2)
            t+=.5*k*(dv@dv);pot-=m[i]*m[j]/np.sqrt(d@d+.05**2);mom[i]+=k*dv;mom[j]-=k*dv
        return t-pot,mom
    dp=(direct(x+step*v,v+step*a)[1]-direct(x-step*v,v-step*a)[1])/(2*step);dx=np.zeros_like(x)
    for i in range(4):
        for j in range(3):
            delta=np.zeros_like(x);delta[i,j]=step;dx[i,j]=(direct(x+delta,v)[0]-direct(x-delta,v)[0])/(2*step)
    residual=float(np.max(abs(dp-dx)));rate=abs((mechanics(x+step*v,v+step*a,m)[1]-mechanics(x-step*v,v-step*a,m)[1])/(2*step))/max(1,abs(E))
    Q=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    covariance=max(np.max(abs(mechanics(x+[1,2,3],v+[2,-3,1],m)[0]-a)),np.max(abs(mechanics(x@Q.T,v@Q.T,m)[0]-a@Q.T)))
    minimum=float(np.linalg.eigvalsh(mat).min());assert max(residual,rate,covariance)<1e-6 and minimum>0
    return dict(partition=rows,euler_lagrange_residual=residual,energy_derivative=rate,covariance_error=float(covariance),minimum_inertia_eigenvalue=minimum)


def integrate(case,kind,duration_override=None):
    x,v,m,duration=setup(case);duration=duration if duration_override is None else duration_override;n=len(m);y0=np.concatenate([x.ravel(),v.ravel()]);results=[];solutions=[];timeseries=[]
    def rhs(t,y):
        xx=y[:3*n].reshape(n,3);vv=y[3*n:].reshape(n,3);a=mechanics(xx,vv,m,kind)[0];return np.r_[vv.ravel(),a.ravel()]
    for resolution,step,rtol,atol in [('base',.1,1e-8,1e-10),('fine',.05,1e-10,1e-12)]:
        sol=solve_ivp(rhs,[0,duration],y0,method='DOP853',t_eval=np.linspace(0,duration,401),max_step=step,rtol=rtol,atol=atol);assert sol.success
        energies=[];momenta=[];angular=[];radii=[];inward=[];closest=[]
        for tt,y in zip(sol.t,sol.y.T):
            xx=y[:3*n].reshape(n,3);vv=y[3*n:].reshape(n,3);a,e,p,j,mat,t,tk=mechanics(xx,vv,m,kind);r=np.linalg.norm(xx[1:]-xx[0],axis=1)
            g=-np.sum((a[1:]-a[0])*(xx[1:]-xx[0]),axis=1)/r
            energies.append(e);momenta.append(p.sum(axis=0));angular.append(j);radii.extend(r);inward.append(g.mean());closest.append(min(np.linalg.norm(xx[i]-xx[j]) for i,j in itertools.combinations(range(n),2)))
            if resolution=='fine':timeseries.append(dict(case=case,kind=kind,duration=duration,time=float(tt),energy=e,extra_kinetic_energy=tk,mean_radius=float(r.mean()),mean_inward_acceleration=float(g.mean())))
        energies=np.array(energies);momenta=np.array(momenta);angular=np.array(angular)
        ed=float(max(abs(energies-energies[0]))/max(abs(energies[0]),1e-10));pd=float(max(np.linalg.norm(momenta-momenta[0],axis=1))/max(1,np.sum(m[:,None]*abs(v))));jd=float(max(np.linalg.norm(angular-angular[0],axis=1))/max(1,np.linalg.norm(angular[0])))
        results.append(dict(case=case,kind=kind,duration=duration,resolution=resolution,nfev=sol.nfev,energy_drift=ed,momentum_drift=pd,angular_drift=jd,conservation_pass=max(ed,pd,jd)<1e-5,min_radius=float(min(radii)),max_radius=float(max(radii)),closest_pair=float(min(closest)),mean_inward_acceleration=float(np.mean(inward)),initial_energy=float(energies[0])))
        solutions.append(sol)
    error=float(np.linalg.norm(solutions[0].y[:3*n,-1]-solutions[1].y[:3*n,-1])/max(1,np.linalg.norm(solutions[1].y[:3*n,-1])))
    results[-1].update(trajectory_resolution_relative=error,trajectory_converged=error<1e-3)
    return results,timeseries


def run():
    dest=OUT/'run001';dest.mkdir(parents=True,exist_ok=False);started=time.time()
    bindings={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),OUT/'PREFLIGHT.md',ROOT/'scripts/mond_atlas_dynamic_program.py']}
    (dest/'bindings.json').write_text(json.dumps(bindings,indent=2),encoding='utf-8')
    (dest/'pre-integration-controls.json').write_text(json.dumps(controls(),indent=2),encoding='utf-8')
    rows=[];samples=[]
    cases=[('nested_co',None),('nested_counter',None),('clustered',None),('flyby',None),('clustered',2.)]
    for case,T in cases:
        for kind in ['newton','corrected']:
            r,s=integrate(case,kind,T);rows.extend(r);samples.extend(s);print(case,kind,T,r[-1]['trajectory_resolution_relative'],r[-1]['energy_drift'],flush=True)
    (dest/'results.json').write_text(json.dumps(dict(status='THEORY_BENCHMARK_ONLY',integrations=len(rows),seconds=time.time()-started,results=rows),indent=2),encoding='utf-8')
    with (dest/'timeseries.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=samples[0]);writer.writeheader();writer.writerows(samples)


if __name__=='__main__':run()
