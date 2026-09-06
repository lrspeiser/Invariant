"""Conservative finite-particle kinetics and explicit pair oscillators."""
import itertools,json,time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/gravity-first-principles/mond-atlas-dynamic-program-001'


def mechanics(x,v,m,kind,q=None,w=None):
    n=len(m);pairs=np.array(list(itertools.combinations(range(n),2)));i,j=pairs.T
    d=x[i]-x[j];dv=v[i]-v[j];r2=np.sum(d*d,axis=1);mm=m[i]*m[j]
    fpair=-mm[:,None]*d/(r2+.05**2)[:,None]**1.5
    V=-np.sum(mm/np.sqrt(r2+.05**2));matrix=np.diag(m.copy());T=.5*np.sum(m[:,None]*v*v)
    oscillator_energy=0.;qw=None;h=.15*np.sqrt(mm)*np.exp(-r2/2)
    if kind=='kinetic':
        k=.5*mm/(m[i]+m[j])*np.exp(-r2/2);grad=-k[:,None]*d
        fpair+=.5*grad*np.sum(dv*dv,axis=1)[:,None]-np.sum(grad*dv,axis=1)[:,None]*dv
        for ii,jj,kk in zip(i,j,k):matrix[ii,ii]+=kk;matrix[jj,jj]+=kk;matrix[ii,jj]-=kk;matrix[jj,ii]-=kk
        T+=.5*np.sum(k*np.sum(dv*dv,axis=1))
    elif kind.startswith('memory'):
        omega=.2 if kind=='memory_slow' else 2.
        fpair+=omega**2*(q-h)[:,None]*(-h[:,None]*d)
        qw=-omega**2*(q-h);oscillator_energy=.5*np.sum(w*w+omega**2*(q-h)**2)
    forces=np.zeros_like(x);np.add.at(forces,i,fpair);np.add.at(forces,j,-fpair)
    acceleration=np.linalg.solve(matrix,forces)
    momentum=matrix@v;angular=np.cross(x,momentum).sum(axis=0)
    return acceleration,float(T+V+oscillator_energy),momentum.sum(axis=0),angular,qw,h,float(oscillator_energy),matrix


def setup(case):
    if case=='flyby':
        m=np.array([1.,.1]);x=np.array([[-3.,0,0],[3,.8,0]]);v=np.array([[.1,0,0],[-1.,0,0]]);duration=12.
    else:
        m=np.array([10.,.05,.05,.05,.05]);x=np.array([[0.,0,0],[1,0,0],[-1,0,0],[0,2,0],[0,-2,0]])
        if case=='clustered':x=np.array([[0.,0,0],[1,.15,0],[1,-.15,0],[-1,.15,0],[-1,-.15,0]])
        r=np.linalg.norm(x[1:],axis=1);speed=np.sqrt(10*r*r/(r*r+.05**2)**1.5)
        v=np.zeros_like(x);v[1:,0]=-x[1:,1]*speed/r;v[1:,1]=x[1:,0]*speed/r
        if case=='nested_counter':v[3:]*=-1
        duration=40.
    x-=np.sum(m[:,None]*x,axis=0)/m.sum();v-=np.sum(m[:,None]*v,axis=0)/m.sum()
    return x,v,m,duration


def rhs(m,kind):
    n=len(m);npair=n*(n-1)//2
    def func(t,state):
        x=state[:3*n].reshape(n,3);v=state[3*n:6*n].reshape(n,3)
        q=state[6*n:6*n+npair];w=state[6*n+npair:]
        a,E,P,J,qw,h,oe,mat=mechanics(x,v,m,kind,q,w)
        return np.concatenate([v.ravel(),a.ravel(),w,qw]) if kind.startswith('memory') else np.concatenate([v.ravel(),a.ravel()])
    return func


def controls():
    rng=np.random.default_rng(912);x=rng.normal(size=(4,3));v=.3*rng.normal(size=(4,3));m=np.array([.7,1.,1.4,2.])
    Q=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    rows=[]
    for kind in ['newton','kinetic','memory_slow','memory_fast']:
        h=mechanics(x,v,m,'newton')[5];q=h+.03;w=np.linspace(-.01,.03,len(h))
        a,E,P,J,qw,h,oe,mat=mechanics(x,v,m,kind,q,w);state=np.concatenate([x.ravel(),v.ravel(),q,w]) if kind.startswith('memory') else np.concatenate([x.ravel(),v.ravel()])
        derivative=rhs(m,kind)(0,state)
        def energy(y):
            return mechanics(y[:12].reshape(4,3),y[12:24].reshape(4,3),m,kind,y[24:30],y[30:])[1]
        de=(energy(state+1e-5*derivative)-energy(state-1e-5*derivative))/2e-5
        rotated=mechanics(x@Q.T+[1,2,3],v@Q.T,m,kind,q,w)[0]
        boosted=mechanics(x,v+[1.,-2,3],m,kind,q,w)[0]
        force=np.linalg.norm(np.sum(m[:,None]*a,axis=0));err=max(np.max(abs(rotated-a@Q.T)),np.max(abs(boosted-a)))
        row=dict(kind=kind,energy_rate_normalized=abs(de)/max(1,abs(E)),total_force=force,covariance_error=float(err),minimum_inertia_eigenvalue=float(np.linalg.eigvalsh(mat).min()))
        assert max(row['energy_rate_normalized'],force,err)<1e-6 and row['minimum_inertia_eigenvalue']>0
        rows.append(row)
    for omega in [.2,2.]:
        t=np.linspace(0,40,401);sol=solve_ivp(lambda tt,z:[z[1],-omega**2*(z[0]-.3)],[0,40],[.4,0],t_eval=t,method='DOP853',rtol=1e-11,atol=1e-13)
        error=float(np.max(abs(sol.y[0]-(.3+.1*np.cos(omega*t)))));assert error<1e-8
        rows.append(dict(control='fixed_radius_memory_oscillator',omega=omega,max_absolute_error=error))
    return rows


def run():
    run=OUT/'run001';run.mkdir(parents=True,exist_ok=False);start=time.time()
    import hashlib,csv
    bindings={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),OUT/'PREFLIGHT.md']}
    (run/'bindings.json').write_text(json.dumps(bindings,indent=2),encoding='utf-8')
    (run/'pre-integration-controls.json').write_text(json.dumps(controls(),indent=2),encoding='utf-8')
    results=[];series=[]
    for case in ['nested_co','nested_counter','clustered','flyby']:
        x,v,m,T=setup(case);n=len(m)
        for kind in ['newton','kinetic','memory_slow','memory_fast']:
            h=mechanics(x,v,m,'newton')[5];y0=np.concatenate([x.ravel(),v.ravel(),h,np.zeros_like(h)]) if kind.startswith('memory') else np.concatenate([x.ravel(),v.ravel()])
            solutions=[]
            for resolution,step,rtol,atol in [('base',.1,1e-8,1e-10),('fine',.05,1e-10,1e-12)]:
                sol=solve_ivp(rhs(m,kind),[0,T],y0,method='DOP853',t_eval=np.linspace(0,T,401),max_step=step,rtol=rtol,atol=atol)
                if not sol.success:raise RuntimeError(sol.message)
                E=[];P=[];J=[];orb=[];osc=[];inward=[];closest=[]
                for k,t in enumerate(sol.t):
                    state=sol.y[:,k];xx=state[:3*n].reshape(n,3);vv=state[3*n:6*n].reshape(n,3);npair=len(h)
                    a,e,p,j,qw,hh,oe,mat=mechanics(xx,vv,m,kind,state[6*n:6*n+npair],state[6*n+npair:])
                    radii=np.linalg.norm(xx[1:]-xx[0],axis=1);E.append(e);P.append(p);J.append(j);orb.append(radii);osc.append(oe)
                    inner=-np.sum((a[1:]-a[0])*(xx[1:]-xx[0]),axis=1)/radii;inward.append(inner.mean())
                    closest.append(min(np.linalg.norm(xx[ii]-xx[jj]) for ii,jj in itertools.combinations(range(n),2)))
                    if resolution=='fine':series.append(dict(case=case,kind=kind,time=float(t),energy=e,oscillator_energy=oe,mean_radius=float(radii.mean()),mean_inward_acceleration=float(inner.mean())))
                E=np.array(E);P=np.array(P);J=np.array(J);orb=np.array(orb)
                ed=float(np.max(abs(E-E[0]))/max(abs(E[0]),1e-10));pd=float(np.max(np.linalg.norm(P-P[0],axis=1))/max(1,np.sum(m[:,None]*abs(v))));jd=float(np.max(np.linalg.norm(J-J[0],axis=1))/max(1,np.linalg.norm(J[0])))
                entry=dict(case=case,kind=kind,resolution=resolution,duration=T,nfev=sol.nfev,energy_drift=ed,momentum_drift=pd,angular_momentum_drift=jd,conservation_pass=max(ed,pd,jd)<1e-5,min_radius=float(orb.min()),max_radius=float(orb.max()),closest_pair=float(min(closest)),mean_inward_acceleration=float(np.mean(inward)),max_oscillator_energy=float(max(osc)),initial_energy=float(E[0]),final_energy=float(E[-1]))
                solutions.append(sol);results.append(entry)
            diff=np.linalg.norm(solutions[0].y[:3*n,-1]-solutions[1].y[:3*n,-1])/max(1,np.linalg.norm(solutions[1].y[:3*n,-1]))
            results[-1]['trajectory_resolution_relative']=float(diff);results[-1]['trajectory_converged']=bool(diff<1e-3)
            print(case,kind,'drift',results[-1]['energy_drift'],'trajectory',diff,flush=True)
    with (run/'timeseries.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=series[0]);w.writeheader();w.writerows(series)
    (run/'results.json').write_text(json.dumps(dict(status='THEORY_BENCHMARK_ONLY',seconds=time.time()-start,integrations=len(results),results=results),indent=2),encoding='utf-8')


if __name__=='__main__':run()
