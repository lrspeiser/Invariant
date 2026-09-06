"""Frozen bounded strengths, interpolate variational coefficients, not acceleration."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from mond_atlas_dynamic_partition import OUT,ROOT,setup,mechanics


def scaled(x,v,m,epsilon):
    a0,E0,p0,J0,M0,T0,Tk0=mechanics(x,v,m,'newton');a1,E1,p1,J1,M1,T1,Tk1=mechanics(x,v,m,'corrected');scale=epsilon/.5
    mat=M0+scale*(M1-M0);force=M0@a0+scale*(M1@a1-M0@a0);a=np.linalg.solve(mat,force);p=mat@v
    return a,E0+scale*(E1-E0),p.sum(axis=0),np.cross(x,p).sum(axis=0)


def run():
    dest=OUT/'strength-sweep';dest.mkdir(parents=True,exist_ok=False)
    bound=[Path(__file__),ROOT/'scripts/mond_atlas_dynamic_partition.py',OUT/'STRENGTH_SWEEP_PREFLIGHT.md',OUT/'run001/results.json']
    (dest/'bindings.json').write_text(json.dumps({str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in bound},indent=2),encoding='utf-8')
    x,v,m,_=setup('nested_co');checks=[]
    for epsilon in [0.,.005,.05,.5]:
        a,E,P,J=scaled(x,v,m,epsilon);aa,ee,pp,jj=scaled(np.repeat(x,2,axis=0),np.repeat(v,2,axis=0),np.repeat(m/2,2),epsilon)
        err=float(np.max(abs(aa-np.repeat(a,2,axis=0))));assert err<1e-10
        if epsilon==0:assert np.max(abs(a-mechanics(x,v,m,'newton')[0]))<1e-12
        checks.append(dict(epsilon=epsilon,partition_acceleration_error=err,newton_limit_checked=epsilon==0))
    (dest/'pre-integration-controls.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    old=json.loads((OUT/'run001/results.json').read_text(encoding='utf-8'));rows=[]
    for row in old['results']:
        if row['case'] in ['nested_co','nested_counter','flyby'] and row['kind']=='corrected':
            rows.append(dict(**row,epsilon=.5,reused=True,reuse_source='partition-repair/run001/results.json'))
    for case in ['nested_co','nested_counter','flyby']:
        x,v,m,T=setup(case);n=len(m);state=np.r_[x.ravel(),v.ravel()];initial_r=np.linalg.norm(x[1:]-x[0],axis=1)
        newton_a=mechanics(x,v,m,'newton')[0]
        newton_inward=-np.sum((newton_a[1:]-newton_a[0])*(x[1:]-x[0]),axis=1)/initial_r
        for epsilon in [.005,.05]:
            initial_a=scaled(x,v,m,epsilon)[0];initial_inward=-np.sum((initial_a[1:]-initial_a[0])*(x[1:]-x[0]),axis=1)/initial_r
            sols=[]
            def rhs(t,y):
                xx=y[:3*n].reshape(n,3);vv=y[3*n:].reshape(n,3);return np.r_[vv.ravel(),scaled(xx,vv,m,epsilon)[0].ravel()]
            for res,step,rtol,atol in [('base',.1,1e-8,1e-10),('fine',.05,1e-10,1e-12)]:
                sol=solve_ivp(rhs,[0,T],state,method='DOP853',t_eval=np.linspace(0,T,401),max_step=step,rtol=rtol,atol=atol);assert sol.success
                ee=[];pp=[];jj=[];rr=[];gg=[]
                for y in sol.y.T:
                    xx=y[:3*n].reshape(n,3);vv=y[3*n:].reshape(n,3);a,e,p,j=scaled(xx,vv,m,epsilon);r=np.linalg.norm(xx[1:]-xx[0],axis=1)
                    ee.append(e);pp.append(p);jj.append(j);rr.extend(r);gg.append(np.mean(-np.sum((a[1:]-a[0])*(xx[1:]-xx[0]),axis=1)/r))
                ee=np.array(ee);pp=np.array(pp);jj=np.array(jj);ed=float(max(abs(ee-ee[0]))/max(abs(ee[0]),1e-10));pd=float(max(np.linalg.norm(pp-pp[0],axis=1))/max(1,np.sum(m[:,None]*abs(v))));jd=float(max(np.linalg.norm(jj-jj[0],axis=1))/max(1,np.linalg.norm(jj[0])))
                rows.append(dict(case=case,kind='corrected',epsilon=epsilon,resolution=res,duration=T,reused=False,initial_energy=float(ee[0]),initial_energy_positive=bool(ee[0]>0),initial_mean_inward_ratio_to_newton=float(np.mean(initial_inward)/np.mean(newton_inward)),min_radius=float(min(rr)),max_radius=float(max(rr)),radius_exceeds_10x_initial=bool(max(rr)>10*max(initial_r)),mean_inward_acceleration=float(np.mean(gg)),energy_drift=ed,momentum_drift=pd,angular_drift=jd,conservation_pass=max(ed,pd,jd)<1e-5));sols.append(sol)
            error=float(np.linalg.norm(sols[0].y[:3*n,-1]-sols[1].y[:3*n,-1])/max(1,np.linalg.norm(sols[1].y[:3*n,-1])));rows[-1].update(trajectory_resolution_relative=error,trajectory_converged=error<1e-3)
            print(case,epsilon,rows[-1]['max_radius'],error,flush=True)
    (dest/'results.json').write_text(json.dumps(dict(status='THEORY_BENCHMARK_ONLY',new_integrations=12,reused_integrations=6,results=rows),indent=2),encoding='utf-8')


if __name__=='__main__':run()
