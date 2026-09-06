"""Registered short-interval check after long clustered trajectory failures."""
import json
import numpy as np
from scipy.integrate import solve_ivp
from mond_atlas_dynamic_program import OUT,setup,mechanics,rhs

def run():
    x,v,m,_=setup('clustered');n=len(m);h=mechanics(x,v,m,'newton')[5];rows=[]
    for kind in ['newton','kinetic','memory_slow','memory_fast']:
        state=np.concatenate([x.ravel(),v.ravel(),h,np.zeros_like(h)]) if kind.startswith('memory') else np.concatenate([x.ravel(),v.ravel()])
        values=[];drifts=[]
        for step,rtol,atol in [(.1,1e-8,1e-10),(.05,1e-10,1e-12)]:
            sol=solve_ivp(rhs(m,kind),[0,2],state,method='DOP853',t_eval=np.linspace(0,2,401),max_step=step,rtol=rtol,atol=atol);assert sol.success
            ee=[];pp=[];jj=[]
            for y in sol.y.T:
                a,e,p,j,*_=mechanics(y[:3*n].reshape(n,3),y[3*n:6*n].reshape(n,3),m,kind,y[6*n:6*n+len(h)],y[6*n+len(h):]);ee.append(e);pp.append(p);jj.append(j)
            ee=np.array(ee);pp=np.array(pp);jj=np.array(jj)
            drifts.append(dict(energy=float(max(abs(ee-ee[0]))/abs(ee[0])),momentum=float(max(np.linalg.norm(pp-pp[0],axis=1))/max(1,np.sum(m[:,None]*abs(v)))),angular_momentum=float(max(np.linalg.norm(jj-jj[0],axis=1))/max(1,np.linalg.norm(jj[0])))))
            values.append(sol.y[:3*n,-1])
        error=float(np.linalg.norm(values[0]-values[1])/max(1,np.linalg.norm(values[1])))
        rows.append(dict(kind=kind,duration=2.,trajectory_resolution_relative=error,trajectory_converged=error<1e-3,invariant_drifts=drifts,conservation_pass=all(max(r.values())<1e-5 for r in drifts)))
    (OUT/'short-time-results.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8');print(json.dumps(rows,indent=2))

if __name__=='__main__':run()
