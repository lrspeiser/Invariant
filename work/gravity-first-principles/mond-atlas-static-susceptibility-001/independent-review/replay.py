import csv,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[4];P=Path(__file__).resolve().parent
source=P.parent/'run001/timeseries.csv';rows=list(csv.DictReader(source.open()))
bindings={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__),P/'SCOPE.md',R/'scripts/mond_atlas_static_susceptibility.py']}
result=[];mu=.1/1.1;mm=.1;M=1.1;eta=.15;b=.05
with threadpool_limits(limits=1):
    for omega in [.2,2.]:
        for r0 in [1.,2.]:
            for kind in ['circular','perturbed']:
                selected=[r for r in rows if float(r['omega'])==omega and float(r['r0'])==r0 and r['kind']==kind];assert len(selected)==401
                ts=np.array([float(r['time']) for r in selected]);speed=math.sqrt(M*r0*r0*((r0*r0+b*b)**-1.5+omega*omega*eta*eta*math.exp(-r0*r0)))
                if kind=='perturbed':speed*=.97
                angular=r0*speed
                def fun(t,y):
                    r,dr,q,dq=y;h=eta*math.sqrt(mm)*math.exp(-r*r/2)
                    dd=angular*angular/r**3-M*r/(r*r+b*b)**1.5-omega*omega*q*h*r/mu
                    return [dr,dd,dq,-omega*omega*(q-h)]
                sol=solve_ivp(fun,[0,40],[r0,0,eta*math.sqrt(mm)*math.exp(-r0*r0/2),0],method='Radau',t_eval=ts,rtol=1e-10,atol=1e-12,max_step=.05);assert sol.success
                rr,dr,q,dq=sol.y;h=eta*math.sqrt(mm)*np.exp(-rr*rr/2)
                internal=.5*dq*dq+.5*omega*omega*q*q-omega*omega*q*h
                energy=.5*mu*(dr*dr+angular*angular/(rr*rr))-.1/np.sqrt(rr*rr+b*b)+internal
                errors={key:float(np.max(abs(values-np.array([float(r[key]) for r in selected])))) for key,values in [('radius',rr),('q',q),('internal_signed_energy',internal),('energy',energy)]}
                drift=float(np.max(abs(energy-energy[0])));assert max(errors.values())<1e-7 and drift<1e-7
                result.append(dict(omega=omega,r0=r0,kind=kind,absolute_errors=errors,energy_drift=drift))
for name,digest in bindings.items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest
receipt=dict(status='PASS_INDEPENDENT_RADIAL_RADAU_REPLAY',cases=len(result),samples=len(rows),files=bindings,results=result,observational_validation=False)
(P/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(cases=len(result),samples=len(rows),maximum_error=max(max(r['absolute_errors'].values()) for r in result))))
