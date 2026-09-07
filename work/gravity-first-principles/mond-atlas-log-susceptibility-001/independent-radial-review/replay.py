import csv,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[4];P=Path(__file__).resolve().parent
source=P.parent/'run002/timeseries.csv';rows=list(csv.DictReader(source.open()))
bindings={p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__),P/'SCOPE.md',R/'scripts/mond_atlas_log_susceptibility.py']}
result=[];mu=.1/1.1;mm=.1;M=1.1;eta=.15;b=.05;L=1.
with threadpool_limits(limits=1):
    for omega in [.2,2.]:
        for r0 in [1.,2.]:
            for kind in ['circular','perturbed']:
                selected=[r for r in rows if float(r['omega'])==omega and float(r['r0'])==r0 and r['kind']==kind];assert len(selected)==401
                ts=np.array([float(r['time']) for r in selected]);s0=math.sqrt(r0*r0+b*b)
                speed=math.sqrt(M*r0*r0*(s0**-3+.5*omega*omega*eta*eta*L/(s0*s0*(s0+L))))
                if kind=='perturbed':speed*=.97
                angular=r0*speed
                def fun(t,y):
                    r,dr,q,dq=y;s=math.sqrt(r*r+b*b);log=math.log1p(L/s);h=eta*math.sqrt(mm*log)
                    dh=-eta*math.sqrt(mm)*L*r/(2*math.sqrt(log)*s*s*(s+L))
                    dd=angular*angular/r**3-M*r/s**3+omega*omega*q*dh/mu
                    return [dr,dd,dq,-omega*omega*(q-h)]
                sol=solve_ivp(fun,[0,40],[r0,0,eta*math.sqrt(mm*math.log1p(L/s0)),0],method='Radau',t_eval=ts,rtol=1e-10,atol=1e-12,max_step=.05);assert sol.success
                rr,dr,q,dq=sol.y;s=np.sqrt(rr*rr+b*b);h=eta*np.sqrt(mm*np.log1p(L/s))
                internal=.5*dq*dq+.5*omega*omega*q*q-omega*omega*q*h
                energy=.5*mu*(dr*dr+angular*angular/(rr*rr))-.1/s+internal
                errors={key:float(np.max(abs(values-np.array([float(r[key]) for r in selected])))) for key,values in [('radius',rr),('q',q),('internal_signed_energy',internal),('energy',energy)]}
                drift=float(np.max(abs(energy-energy[0])));assert max(errors.values())<1e-7 and drift<1e-7
                result.append(dict(omega=omega,r0=r0,kind=kind,absolute_errors=errors,energy_drift=drift))
for name,digest in bindings.items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest
receipt=dict(status='PASS_INDEPENDENT_LOG_RADIAL_RADAU_REPLAY',cases=len(result),samples=len(rows),files=bindings,results=result,observational_validation=False)
(P/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(cases=len(result),samples=len(rows),maximum_error=max(max(r['absolute_errors'].values()) for r in result))))
