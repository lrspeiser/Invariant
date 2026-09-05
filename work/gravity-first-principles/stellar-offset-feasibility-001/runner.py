"""Conditional source feasibility with small projected offsets; no gravity fits."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.optimize import linprog

def offset_fraction(R,a,d):
    R,a,d=np.broadcast_arrays(R,a,d)
    D=np.sqrt(((R-d)**2+a*a)*((R+d)**2+a*a))
    X=R*R-a*a-d*d
    # Avoid cancellation for small apertures.
    return np.where(X<0,2*a*a*R*R/(D*(D-X)),.5*(1+X/D))

root=Path(__file__).parent/'Invariant'
base=root/'work/gravity-first-principles'
paths=[base/'projected-stellar-packet-001/result.json',base/'xcop-pressure-002/source_preflight.json']
packet,old=[json.loads(p.read_bytes()) for p in paths]
r500={p['cluster']:p['r500_kpc'] for p in old['packets']}
dest=base/'stellar-offset-feasibility-001'
dest.mkdir(exist_ok=False)
registration=dict(input_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    offset_fractions_r500=[0,.005,.01,.02], scale_fractions_r500=[.001,.002,.005,.01,.02],
    runs=['shell_only','shell_plus_offset_components'],
    scope='All projected mass bounds retained. Positive shell masses plus nonnegative offset Plummer components. Multiple offsets form a flexible source hypothesis, not measured galaxy centers. No gravity residuals or physical source admission.')
(dest/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
controls=[]
for a in [.1,1,3]:
    for d in [0,.2,2]:
        for R in [.01,.3,1,5]:
            # Independent radial integral of angularly integrated Plummer surface density.
            def integrand(s):
                A=s*s+d*d+a*a
                D=np.sqrt(((s-d)**2+a*a)*((s+d)**2+a*a))
                return 2*a*a*s*A/D**3
            numerical=quad(integrand,0,R,epsabs=1e-12,epsrel=1e-12)[0]
            exact=float(offset_fraction(R,a,d))
            error=abs(numerical-exact)/max(exact,1e-15)
            assert error<1e-9
            controls.append(dict(a=a,d=d,R=R,relative_error=error))
rows=[]
for p in packet['stellar_packets']:
    c=p['columns']; r,m,lo,hi=[np.array(c[k]) for k in ['RADIUS','MSTAR','MSTAR_LO','MSTAR_HI']]
    shells=np.unique(np.r_[r,np.sqrt(r[:-1]*r[1:]),np.geomspace(.1*r[0],10*r[-1],128)])
    u=np.minimum(r[:,None]/shells,1.)
    shell_design=u*u/(1+np.sqrt(1-u*u))
    components=[dict(kind='shell',radius_kpc=float(s)) for s in shells]
    extras=[dict(kind='offset_plummer',a_kpc=a*r500[p['cluster']],d_kpc=d*r500[p['cluster']])
            for a in registration['scale_fractions_r500'] for d in registration['offset_fractions_r500']]
    extra_design=np.array([offset_fraction(r,e['a_kpc'],e['d_kpc']) for e in extras]).T
    for mode in registration['runs']:
        fraction=shell_design if mode=='shell_only' else np.c_[shell_design,extra_design]
        labels=components if mode=='shell_only' else components+extras
        design=fraction*m[-1]/m[:,None]
        A=np.r_[np.c_[design,-np.ones(len(r))],np.c_[-design,-np.ones(len(r))]]
        b=np.r_[hi/m,-lo/m]
        answer=linprog(np.r_[np.zeros(len(labels)),1.],A_ub=A,b_ub=b,bounds=(0,None),method='highs',
                       options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        row=dict(cluster=p['cluster'],mode=mode,status=int(answer.status))
        if answer.success:
            masses=answer.x[:-1]*m[-1]
            pred=fraction@masses
            violation=float(max(0,np.max((lo-pred)/m),np.max((pred-hi)/m)))
            assert abs(violation-answer.x[-1])<2e-8 and np.all(masses>=0)
            row.update(relaxation=float(answer.x[-1]),verified_violation=violation,
                within_bounds=violation<=1e-8,projected_mass=pred.tolist(),
                positive_components=[dict(**e,mass_msun=float(v)) for e,v in zip(labels,masses) if v>0])
        rows.append(row)
        print(p['cluster'],mode,row.get('relaxation'),flush=True)
(dest/'result.json').write_text(json.dumps(dict(registration=registration,controls=controls,rows=rows,
    gravity_scores=0,physical_exclusions=0),indent=2),encoding='utf-8')
