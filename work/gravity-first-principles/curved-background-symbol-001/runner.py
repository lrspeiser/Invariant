"""Local high-wave-number symbol on positive finite Plummer backgrounds."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
root=Path(__file__).parent/'Invariant'
sys.path.insert(0,str(root/'src'))
from invariant_gravity_extensions.length_screening import LengthScreening,anomalous_flux
dest=root/'work/gravity-first-principles/curved-background-symbol-001'
dest.mkdir(exist_ok=False)
module=root/'src/invariant_gravity_extensions/length_screening.py'
registration=dict(module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
    shapes=[.5,1.,2.],radius_over_scale=[.01,.1,.3,1.,3.],length_over_scale=[.01,.1,1.,10.],
    directions=['radial','tangential'],a0=1.,G=1.,scale=1.,mass=2*np.sqrt(2),
    leading_symbol='B(n)=P_h+2*ell²*P_hh*(n.H.n)²/a0²; deltaPhi/deltapsi ~ ell²*B*k².',
    matter_assumption='Under standard local barotropic Euler dynamics, sigma² ~ [4*pi*G*rho*ell²*B-c_s²]*k². Positive B is an unbounded shortwave-growth warning for pressureless continuum, not a covariant ghost proof.',
    scope='Local principal-symbol diagnostic on finite positive sources. No equilibrium, full time evolution, observational exclusion or complete stability claim.')
(dest/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
(dest/'length_screening.py').write_bytes(module.read_bytes())
rows=[]
M=registration['mass']
for shape in registration['shapes']:
    spec=LengthScreening(shape)
    for r in registration['radius_over_scale']:
        g=M*r/(1+r*r)**1.5
        first=M*(1-2*r*r)/(1+r*r)**2.5
        second=3*M*r*(2*r*r-3)/(1+r*r)**3.5
        H=np.diag([first,g/r,g/r]); p=np.array([g,0.,0.])
        rho=3*M/(4*np.pi)*(1+r*r)**-2.5
        dH2=np.array([2*first*second+4*g/r*(first/r-g/r**2),0.,0.])
        dlap=np.array([-20*np.pi*r*rho/(1+r*r),0.,0.])
        for ell in registration['length_over_scale']:
            x=g*g; h=ell*ell*np.sum(H*H); u=x+h
            _,ph,k1,k2,fraction=spec.partials(x,h)
            for direction,index in [('radial',0),('tangential',1)]:
                B=float(ph+2*ell*ell*fraction*k2/u*H[index,index]**2)
                rows.append(dict(shape=shape,radius=r,length=ell,direction=direction,
                    density=rho,x=x,h=h,B=B,pressureless_shortwave_growth=B>0,
                    sound_speed_squared_threshold=4*np.pi*rho*ell*ell*B))
# Independently sample the full local flux at the strongest positive B case
# per shape; isolate its k² coefficient as k increases.
checks=[]
for shape in registration['shapes']:
    row=max((v for v in rows if v['shape']==shape),key=lambda v:v['B'])
    r,ell=row['radius'],row['length']; spec=LengthScreening(shape)
    g=M*r/(1+r*r)**1.5; first=M*(1-2*r*r)/(1+r*r)**2.5
    second=3*M*r*(2*r*r-3)/(1+r*r)**3.5
    H=np.diag([first,g/r,g/r]); p=np.array([g,0.,0.])
    rho=3*M/(4*np.pi)*(1+r*r)**-2.5
    dH2=np.array([2*first*second+4*g/r*(first/r-g/r**2),0.,0.])
    dlap=np.array([-20*np.pi*r*rho/(1+r*r),0.,0.])
    n=np.eye(3)[0 if row['direction']=='radial' else 1]
    J0=p+anomalous_flux(spec,p,H,dH2,dlap,ell)
    values=[]
    for q in [100.,1000.,10000.]:
        k=q/ell
        dp=1e-5*g/(1+k*k)
        perturbed=p-dp*n
        J=perturbed+anomalous_flux(spec,perturbed,H,
            dH2+2*(n@H@n)*dp*k*k*n,dlap+dp*k*k*n,ell)
        measured=float(np.dot(J-J0,n)/(-dp)/(ell*ell*k*k))
        values.append(dict(k_times_length=q,measured_B=measured,relative_error=abs(measured-row['B'])/abs(row['B'])))
    checks.append(dict(background=row,checks=values))
out=dict(registration=registration,rows=rows,checks=checks,
    positive_B_cases=sum(r['B']>0 for r in rows),cases=len(rows),physical_exclusions=0,observational_scores=0)
(dest/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('Positive B',out['positive_B_cases'],'/',out['cases'])
print([(c['background']['shape'],c['background']['B'],c['checks'][-1]['relative_error']) for c in checks])
