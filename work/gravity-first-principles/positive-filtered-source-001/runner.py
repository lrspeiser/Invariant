"""Positive finite sources with an exact filtered Plummer potential."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import roots_legendre, spherical_in
root=Path(__file__).parent/'Invariant'
sys.path.insert(0,str(root/'src'))
from invariant_gravity_extensions.length_screening import LengthScreening
dest=root/'work/gravity-first-principles/positive-filtered-source-001'
dest.mkdir(exist_ok=False)
registration=dict(a=1.,G=1.,a0=1.,masses=[.01,1.,100.],L=[.1,.3,.5],
    ell=[.1,1.,10.],shapes=[.5,1.,2.],radii=np.geomspace(.01,100,41).tolist(),
    quadrature_orders=[32,64],relative_force_tolerance=1e-5,
    chi='-M/sqrt(r^2+a^2)',psi='chi-L^2 Laplacian chi',
    rho='3 M a^2/(4 pi (r^2+a^2)^(5/2)) * [1-5 L^2 (4r^2-3a^2)/(r^2+a^2)^2]',
    positivity='rho>=rho_Plummer*(1-20 L^2/(7 a^2)); positive for L/a<sqrt(7/20). Total mass M.',
    boundary='isolated regular source, no point mass or growing homogeneous field',
    scope='81 parameter/source combinations, diagnostic only. Source construction depends on L; this is not an unchanged-source parameter comparison or an observational fit.')
(dest/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
module=root/'src/invariant_gravity_extensions/length_screening.py'
(dest/'module_sha256.txt').write_text(hashlib.sha256(module.read_bytes()).hexdigest(),encoding='utf-8')

def fields(r,M,L):
    D=1+r*r
    p=M*r/D**1.5+15*L*L*M*r/D**3.5
    dp=M*(1-2*r*r)/D**2.5+15*L*L*M*(1-6*r*r)/D**4.5
    hr=M*(1-2*r*r)/D**2.5
    ht=M/D**1.5
    dhr=M*r*(6*r*r-9)/D**3.5
    dht=-3*M*r/D**2.5
    return p,dp,hr,ht,dhr,dht

def terms(r,M,L,ell,spec):
    p,dp,hr,ht,dhr,dht=fields(r,M,L)
    x=p*p; h=ell*ell*(hr*hr+2*ht*ht)
    px,ph,k1,k2,f=spec.partials(x,h)
    dx=2*p*dp; dh=2*ell*ell*(hr*dhr+2*ht*dht)
    dph=((k1+f*k2)*dx+f*k2*dh)/(x+h)
    return p,(1+px)*p,dph*hr+ph*(dhr+2*dht)

def quadrature(r,L,n,tail=40):
    end=r+tail*L
    cuts=[0.,end,r]
    cuts.extend(x for x in [.01,.03,.1,.3,1.,3.,10.,30.,100.] if x<end)
    cuts.extend(r+sign*factor*L for sign in [-1,1] for factor in [1,4,16] if 0<r+sign*factor*L<end)
    cuts=np.unique(cuts)
    z,w=roots_legendre(n)
    s=np.concatenate([(lo+hi)/2+(hi-lo)*z/2 for lo,hi in zip(cuts[:-1],cuts[1:])])
    weights=np.concatenate([(hi-lo)*w/2 for lo,hi in zip(cuts[:-1],cuts[1:])])
    q=r/L; t=s/L
    small=np.minimum(q,t); large=np.maximum(q,t)
    si=np.empty_like(small); mask=small<20
    si[mask]=np.exp(-small[mask])*spherical_in(1,small[mask])
    z=small[~mask]
    si[~mask]=((z-1)+(z+1)*np.exp(-2*z))/(2*z*z)
    kernel=t*t*np.exp(-(large-small))*si*(large+1)/large**2/L
    return s,weights*kernel

rows=[]
radii=np.array(registration['radii'])
for L in registration['L']:
    grids={n:[quadrature(r,L,n) for r in radii] for n in registration['quadrature_orders']}
    for M in registration['masses']:
        for shape in registration['shapes']:
            spec=LengthScreening(shape)
            for ell in registration['ell']:
                p,first,_=terms(radii,M,L,ell,spec)
                predictions={}
                for n in registration['quadrature_orders']:
                    reaction=np.array([np.dot(w,terms(s,M,L,ell,spec)[2]) for s,w in grids[n]])
                    predictions[n]=first-ell*ell*reaction
                err=float(np.max(abs(predictions[64]-predictions[32])/np.maximum(abs(predictions[64]),p)))
                rows.append(dict(L=L,M=M,shape=shape,ell=ell,newtonian=p.tolist(),
                    force32=predictions[32].tolist(),force64=predictions[64].tolist(),
                    refinement_error=err,minimum_force_over_newtonian=float(np.min(predictions[64]/p)),
                    outward_sample_count=int(np.sum(predictions[64]<0))))
    print(f'Completed L={L}',flush=True)
result=dict(registration=registration,rows=rows,
    worst_refinement_error=max(r['refinement_error'] for r in rows),
    outward_cards=sum(r['outward_sample_count']>0 for r in rows),
    passed_quadrature=bool(all(r['refinement_error']<1e-5 for r in rows)),
    observational_scores=0,admitted_candidates=0,
    limitations='Finite radius samples cannot prove inward force everywhere. Tail beyond r+40L is truncated, not rigorously bounded here. No equilibrium or dynamical test.')
(dest/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['rows','registration']}))
