"""Independent weak/action/strong-flux variation checks in spherical symmetry."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import roots_legendre, spherical_in

root=Path(__file__).parent/'Invariant'
sys.path.insert(0,str(root/'src'))
from invariant_gravity_extensions.length_screening import LengthScreening
dest=root/'work/gravity-first-principles/isolated-action-variation-001'
dest.mkdir(exist_ok=False)
module=root/'src/invariant_gravity_extensions/length_screening.py'
registration=dict(module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
    shapes=[.5,1.,2.],filter_lengths=[.3,1.,3.],curvature_lengths=[.1,1.],
    nodes=[256,512,1024],outer_radii=[20.,40.],
    epsilon=[1e-3,1e-4,1e-5],relative_flux_tolerance=1e-4,
    relative_finite_variation_tolerance=1e-6,
    construction='Choose smooth Gaussian chi and delta_chi; psi=(1-L^2 Delta)chi so S psi=chi exactly on the isolated full space.',
    base_gaussians=[[-.2,1.],[-.03,2.]],direction_gaussians=[[.1,.7],[-.03,1.7]],
    scope='Signed manufactured potentials, not positive matter profiles, equilibrium, full stability, or astronomical admission. Common angular factor 4pi omitted; a0=1.')
(dest/'registration.json').write_text(json.dumps(registration,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())

def fields(r,L,components):
    p,dp,hr,ht,dhr,dht=[np.zeros_like(r) for _ in range(6)]
    for amplitude,b in components:
        e=amplitude*np.exp(-r*r/(2*b*b))
        p+=e*(-r/b**2-L*L*(5*r/b**4-r**3/b**6))
        dp+=e*(-1/b**2+r*r/b**4-L*L*(5/b**4-8*r*r/b**6+r**4/b**8))
        hr+=e*(r*r/b**4-1/b**2)
        ht-=e/b**2
        dhr+=e*(3*r/b**4-r**3/b**6)
        dht+=e*r/b**4
    return p,dp,hr,ht,dhr,dht

def matrix(r,weights,L):
    q=r/L
    small=np.minimum(q[:,None],q[None,:])
    large=np.maximum(q[:,None],q[None,:])
    si=np.empty_like(small)
    mask=small<20
    si[mask]=np.exp(-small[mask])*spherical_in(1,small[mask])
    z=small[~mask]
    si[~mask]=((z-1)+(z+1)*np.exp(-2*z))/(2*z*z)
    return np.exp(-(large-small))*si*(large+1)/large**2*(q*q*weights/L)[None,:]

rows=[]
for R in registration['outer_radii']:
    for n in registration['nodes']:
        z,weights=roots_legendre(n)
        r=(z+1)*R/2; weights=weights*R/2; measure=weights*r*r
        for L in registration['filter_lengths']:
            S=matrix(r,weights,L)
            p,dp,hr,ht,dhr,dht=fields(r,L,registration['base_gaussians'])
            v,_,vr,vt,_,_=fields(r,L,registration['direction_gaussians'])
            for shape in registration['shapes']:
                spec=LengthScreening(shape)
                for ell in registration['curvature_lengths']:
                    x=p*p; h=ell*ell*(hr*hr+2*ht*ht)
                    px,ph,k1,k2,fraction=spec.partials(x,h)
                    dx=2*p*dp; dh=2*ell*ell*(hr*dhr+2*ht*dht)
                    dph=np.divide((k1+fraction*k2)*dx+fraction*k2*dh,x+h,
                                   out=np.zeros_like(x),where=(x+h)>0)
                    divergence=dph*hr+ph*(dhr+2*dht)
                    J=(1+px)*p-ell*ell*(S@divergence)
                    weak=float(2*np.dot(measure,(1+px)*p*v+ell*ell*ph*(hr*vr+2*ht*vt)))
                    flux=float(2*np.dot(measure,J*v))
                    wrong=float(2*np.dot(measure,((1+px)*p-ell*ell*divergence)*v))
                    checks=[]
                    for eps in registration['epsilon']:
                        energies=[]
                        for sign in [1,-1]:
                            en=spec.value((p+sign*eps*v)**2,
                                ell*ell*((hr+sign*eps*vr)**2+2*(ht+sign*eps*vt)**2))
                            energies.append(float(np.dot(measure,en)))
                        finite=(energies[0]-energies[1])/(2*eps)
                        checks.append(dict(epsilon=eps,finite=finite,relative_error=abs(finite-weak)/max(abs(weak),1e-12)))
                    rows.append(dict(R=R,n=n,L=L,shape=shape,ell=ell,weak=weak,flux=flux,
                        relative_flux_error=abs(flux-weak)/max(abs(weak),1e-12),
                        omitted_filter_relative_error=abs(wrong-weak)/max(abs(weak),1e-12),finite_checks=checks))
        print(f'Completed R={R}, nodes={n}',flush=True)
fine=[r for r in rows if r['n']==1024]
worst=max(r['relative_flux_error'] for r in fine)
worst_fd=max(r['finite_checks'][-1]['relative_error'] for r in fine)
out=dict(registration=registration,rows=rows,worst_fine_flux_relative_error=worst,
    worst_fine_finite_relative_error=worst_fd,
    passed=bool(worst<1e-4 and worst_fd<1e-6),observational_scores=0,admitted_candidates=0)
(dest/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ['registration','rows']}))
