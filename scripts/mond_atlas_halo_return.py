"""Spherical published halo fields and explicitly phenomenological return shapes."""
import numpy as np
from scipy.optimize import least_squares


def mass_shape(x,profile):
    x=np.asarray(x,float)
    if np.any(x<0) or not np.isfinite(x).all():raise ValueError('Finite nonnegative radius required')
    if profile=='NFW':
        exact=np.log1p(x)-x/(1+x)
        small=sum((-1)**k*(k-1)/k*x**k for k in range(2,12))
        return np.where(x<1e-3,small,exact)
    if profile=='Burkert':
        exact=.5*(np.log1p(x)+.5*np.log1p(x*x)-np.arctan(x))
        small=sum(x**(4*k+3)/(4*k+3)-x**(4*k+4)/(4*k+4) for k in range(4))
        return np.where(x<.01,small,exact)
    raise ValueError('Unknown profile')


def density_shape(x,profile):
    x=np.asarray(x,float)
    if np.any(x<=0):raise ValueError('Positive radius required')
    if profile=='NFW':return 1/(x*(1+x)**2)
    if profile=='Burkert':return 1/((1+x)*(1+x*x))
    raise ValueError('Unknown profile')


def field(points,rho_s,rs,profile,G=1.):
    p=np.asarray(points,float);r=np.linalg.norm(p,axis=-1)
    if p.shape[-1]!=3 or np.any(r<=0) or min(rho_s,rs,G)<=0:raise ValueError('Positive parameters and nonzero positions required')
    mass=4*np.pi*rho_s*rs**3*mass_shape(r/rs,profile)
    return -G*mass[...,None]*p/r[...,None]**3


def nfw_potential(points,rho_s,rs,G=1.):
    r=np.linalg.norm(points,axis=-1);x=r/rs
    return -4*np.pi*G*rho_s*rs**2*np.log1p(x)/x


def return_shape(x,kind,a,L=1.):
    x=np.asarray(x,float)
    if np.any(x<=0) or min(a,L)<=0:raise ValueError('Positive parameters required')
    if kind=='inverse_radius':return a/x
    if kind=='bounded_p2':return a/(1+x/L)**2
    if kind=='bounded_p3':return a*(x/L)/(1+x/L)**3
    raise ValueError('Unknown return family')


def fit_return(profile,kind,x):
    x=np.asarray(x);target=mass_shape(x,profile)/x**2
    starts=[(.1,.1),(.5,1.),(1.,10.)];results=[]
    if kind=='inverse_radius':
        a=float(np.exp(np.mean(np.log(target*x))))
        return dict(kind=kind,a=a,L=1.,loss=float(np.mean(np.log(return_shape(x,kind,a)/target)**2)),success=True,starts=[])
    for a,L in starts:
        result=least_squares(lambda v:np.log(return_shape(x,kind,*np.exp(v))/target),np.log([a,L]),
            bounds=(np.log([1e-6,1e-4]),np.log([1e6,1e4])),ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=2000)
        aa,ll=np.exp(result.x)
        results.append(dict(start=[a,L],a=float(aa),L=float(ll),loss=float(np.mean(result.fun**2)),
            success=bool(result.success),status=int(result.status),evaluations=int(result.nfev),at_bound=bool(np.any(result.active_mask))))
    best=min(results,key=lambda r:r['loss'])
    if not best['success']:raise RuntimeError('Best training fit failed')
    return dict(kind=kind,a=best['a'],L=best['L'],loss=best['loss'],success=best['success'],starts=results)


def mn_potential(point,M=6e10,a=3.,b=.3,G=4.30091727003628e-6):
    x,y,z=np.asarray(point);B=np.sqrt(z*z+b*b)
    return -G*M/np.sqrt(x*x+y*y+(a+B)**2)


def mn_field(point,M=6e10,a=3.,b=.3,G=4.30091727003628e-6):
    x,y,z=np.asarray(point);B=np.sqrt(z*z+b*b);den=(x*x+y*y+(a+B)**2)**1.5
    return -G*M/den*np.array([x,y,(a+B)*z/B])


def numerical_gradient(function,point,h=1e-4):
    point=np.asarray(point,float);result=[]
    for axis in np.eye(3):
        result.append((-function(point+2*h*axis)+8*function(point+h*axis)-8*function(point-h*axis)+function(point-2*h*axis))/(12*h))
    return np.array(result)


def curl(function,point,h=1e-4):
    jac=np.column_stack([(function(np.asarray(point)+h*axis)-function(np.asarray(point)-h*axis))/(2*h) for axis in np.eye(3)])
    return np.array([jac[2,1]-jac[1,2],jac[0,2]-jac[2,0],jac[1,0]-jac[0,1]])
