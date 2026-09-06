"""Anisotropic Cartesian finite-domain QUMOND, with explicit boundary models.

This extends the independently tested DST primitive without changing prior
milestone code. Units are caller supplied. No observational inference occurs.
"""
from __future__ import annotations
import numpy as np
from mond_atlas_fields import dst1


def steps(spacing):
    h=np.asarray(spacing,float)
    if h.shape!=(3,) or not np.isfinite(h).all() or np.any(h<=0):
        raise ValueError("three positive finite grid spacings required")
    return h


def laplacian(phi,spacing):
    h=steps(spacing);mid=phi[1:-1,1:-1,1:-1]
    out=np.zeros_like(mid)
    for a in range(3):
        lo=[slice(1,-1)]*3;hi=lo.copy();lo[a]=slice(None,-2);hi[a]=slice(2,None)
        out+=(phi[tuple(lo)]+phi[tuple(hi)]-2*mid)/h[a]**2
    return out


def poisson(rhs,boundary,spacing):
    h=steps(spacing);rhs=np.asarray(rhs,float);boundary=np.asarray(boundary,float)
    if rhs.ndim!=3 or min(rhs.shape)<5 or rhs.shape!=boundary.shape:
        raise ValueError("matching 3D grids of at least five nodes required")
    if not np.isfinite(rhs).all() or not np.isfinite(boundary).all():
        raise ValueError("nonfinite Poisson inputs")
    b=rhs[1:-1,1:-1,1:-1].copy();eigen=[]
    for a in range(3):
        for end in (0,-1):
            dst=[slice(None)]*3;src=[slice(1,-1)]*3;dst[a]=src[a]=end
            b[tuple(dst)]-=boundary[tuple(src)]/h[a]**2
        n=b.shape[a]
        eigen.append(-4*np.sin(np.pi*np.arange(1,n+1)/(2*(n+1)))**2/h[a]**2)
    for a in range(3):b=dst1(b,a)
    b/=eigen[0][:,None,None]+eigen[1][None,:,None]+eigen[2][None,None,:]
    for a in range(3):b=dst1(b,a)
    phi=boundary.copy();phi[1:-1,1:-1,1:-1]=b
    return phi


def qumond_source(phi,spacing,a0):
    h=steps(spacing)
    if not np.isfinite(a0) or a0<=0:raise ValueError("invalid a0")
    gradients=np.gradient(phi,*h,edge_order=2)
    out=np.zeros_like(phi)
    for a in range(3):
        lo=[slice(None)]*3;hi=lo.copy();lo[a]=slice(None,-1);hi[a]=slice(1,None)
        lo,hi=tuple(lo),tuple(hi)
        normal=(phi[hi]-phi[lo])/h[a];norm2=normal**2
        for t in range(3):
            if t!=a:norm2+=(.5*(gradients[t][lo]+gradients[t][hi]))**2
        magnitude=np.sqrt(norm2);factor=np.zeros_like(magnitude);positive=magnitude>0
        factor[positive]=.5+np.sqrt(.25+a0/magnitude[positive])
        flux=factor*normal
        lower=[slice(1,-1)]*3;upper=lower.copy();lower[a]=slice(None,-1);upper[a]=slice(1,None)
        out[1:-1,1:-1,1:-1]+=(flux[tuple(upper)]-flux[tuple(lower)])/h[a]
    return out


def solve(density,spacing,boundary_n,boundary_m,G,a0):
    if not np.isfinite(density).all() or np.any(density<0) or not np.isfinite(G) or G<=0:
        raise ValueError("nonnegative finite density and positive G required")
    rhs=4*np.pi*G*density
    pn=poisson(rhs,boundary_n,spacing)
    rn=float(np.max(np.abs(laplacian(pn,spacing)-rhs[1:-1,1:-1,1:-1]))/max(np.max(np.abs(rhs)),1e-30))
    del rhs
    source=qumond_source(pn,spacing,a0)
    pm=poisson(source,boundary_m,spacing)
    rm=float(np.max(np.abs(laplacian(pm,spacing)-source[1:-1,1:-1,1:-1]))/max(np.max(np.abs(source)),1e-30))
    return pn,pm,dict(newton_relative_pde_residual=rn,mond_relative_pde_residual=rm)


def simple_monopole_potential(radius,GM,a0):
    """Exact spherical point-source simple-nu potential, arbitrary additive zero.

    dPhi/dr = GM/(2r^2) + sqrt(GM^2/(4r^4) + GM*a0/r^2).
    Only called outside the source; no claim of nonspherical MOND boundary truth.
    """
    r=np.asarray(radius,float)
    if np.any(r<=0) or GM<=0 or a0<=0:raise ValueError("positive monopole arguments")
    a=GM/2;b=np.sqrt(GM*a0);root=np.sqrt(a*a+(b*r)**2)
    return -a/r-root/r+b*np.arcsinh(b*r/a)


def multipole_boundary(density,axes,G,a0):
    """Newton l<=2 and MOND spherical monopole at six actual box faces."""
    h=[v[1]-v[0] for v in axes];dv=float(np.prod(h))
    mass=float(np.sum(density)*dv)
    if mass<=0:raise ValueError("positive source mass required")
    coords=[axes[0][:,None,None],axes[1][None,:,None],axes[2][None,None,:]]
    dipole=np.array([np.sum(density*x)*dv for x in coords])
    moment=np.array([[np.sum(density*x*y)*dv for y in coords] for x in coords])
    quad=3*moment-np.eye(3)*np.trace(moment)
    bn=np.zeros_like(density);bm=np.zeros_like(density)
    for a in range(3):
        others=[i for i in range(3) if i!=a]
        u,v=np.meshgrid(axes[others[0]],axes[others[1]],indexing='ij')
        for end in (0,-1):
            xyz=[None]*3;xyz[a]=np.full_like(u,axes[a][end]);xyz[others[0]]=u;xyz[others[1]]=v
            r=np.sqrt(sum(x*x for x in xyz));dot=sum(x*d for x,d in zip(xyz,dipole))
            q=sum(quad[i,j]*xyz[i]*xyz[j] for i in range(3) for j in range(3))
            sel=[slice(None)]*3;sel[a]=end
            bn[tuple(sel)]=-G*(mass/r+dot/r**3+.5*q/r**5)
            bm[tuple(sel)]=simple_monopole_potential(r,G*mass,a0)
    return bn,bm,dict(mass_msun=mass,dipole_msun_kpc=dipole.tolist(),quadrupole_msun_kpc2=quad.tolist())
