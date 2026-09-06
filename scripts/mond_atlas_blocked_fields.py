"""Disk-backed Cartesian Poisson/QUMOND; identical stencil, bounded work arrays.

Every DST transforms the entire length of its selected axis. Chunking applies
only across independent pencils, so it is not a local or tiled gravity model.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from mond_atlas_fields import dst1
from mond_atlas_rectangular_fields import steps,qumond_source,simple_monopole_potential,laplacian


def array_file(path,shape):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    result=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=tuple(shape))
    result[:]=0
    return result


def dst_inplace(array,axis,max_elements=1500000):
    if axis not in (0,1,2) or array.ndim!=3 or max_elements<1:raise ValueError('invalid transform block')
    # Prefer contiguous x slabs whenever x is not the transform axis.
    split_axis=1 if axis==0 else 0
    pencil_elements=int(np.prod([n for a,n in enumerate(array.shape) if a!=split_axis]))
    width=max(1,max_elements//pencil_elements)
    maximum=0;count=0
    for lo in range(0,array.shape[split_axis],width):
        sel=[slice(None)]*3;sel[split_axis]=slice(lo,min(lo+width,array.shape[split_axis]));sel=tuple(sel)
        block=dst1(array[sel],axis)
        array[sel]=block;maximum=max(maximum,int(block.size));count+=1
    array.flush()
    return dict(axis=axis,split_axis=split_axis,blocks=count,maximum_block_elements=maximum)


def poisson_stream(rhs_block,potential,work,spacing,slab_rows=3,max_elements=1500000,progress=None):
    """Boundary is already present in potential. rhs_block(lo,hi) returns full yz.

    No current or prior interior potential is used as a boundary approximation.
    Work has the exact global interior shape and is overwritten by the solve.
    """
    h=steps(spacing);shape=potential.shape
    if potential.ndim!=3 or min(shape)<5 or work.shape!=tuple(n-2 for n in shape):raise ValueError('grid mismatch')
    if slab_rows<1:raise ValueError('invalid slab width')
    for lo in range(1,shape[0]-1,slab_rows):
        hi=min(lo+slab_rows,shape[0]-1);value=np.asarray(rhs_block(lo,hi),float)
        if value.shape!=(hi-lo,shape[1],shape[2]) or not np.isfinite(value).all():raise ValueError('invalid source slab')
        work[lo-1:hi-1]=value[:,1:-1,1:-1]
    for axis in range(3):
        for end in (0,-1):
            dst=[slice(None)]*3;src=[slice(1,-1)]*3;dst[axis]=src[axis]=end
            face=np.asarray(potential[tuple(src)])
            if not np.isfinite(face).all():raise ValueError('nonfinite boundary')
            work[tuple(dst)]-=face/h[axis]**2
    work.flush();transforms=[]
    for axis in range(3):
        if progress:progress('forward DST axis '+str(axis))
        transforms.append(dst_inplace(work,axis,max_elements))
    eigen=[-4*np.sin(np.pi*np.arange(1,n+1)/(2*(n+1)))**2/h[a]**2 for a,n in enumerate(work.shape)]
    for lo in range(0,work.shape[0],slab_rows):
        hi=min(lo+slab_rows,work.shape[0])
        work[lo:hi]/=eigen[0][lo:hi,None,None]+eigen[1][None,:,None]+eigen[2][None,None,:]
    work.flush()
    for axis in range(3):
        if progress:progress('inverse DST axis '+str(axis))
        transforms.append(dst_inplace(work,axis,max_elements))
    for lo in range(1,shape[0]-1,slab_rows):
        hi=min(lo+slab_rows,shape[0]-1);potential[lo:hi,1:-1,1:-1]=work[lo-1:hi-1]
    potential.flush()
    maximum_error=0.;maximum_source=0.
    for lo in range(1,shape[0]-1,slab_rows):
        hi=min(lo+slab_rows,shape[0]-1);block=np.asarray(potential[lo-1:hi+1])
        exact=np.asarray(rhs_block(lo,hi))[:,1:-1,1:-1]
        maximum_error=max(maximum_error,float(np.max(np.abs(laplacian(block,h)-exact))))
        maximum_source=max(maximum_source,float(np.max(np.abs(exact))))
    return dict(relative_pde_residual=maximum_error/max(maximum_source,1e-30),absolute_pde_residual=maximum_error,
        transform_blocks=transforms)


def qumond_stream(potential,out,spacing,a0,slab_rows=3,progress=None):
    if out.shape!=potential.shape:raise ValueError('QUMOND grid mismatch')
    for lo in range(1,potential.shape[0]-1,slab_rows):
        hi=min(lo+slab_rows,potential.shape[0]-1)
        # Two halo rows cover the existing face-gradient stencil at internal seams.
        begin=max(0,lo-2);end=min(potential.shape[0],hi+2)
        source=qumond_source(np.asarray(potential[begin:end]),spacing,a0)
        out[lo:hi,1:-1,1:-1]=source[lo-begin:hi-begin,1:-1,1:-1]
        if progress and (lo//slab_rows)%32==0:progress('QUMOND physical slab '+str(lo))
    out.flush()


def moments_separable(components,axes,spacing):
    """Sum moments of rho=sum(Sigma_xy * f_z), independently of a 3D allocation."""
    hx,hy,hz=steps(spacing);x=axes[0][:,None];y=axes[1][None,:];z=axes[2]
    mass=0.;dipole=np.zeros(3);moment=np.zeros((3,3))
    for surface,vertical in components:
        mxy=float(surface.sum()*hx*hy);v0=float(vertical.sum()*hz)
        px=float(np.sum(surface*x)*hx*hy);py=float(np.sum(surface*y)*hx*hy)
        pz=float(np.sum(vertical*z)*hz)
        mass+=mxy*v0;dipole+=np.array([px*v0,py*v0,mxy*pz])
        moment[0,0]+=float(np.sum(surface*x*x)*hx*hy)*v0
        moment[1,1]+=float(np.sum(surface*y*y)*hx*hy)*v0
        moment[2,2]+=mxy*float(np.sum(vertical*z*z)*hz)
        moment[0,1]+=float(np.sum(surface*x*y)*hx*hy)*v0
        moment[0,2]+=px*pz;moment[1,2]+=py*pz
    moment[1,0]=moment[0,1];moment[2,0]=moment[0,2];moment[2,1]=moment[1,2]
    if mass<=0:raise ValueError('nonpositive mass')
    quadrupole=3*moment-np.eye(3)*np.trace(moment)
    return dict(mass_msun=mass,dipole_msun_kpc=dipole.tolist(),quadrupole_msun_kpc2=quadrupole.tolist())


def fill_boundary(potential,axes,moments,G,a0,kind):
    mass=moments['mass_msun'];dipole=np.array(moments['dipole_msun_kpc']);quad=np.array(moments['quadrupole_msun_kpc2'])
    if kind not in ('newton','mond'):raise ValueError('unknown boundary type')
    for axis in range(3):
        others=[a for a in range(3) if a!=axis];u,v=np.meshgrid(axes[others[0]],axes[others[1]],indexing='ij')
        for end in (0,-1):
            xyz=[None]*3;xyz[axis]=np.full_like(u,axes[axis][end]);xyz[others[0]]=u;xyz[others[1]]=v
            r=np.sqrt(sum(a*a for a in xyz));sel=[slice(None)]*3;sel[axis]=end
            if kind=='newton':
                dot=sum(a*d for a,d in zip(xyz,dipole));q=sum(quad[i,j]*xyz[i]*xyz[j] for i in range(3) for j in range(3))
                potential[tuple(sel)]=-G*(mass/r+dot/r**3+.5*q/r**5)
            else:potential[tuple(sel)]=simple_monopole_potential(r,G*mass,a0)
    potential.flush()
