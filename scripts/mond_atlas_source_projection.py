"""Finite-cell projection and nonnegative source inversion for a flat disk.

All arrays have galaxy-major X as axis 0 and deprojected minor Y as axis 1.
The source is beam-blurred at the input map's effective resolution. Height is
an assumed physical vertical exponential scale; no spectral velocity is depth.
"""
from __future__ import annotations
import numpy as np


def projection_matrix(nodes,spacing,height,inclination_deg):
    if nodes<3 or spacing<=0 or height<0 or not np.isfinite([spacing,height,inclination_deg]).all() or not 0<=inclination_deg<89:
        raise ValueError('invalid flat-disk projection grid')
    scale=height*np.tan(np.deg2rad(inclination_deg))
    if scale==0:return np.eye(nodes)
    q=spacing/scale;one_minus=-np.expm1(-q)
    k=np.abs(np.arange(nodes)[:,None]-np.arange(nodes)[None,:])
    a=np.zeros((nodes,nodes),float)
    np.fill_diagonal(a,1-one_minus/q)
    off=k>0
    a[off]=one_minus**2/(2*q)*np.exp(-(k[off]-1)*q)
    # No renormalization: flux projecting outside a finite field is not returned.
    return a


def project(surface,matrix):return np.asarray(surface)@matrix.T


def adjoint(image,matrix):return np.asarray(image)@matrix


def roughness_gradient(surface):
    result=np.zeros_like(surface)
    for axis in (0,1):
        lo=[slice(None)]*2;hi=lo.copy();lo[axis]=slice(None,-1);hi[axis]=slice(1,None)
        lo,hi=tuple(lo),tuple(hi);difference=surface[hi]-surface[lo]
        result[hi]+=difference;result[lo]-=difference
    return result


def objective(surface,target,weight,matrix,regularization):
    residual=project(surface,matrix)-target
    value=.5*np.sum(weight*residual**2)
    for axis in (0,1):value+=.5*regularization*np.sum(np.diff(surface,axis=axis)**2)
    return float(value)


def fit_nonnegative(target,weight,matrix,support,regularization=1e-4,max_iterations=4000,tolerance=1e-6):
    target=np.asarray(target,float);weight=np.asarray(weight,float);support=np.asarray(support,bool)
    if target.ndim!=2 or target.shape!=weight.shape or support.shape!=target.shape or matrix.shape!=(target.shape[1],)*2:
        raise ValueError('source/image grid mismatch')
    if not np.isfinite(target).all() or not np.isfinite(weight).all() or np.any(weight<0) or np.max(weight)>1 or regularization<0:
        raise ValueError('invalid least-squares inputs')
    if not np.isfinite(matrix).all() or np.any(matrix<0) or np.max(np.sum(matrix,axis=0))>1+1e-12 or np.max(np.sum(matrix,axis=1))>1+1e-12:
        raise ValueError('projection must be a positive contraction')
    if not np.any(weight>0) or not np.any(support):raise ValueError('no fitted data or support')
    scale=max(float(np.sqrt(np.sum(weight*target**2)/np.sum(weight))),1e-12)
    observed=target/scale
    current=np.where(support&(weight>0),np.maximum(observed,0),0);extrapolated=current.copy();t=1.
    lipschitz=1+8*regularization
    history=[];gradient_error=float('inf');converged=False
    def gradient(s):return adjoint(weight*(project(s,matrix)-observed),matrix)+regularization*roughness_gradient(s)
    for iteration in range(1,max_iterations+1):
        grad=gradient(extrapolated)
        new=np.where(support,np.maximum(extrapolated-grad/lipschitz,0),0)
        tnew=(1+np.sqrt(1+4*t*t))/2
        extrapolated=new+(t-1)/tnew*(new-current);current=new;t=tnew
        if iteration%50==0 or iteration==max_iterations:
            projected=np.where(support,np.maximum(current-gradient(current)/lipschitz,0),0)
            gradient_error=float(np.sqrt(np.mean((current-projected)**2)))
            history.append(dict(iteration=iteration,objective=objective(current,observed,weight,matrix,regularization),projected_gradient_relative_rms=gradient_error))
            if gradient_error<tolerance:converged=True;break
    return current*scale,dict(iterations=iteration,converged=converged,
        projected_gradient_relative_rms=gradient_error,normalizing_intensity=scale,history=history,
        objective_interpretation='coverage-weighted source diagnostic, not a noise likelihood')


def weighted_relative_rms(predicted,target,weight):
    denominator=float(np.sum(weight*target**2))
    if denominator<=0:raise ValueError('no positive source diagnostic denominator')
    return float(np.sqrt(np.sum(weight*(predicted-target)**2)/denominator))
