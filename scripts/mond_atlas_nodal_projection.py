"""Project the same bilinear planar source basis used by the field loader."""
from __future__ import annotations
import numpy as np
from mond_atlas_source_projection import roughness_gradient


def nodal_projection_matrix(nodes,spacing,height,inclination_deg):
    if nodes<3 or spacing<=0 or height<0 or not np.isfinite([spacing,height,inclination_deg]).all() or not 0<=inclination_deg<89:
        raise ValueError('invalid node-basis projection')
    scale=height*np.tan(np.deg2rad(inclination_deg))
    k=np.abs(np.arange(nodes)[:,None]-np.arange(nodes)[None,:]);a=np.zeros((nodes,nodes),float)
    if scale==0:
        a[k==0]=.75;a[k==1]=.125;return a
    q=spacing/scale
    if q<.001:raise ValueError('tiny spacing/kernel ratio requires a higher-precision coefficient expansion')
    e=lambda t:-np.expm1(-t*q)
    a[k==0]=.75+(e(1.5)-3*e(.5))/q**2
    a[k==1]=.125+(e(2.5)-3*e(1.5)+4*e(.5))/(2*q**2)
    far=k>=2;a[far]=e(1.)**3/(2*q**2)*np.exp(-(k[far]-1.5)*q)
    if np.any(a<0) or np.max(a.sum(axis=0))>1+1e-10:raise ArithmeticError('projection coefficient positivity/normalization failure')
    return a


def project_nodes(surface,left,right):return left@surface@right.T


def adjoint_nodes(image,left,right):return left.T@image@right


def fit_nodes(target,weight,left,right,support,regularization=1e-4,max_iterations=4000,tolerance=1e-6):
    target=np.asarray(target,float);weight=np.asarray(weight,float);support=np.asarray(support,bool)
    if target.ndim!=2 or weight.shape!=target.shape or support.shape!=target.shape or left.shape!=(target.shape[0],)*2 or right.shape!=(target.shape[1],)*2:
        raise ValueError('projection shape mismatch')
    if not np.isfinite(target).all() or not np.isfinite(weight).all() or np.any(weight<0) or np.max(weight)>1 or regularization<0 or not np.any(weight>0):raise ValueError('invalid source fit')
    for a in (left,right):
        if not np.isfinite(a).all() or np.any(a<0) or max(a.sum(axis=0).max(),a.sum(axis=1).max())>1+1e-10:raise ValueError('noncontractive projection')
    scale=max(float(np.sqrt(np.sum(weight*target**2)/np.sum(weight))),1e-12);data=target/scale
    current=np.where(support&(weight>0),np.maximum(data,0),0);extrapolated=current.copy();t=1.;L=1+8*regularization
    history=[];converged=False
    def gradient(s):return adjoint_nodes(weight*(project_nodes(s,left,right)-data),left,right)+regularization*roughness_gradient(s)
    for iteration in range(1,max_iterations+1):
        new=np.where(support,np.maximum(extrapolated-gradient(extrapolated)/L,0),0)
        tnew=(1+np.sqrt(1+4*t*t))/2;extrapolated=new+(t-1)/tnew*(new-current);current=new;t=tnew
        if iteration%50==0 or iteration==max_iterations:
            residual=project_nodes(current,left,right)-data
            error=float(np.sqrt(np.mean((current-np.where(support,np.maximum(current-gradient(current)/L,0),0))**2)))
            value=.5*np.sum(weight*residual**2)+.5*regularization*sum(np.sum(np.diff(current,axis=a)**2) for a in (0,1))
            history.append(dict(iteration=iteration,objective=float(value),projected_gradient_relative_rms=error))
            if error<tolerance:converged=True;break
    return current*scale,dict(converged=converged,iterations=iteration,projected_gradient_relative_rms=error,
        normalizing_intensity=scale,history=history,source_representation='bilinear planar nodes')
