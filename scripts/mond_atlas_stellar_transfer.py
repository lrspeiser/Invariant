"""Relative image registration with a fixed source mask and disjoint calibration blocks."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize


def bilinear_reference(image, xy):
    xy=np.asarray(xy,float); x,y=xy.T; ix=np.floor(x).astype(int); iy=np.floor(y).astype(int)
    if np.any(ix<0)|np.any(iy<0)|np.any(ix+1>=image.shape[1])|np.any(iy+1>=image.shape[0]):
        raise ValueError('reference sampler requires interior points')
    fx,fy=x-ix,y-iy
    return ((1-fx)*(1-fy)*image[iy,ix]+fx*(1-fy)*image[iy,ix+1]
            +(1-fx)*fy*image[iy+1,ix]+fx*fy*image[iy+1,ix+1])


def sample(image, xy, shift):
    points=np.asarray(xy)+np.asarray(shift)
    return map_coordinates(image,points.T[::-1],order=1,mode='constant',cval=np.nan,prefilter=False)


def flux_fit(observed, reconstruction):
    x=np.asarray(observed); y=np.asarray(reconstruction)
    if not np.isfinite(x).all() or not np.isfinite(y).all(): raise ValueError('nonfinite calibration')
    xc=x-x.mean(); yc=y-y.mean(); denominator=float(xc@xc)
    if denominator<=1e-24: raise ValueError('calibration lacks contrast')
    scale=float(xc@yc/denominator); offset=float(y.mean()-scale*x.mean())
    return scale,offset


def score(observed,reconstruction,scale,offset):
    prediction=scale*observed+offset
    norm=float(np.sqrt(np.mean(reconstruction**2)))
    corr=float(np.corrcoef(observed,reconstruction)[0,1]) if np.std(observed)>0 and np.std(reconstruction)>0 else None
    return dict(samples=len(observed),relative_rms=float(np.sqrt(np.mean((prediction-reconstruction)**2))/max(norm,1e-24)),
                correlation=corr)


def fit_shift(image, calibration_xy, reconstruction, radius=8, step=1):
    normalization=max(float(np.mean(reconstruction**2)),1e-24)
    def objective(shift):
        observed=sample(image,calibration_xy,shift)
        a,b=flux_fit(observed,reconstruction)
        return float(np.mean((a*observed+b-reconstruction)**2)/normalization)
    grid=np.arange(-radius,radius+step/2,step)
    candidates=[]
    for dx in grid:
        for dy in grid:candidates.append((objective([dx,dy]),float(dx),float(dy)))
    best=min(candidates); start=np.array(best[1:])
    result=minimize(objective,start,method='L-BFGS-B',bounds=[(-radius,radius)]*2,
                    options=dict(ftol=1e-14,gtol=1e-10,maxiter=200))
    refined=np.asarray(result.x)
    use_refined=bool(np.isfinite(result.fun) and result.fun<=best[0])
    shift=refined if use_refined else start
    a,b=flux_fit(sample(image,calibration_xy,shift),reconstruction)
    return dict(shift=shift.tolist(),scale=a,offset=b,calibration_objective=objective(shift),
        integer_best_shift=start.tolist(),integer_best_objective=best[0],optimizer_success=bool(result.success),
        optimizer_message=str(result.message),refinement_used=use_refined)
