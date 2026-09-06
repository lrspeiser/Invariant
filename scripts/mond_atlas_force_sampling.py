"""Sparse three-dimensional sampling of centered-gradient potential fields."""
from __future__ import annotations
import numpy as np


def sample_force(potential, origin, spacing, points):
    origin=np.asarray(origin,float);spacing=np.asarray(spacing,float);points=np.asarray(points,float)
    if potential.ndim!=3 or origin.shape!=(3,) or spacing.shape!=(3,) or points.ndim!=2 or points.shape[1]!=3:
        raise ValueError('invalid force sampling shape')
    if not np.isfinite(origin).all() or not np.isfinite(spacing).all() or np.any(spacing<=0) or not np.isfinite(points).all():
        raise ValueError('invalid force sampling coordinates')
    coordinate=(points-origin)/spacing
    index=np.floor(coordinate).astype(np.int64);fraction=coordinate-index
    if np.any(index<1) or np.any(index+2>=np.array(potential.shape)):
        raise ValueError('force stencil reaches the boundary')
    force=np.zeros((len(points),3),float)
    for x in (0,1):
        for y in (0,1):
            for z in (0,1):
                offset=np.array([x,y,z]);nodes=index+offset
                weight=np.prod(np.where(offset,fraction,1-fraction),axis=1)
                for component in range(3):
                    plus=nodes.copy();minus=nodes.copy()
                    plus[:,component]+=1;minus[:,component]-=1
                    derivative=(potential[tuple(plus.T)]-potential[tuple(minus.T)])/(2*spacing[component])
                    force[:,component]-=weight*derivative
    if not np.isfinite(force).all():raise ValueError('nonfinite sampled force')
    return force


def convergence(reference, trial, groups):
    reference=np.asarray(reference,float);trial=np.asarray(trial,float);groups=np.asarray(groups)
    if reference.shape!=trial.shape or reference.ndim!=2 or reference.shape[1]!=3 or groups.shape!=(len(trial),):
        raise ValueError('convergence sampling differs')
    difference=trial-reference
    norm=float(np.sum(trial**2))
    if norm<=0:raise ValueError('zero force cannot normalize convergence')
    values=[]
    for group in np.unique(groups):
        use=groups==group;denom=float(np.sum(trial[use]**2))
        if denom<=0:raise ValueError('zero group force')
        values.append(float(np.sqrt(np.sum(difference[use]**2)/denom)))
    zdenom=float(np.sum(trial[:,2]**2))
    return dict(vector_relative_rms=float(np.sqrt(np.sum(difference**2)/norm)),
        maximum_group_relative_rms=max(values),
        vertical_component_relative_rms=float(np.sqrt(np.sum(difference[:,2]**2)/zdenom)) if zdenom>0 else None)
