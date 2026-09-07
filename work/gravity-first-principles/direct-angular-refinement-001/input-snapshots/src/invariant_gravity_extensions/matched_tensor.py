"""One gauged C3 tensor potential joined to the admitted exterior potential."""
from __future__ import annotations

import numpy as np

from .potential_join import blend_potential_jets, pack_cartesian
from .tensor_potential import C3TensorPotential


class MatchedTensorPotential:
    """Restore the source gauge before any product-rule joining.

    The exterior alone is evaluated beyond the join. Construction establishes
    representation consistency, not source accuracy or numerical admission.
    """

    def __init__(self, radius, height, mixed, exterior, *, inner=60., outer=80.):
        data = np.array(mixed, dtype=float, copy=True)
        self.gauge = float(data[0, 0, 0, 0])
        data[0, 0] -= self.gauge
        self.near = C3TensorPotential(radius, height, data)
        if (not np.isfinite(inner) or not np.isfinite(outer) or not 0 < inner < outer or
                inner < exterior.minimum_radius or self.near.radius[-1] < outer or self.near.height[-1] < outer):
            raise ValueError('tensor coverage through outer join and admitted exterior required')
        self.exterior, self.inner, self.outer = exterior, inner, outer

    def fields(self, radius, height):
        R, z = np.broadcast_arrays(np.asarray(radius, float), np.asarray(height, float))
        if np.any(~np.isfinite(R)) or np.any(~np.isfinite(z)) or np.any(R < 0):
            raise ValueError('finite nonnegative radii and finite heights required')
        shape = R.shape
        rflat, zflat = R.ravel(), z.ravel()
        spherical = np.hypot(rflat, zflat)
        n = rflat.size
        result = pack_cartesian(np.zeros(n), np.zeros((3, n)), np.zeros((3, 3, n)),
            np.zeros((3, 3, 3, n)), rflat, zflat)
        keys = set(result)-{'radius', 'height'}
        near_mask, far_mask = spherical < self.outer, spherical >= self.outer
        if np.any(near_mask):
            rn, zn = rflat[near_mask], zflat[near_mask]
            near = self.near.fields(rn, zn)
            near['potential'] += self.gauge
            transition = spherical[near_mask] >= self.inner
            if np.any(transition):
                rt, zt = rn[transition], zn[transition]
                selected = {key: near[key][..., transition] for key in keys}
                joined = blend_potential_jets(selected, self.exterior.fields(rt, zt), rt, zt,
                    inner=self.inner, outer=self.outer)
                for key in keys:
                    near[key][..., transition] = joined[key]
            for key in keys:
                result[key][..., near_mask] = near[key]
        if np.any(far_mask):
            far = self.exterior.fields(rflat[far_mask], zflat[far_mask])
            for key in keys:
                result[key][..., far_mask] = far[key]
        return {key: value.reshape(value.shape[:-1]+shape) for key, value in result.items()}
