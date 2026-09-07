"""Assemble one matched potential on a cylindrical product grid.

Only the active provider is used outside the transition. The entire assembled
field still needs source, derivative and refinement checks; smooth joining
does not imply that its input quadratures are accurate.
"""
from __future__ import annotations

import numpy as np

from .potential_join import blend_potential_jets


def matched_grid(near, exterior, radius, height, *, inner, outer):
    radius, height = np.asarray(radius, float), np.asarray(height, float)
    if (radius.ndim != 1 or height.ndim != 1 or np.any(radius < 0) or
            np.any(~np.isfinite(radius)) or np.any(~np.isfinite(height)) or
            not np.isfinite(inner) or not np.isfinite(outer) or not 0 < inner < outer or
            inner < exterior.minimum_radius or np.shape(near['potential']) != (len(radius), len(height))):
        raise ValueError('finite product grid and an admitted exterior join interval required')
    R, z = np.meshgrid(radius, height, indexing='ij')
    r = np.hypot(R, z)
    transition = (r >= inner) & (r < outer)
    outside = r >= outer
    result = {k: np.array(v, copy=True) for k, v in near.items()}
    result['radius'], result['height'] = radius, height
    keys = set(result)-{'radius', 'height'}
    if np.any(transition):
        far = exterior.fields(R[transition], z[transition])
        selected = {k: result[k][..., transition] for k in keys}
        joined = blend_potential_jets(selected, far, R[transition], z[transition], inner=inner, outer=outer)
        for k in keys:
            result[k][..., transition] = joined[k]
    if np.any(outside):
        far = exterior.fields(R[outside], z[outside])
        for k in keys:
            result[k][..., outside] = far[k]
    if any(np.any(~np.isfinite(result[k])) for k in keys):
        raise ValueError('nonfinite derivative in the active matched potential')
    return result
