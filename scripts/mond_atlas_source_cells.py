"""Mass-conserving finite cells for a frozen bilinear/exponential source."""
from __future__ import annotations
import numpy as np
from mond_atlas_source_resolution import cell_projection_matrix, project
from run_mond_atlas_reprojected_fields import vertical_cell_density


def uniform_axis(axis):
    axis = np.asarray(axis, float)
    if axis.ndim != 1 or len(axis) < 3 or not np.isfinite(axis).all():
        raise ValueError('finite uniform axis required')
    spacing = float((axis[-1]-axis[0])/(len(axis)-1))
    if spacing <= 0 or not np.allclose(np.diff(axis),spacing,rtol=1e-11,atol=1e-13*spacing):
        raise ValueError('strictly increasing uniform axis required')
    return axis, spacing


def component_cells(surface, source_axis, axes, vertical_layers, conversion=1.0):
    """Source surface in tracer/pc2; return Msun/kpc2 plane and inverse-kpc z.

    For a CO brightness source, conversion includes the mass/brightness factor.
    The coefficient basis is the continuous bilinear interpolant, with zero
    boundary nodes. Truncation is measured, never silently renormalized.
    """
    nodes,h = uniform_axis(source_axis)
    surface = np.asarray(surface,float)
    if (surface.shape != (len(nodes),len(nodes)) or not np.isfinite(surface).all()
            or np.any(surface < 0) or not np.isfinite(conversion) or conversion <= 0):
        raise ValueError('invalid nonnegative source or conversion')
    if np.any(surface[[0,-1],:]) or np.any(surface[:,[0,-1]]):
        raise ValueError('nonzero boundary nodes do not define the admitted compact source')
    if len(axes) != 3:
        raise ValueError('three field axes required')
    pairs = [uniform_axis(a) for a in axes]
    layers = np.asarray(vertical_layers,float)
    if (layers.ndim != 2 or layers.shape[1] != 2 or not layers.shape[0]
            or not np.isfinite(layers).all() or np.any(layers[:,0] < 0)
            or np.any(layers[:,1] <= 0) or abs(layers[:,0].sum()-1) > 1e-12):
        raise ValueError('normalized positive-height layers required; no sheet substitution')
    x,dx = pairs[0]
    y,dy = pairs[1]
    z,dz = pairs[2]
    converted = surface*(conversion*1e6)
    left = cell_projection_matrix(x,dx,nodes,h,0)
    right = cell_projection_matrix(y,dy,nodes,h,0)
    plane = project(converted,left,right)
    vertical = sum(f*vertical_cell_density(z,dz,height) for f,height in layers)
    full_mass = float(converted.sum()*h*h)
    plane_mass = float(plane.sum()*dx*dy)
    vertical_fraction = float(vertical.sum()*dz)
    finite_mass = plane_mass*vertical_fraction
    if full_mass <= 0:
        raise ValueError('positive source mass required')
    if min(plane.min(),vertical.min()) < 0:
        raise ArithmeticError('negative cell integral')
    if plane_mass > full_mass*(1+1e-11) or vertical_fraction > 1+1e-11:
        raise ArithmeticError('cell integration creates mass')
    return plane,vertical,dict(full_source_mass_msun=full_mass,
        planar_enclosed_mass_msun=plane_mass,vertical_enclosed_fraction=vertical_fraction,
        finite_grid_mass_msun=finite_mass,finite_fraction=finite_mass/full_mass,
        resampling_normalization_applied=False,
        discretization='exact bilinear planar and exponential vertical cell averages')
