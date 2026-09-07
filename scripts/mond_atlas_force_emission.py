"""Theory-only steady surface-column dynamics to thin-ring emission adapter."""
from dataclasses import dataclass
import numpy as np
from mond_atlas_pressure_support import SurfaceColumn, surface_balance
from mond_atlas_motion_controls import observe_particles


@dataclass(frozen=True)
class Projection:
    inclination_deg: float
    position_angle_deg: float
    systemic_km_s: float

    def __post_init__(self):
        if not np.isfinite(list(vars(self).values())).all():
            raise ValueError("Finite projection parameters required")
        if not 0 <= self.inclination_deg <= 90:
            raise ValueError("Inclination must lie within 0 to 90 degrees")


def emitters(column, density_weighted_inward_force, ring_flux, projection,
             nphi=128, *, radial_flow=0.0, vertical_flow=0.0):
    """Force is column-weighted dPhi/dR, never implicitly a midplane sample.

    Integrated ring flux is a separately specified tracer observable. It is
    neither the column mass nor normalized to an observed image here.
    """
    if not isinstance(column, SurfaceColumn) or not isinstance(projection, Projection):
        raise TypeError("SurfaceColumn and Projection required")
    radius = np.asarray(column.radius, float)
    flux = np.asarray(ring_flux, float)
    if (radius.ndim != 1 or flux.shape != radius.shape or radius.size == 0
            or not np.isfinite(flux).all() or np.any(flux < 0)
            or np.any(np.diff(radius) <= 0)):
        raise ValueError("Ordered unique radii and finite nonnegative ring flux required")
    if int(nphi) != nphi or nphi < 4:
        raise ValueError("At least four angular nodes required")
    balance = surface_balance(column, density_weighted_inward_force,
                              radial_flow=radial_flow, vertical_flow=vertical_flow)
    speed = np.repeat(balance.speed(), nphi)
    r = np.repeat(radius, nphi)
    phi = np.tile((np.arange(nphi)+0.5)*2*np.pi/nphi, radius.size)
    inc, pa = np.deg2rad([projection.inclination_deg, projection.position_angle_deg])
    cp, sp = np.cos(phi), np.sin(phi)
    x = r*(np.cos(pa)*cp-np.sin(pa)*np.cos(inc)*sp)
    y = r*(np.sin(pa)*cp+np.cos(pa)*np.cos(inc)*sp)
    depth = r*np.sin(inc)*sp
    los = projection.systemic_km_s + speed*np.sin(inc)*cp
    return dict(x=x, y=y, depth=depth, los=los,
                weights=np.repeat(flux/nphi, nphi), balance=balance)


def render(column, density_weighted_inward_force, ring_flux, projection,
           tracer_dispersion_km_s, instrument, nphi=128, *, radial_flow=0.0,
           vertical_flow=0.0):
    """Tracer dispersion is explicit and independent of the pressure closure."""
    width = np.asarray(tracer_dispersion_km_s, float)
    if width.ndim != 0 or not np.isfinite(width) or width <= 0:
        raise ValueError("Finite positive scalar tracer dispersion required")
    p = emitters(column, density_weighted_inward_force, ring_flux, projection,
                 nphi, radial_flow=radial_flow, vertical_flow=vertical_flow)
    cube, ledger = observe_particles(p['x'], p['y'], p['los'], p['weights'],
                                     float(width), instrument, accounting=True)
    return cube, ledger, p['balance']
