"""Theory-only thin emitting-ring kinematics, with no force or mass inference.

Coordinate/equation contract and source restrictions are frozen in
work/gravity-first-principles/mond-atlas-motion-controls-001/PREFLIGHT.md.
Only project_emission and spatial_beam are imported from the existing cube code.
The reference path deliberately uses separate geometry, CDF and convolution.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import convolve2d
from scipy.special import ndtr

from mond_atlas_cube import project_emission, spatial_beam


PARAMETERS = (
    "rotation_km_s", "inclination_deg", "position_angle_deg", "systemic_km_s",
    "dispersion_km_s", "warp_inclination_deg", "warp_position_angle_deg",
    "radial_km_s", "asymmetry",
)
CIRCULAR_PARAMETERS = PARAMETERS[:5]


@dataclass(frozen=True)
class Geometry:
    radius_max_kpc: float = 4.0
    scale_kpc: float = 1.4
    turn_kpc: float = 1.1
    asymmetry_phase_deg: float = 35.0

    def __post_init__(self):
        if (not np.isfinite(list(vars(self).values())).all()
                or min(self.radius_max_kpc, self.scale_kpc, self.turn_kpc) <= 0):
            raise ValueError("finite positive physical lengths required")


@dataclass(frozen=True)
class Instrument:
    npix: int = 21
    pixel_kpc: float = 0.5
    beam_sigma_kpc: float = 0.6
    beam_half_width: int = 4
    channel_min_km_s: float = -125.0
    channel_max_km_s: float = 125.0
    nchannel: int = 25

    def __post_init__(self):
        if (not np.isfinite(list(vars(self).values())).all()
                or any(int(n) != n for n in (self.npix, self.nchannel, self.beam_half_width))
                or self.npix < 2 or self.nchannel < 1 or self.beam_half_width < 0
                or self.pixel_kpc <= 0 or self.beam_sigma_kpc < 0
                or self.channel_max_km_s <= self.channel_min_km_s):
            raise ValueError("invalid finite instrument geometry")

    @property
    def edges(self):
        return np.linspace(self.channel_min_km_s, self.channel_max_km_s, self.nchannel + 1)

    @property
    def kernel(self):
        h = self.beam_half_width
        if self.beam_sigma_kpc == 0:
            k = np.zeros((2*h+1, 2*h+1))
            k[h, h] = 1.0
            return k
        a = np.arange(-h, h+1)*self.pixel_kpc/self.beam_sigma_kpc
        k = np.exp(-0.5*(a[:, None]**2+a[None, :]**2))
        return k/k.sum()

    @property
    def omitted_continuous_gaussian_tail(self):
        """Continuous square-tail diagnostic; sampled beam is the exact operator."""
        if self.beam_sigma_kpc == 0:
            return 0.0
        a = (self.beam_half_width+0.5)*self.pixel_kpc/self.beam_sigma_kpc
        return float(1-math.erf(a/math.sqrt(2))**2)


def check_parameters(p):
    if (set(p) != set(PARAMETERS) or not np.isfinite(list(p.values())).all()
            or p["dispersion_km_s"] <= 0 or abs(p["asymmetry"]) >= 1):
        raise ValueError("finite complete parameters, positive dispersion and |asymmetry|<1 required")


@lru_cache(maxsize=32)
def ring_nodes(geometry, nr, nphi):
    """Exact radial-bin emission, midpoint positions, periodic midpoint azimuth.

    The radial integrated-flux fraction is 1-(1+R/h)exp(-R/h). Using its
    difference gives exact annular weights; no output/observed normalization.
    """
    if int(nr) != nr or int(nphi) != nphi or nr < 1 or nphi < 4:
        raise ValueError("positive radial and >=4 azimuthal quadrature counts required")
    edges = np.linspace(0, geometry.radius_max_kpc, nr+1)
    u = edges/geometry.scale_kpc
    cumulative = 1-(1+u)*np.exp(-u)
    radial_weight = np.diff(cumulative)/cumulative[-1]
    radius = np.repeat((edges[:-1]+edges[1:])/2, nphi)
    phi = np.tile((np.arange(nphi)+0.5)*(2*np.pi/nphi), nr)
    weight = np.repeat(radial_weight/nphi, nphi)
    for a in (radius, phi, weight):
        a.flags.writeable = False
    return radius, phi, weight


def projected_particles(p, geometry=Geometry(), nr=48, nphi=144):
    """Return sky x,y,depth [kpc], LOS [km/s], and intrinsic emission weights.

    x/y/z and velocities follow e1,e2 in PREFLIGHT; +z and positive LOS recede.
    This is a snapshot of independent annuli, not a continuous fluid solution.
    """
    check_parameters(p)
    radius, phi, weights = ring_nodes(geometry, nr, nphi)
    warp = (radius/geometry.radius_max_kpc)**2
    inc = np.deg2rad(p["inclination_deg"]+p["warp_inclination_deg"]*warp)
    pa = np.deg2rad(p["position_angle_deg"]+p["warp_position_angle_deg"]*warp)
    cp, sp, ci = np.cos(phi), np.sin(phi), np.cos(inc)
    x = radius*(np.cos(pa)*cp-np.sin(pa)*ci*sp)
    y = radius*(np.sin(pa)*cp+np.cos(pa)*ci*sp)
    z = radius*np.sin(inc)*sp
    law = -np.expm1(-radius/geometry.turn_kpc)
    los = p["systemic_km_s"]+np.sin(inc)*law*(p["rotation_km_s"]*cp+p["radial_km_s"]*sp)
    weights = weights*(1+p["asymmetry"]*np.cos(phi-np.deg2rad(geometry.asymmetry_phase_deg)))
    return x, y, z, los, weights


def deposit_particles(x, y, channel_weights, instrument):
    """Conservative separable tent assignment on a grid with beam-support halo."""
    h, n = instrument.beam_half_width, instrument.npix
    size = n+2*h
    x, y, values = np.asarray(x), np.asarray(y), np.asarray(channel_weights)
    if (x.ndim != 1 or y.shape != x.shape or values.ndim != 2
            or values.shape[1] != len(x) or not np.isfinite(x).all()
            or not np.isfinite(y).all() or not np.isfinite(values).all()
            or np.any(values < -1e-14)):
        raise ValueError("invalid emitting particle arrays")
    u, v = x/instrument.pixel_kpc+(n-1)/2+h, y/instrument.pixel_kpc+(n-1)/2+h
    ix, iy = np.floor(u).astype(int), np.floor(v).astype(int)
    fx, fy = u-ix, v-iy
    cube = np.zeros((values.shape[0], size*size))
    for dx, dy, weight in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                           (0, 1, (1-fx)*fy), (1, 1, fx*fy)):
        xx, yy = ix+dx, iy+dy
        keep = (xx >= 0) & (xx < size) & (yy >= 0) & (yy < size)
        idx = yy[keep]*size+xx[keep]
        for channel in range(values.shape[0]):
            cube[channel] += np.bincount(idx, weights=values[channel, keep]*weight[keep],
                                         minlength=size*size)
    return cube.reshape(values.shape[0], size, size)


def observe_particles(x, y, los, weights, dispersion, instrument, accounting=False):
    """Integrate channels through the shared primitive, then spatially project.

    The primitive sees each emitter as a unit-path, one-depth spatial sample.
    Its spatial samples are particles until the subsequent explicit deposition.
    No observed cube is treated as physical source density.
    """
    weights = np.asarray(weights, float)
    spectral = project_emission(weights[None, None, :], np.asarray(los)[None, None, :],
                                dispersion, instrument.edges, 1.0)[:, 0, :]
    extended = deposit_particles(x, y, spectral, instrument)
    blurred = spatial_beam(extended, instrument.kernel)
    h, n = instrument.beam_half_width, instrument.npix
    science = blurred[:, h:h+n, h:h+n]
    if not accounting:
        return science
    inner = extended[:, h:h+n, h:h+n]
    inner_only = spatial_beam(inner, instrument.kernel)
    intrinsic, band, ext = float(weights.sum()), float(spectral.sum()), float(extended.sum())
    before, after_inner, final = float(inner.sum()), float(inner_only.sum()), float(science.sum())
    return science, {
        "intrinsic_flux": intrinsic, "finite_band_flux": band,
        "spectral_loss": intrinsic-band, "outside_halo_assignment_loss": band-ext,
        "before_beam_in_science_field": before, "before_beam_in_halo": ext-before,
        "beam_export_from_science_field": before-after_inner,
        "beam_import_from_halo": final-after_inner,
        "extended_grid_beam_boundary_loss": ext-float(blurred.sum()),
        "final_science_flux": final, "total_spatial_loss_after_band": band-final,
        "total_loss": intrinsic-final,
        "flux_accounting_residual": final-(before-(before-after_inner)+(final-after_inner)),
        "beam_continuous_square_tail_diagnostic": instrument.omitted_continuous_gaussian_tail,
    }


def forward_cube(p, geometry=Geometry(), instrument=Instrument(), nr=48, nphi=144,
                 accounting=False):
    x, y, _, los, weights = projected_particles(p, geometry, nr, nphi)
    return observe_particles(x, y, los, weights, p["dispersion_km_s"], instrument, accounting)


def direct_reference_cube(p, geometry, instrument, nr=96, nphi=288):
    """Separately implemented direct quadrature; no production geometry/primitives.

    Gauss-Legendre radius; offset periodic azimuth; Rx then Rz rotation matrices;
    direct tent integrals at every supported image sample; scipy.special.ndtr;
    scipy.signal.convolve2d. Shared inputs are only the declared physical contract.
    """
    check_parameters(p)
    roots, wg = np.polynomial.legendre.leggauss(nr)
    radii = (roots+1)*geometry.radius_max_kpc/2
    dr = wg*geometry.radius_max_kpc/2
    angles = (np.arange(nphi)+0.271828)*2*np.pi/nphi
    cosine, sine = np.cos(angles), np.sin(angles)
    edge = instrument.edges
    h, n = instrument.beam_half_width, instrument.npix
    size = n+2*h
    xy = (np.arange(size)-(n-1)/2-h)*instrument.pixel_kpc
    yy, xx = np.meshgrid(xy, xy, indexing="ij")
    extent_x, extent_y = xx.ravel(), yy.ravel()
    image = np.zeros((instrument.nchannel, size*size))
    umax = geometry.radius_max_kpc/geometry.scale_kpc
    normalization = 2*np.pi*geometry.scale_kpc**2*(1-(1+umax)*np.exp(-umax))
    for radius, radial_step in zip(radii, dr):
        f = (radius/geometry.radius_max_kpc)**2
        i = math.radians(p["inclination_deg"]+f*p["warp_inclination_deg"])
        a = math.radians(p["position_angle_deg"]+f*p["warp_position_angle_deg"])
        rx = np.array([[1, 0, 0], [0, math.cos(i), -math.sin(i)],
                       [0, math.sin(i), math.cos(i)]])
        rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        transform = rz @ rx
        xyz = transform @ np.array([radius*cosine, radius*sine, np.zeros(nphi)])
        shape = 1-math.exp(-radius/geometry.turn_kpc)
        tangential, radial = p["rotation_km_s"]*shape, p["radial_km_s"]*shape
        velocity = transform @ np.array([radial*cosine-tangential*sine,
                                         radial*sine+tangential*cosine, np.zeros(nphi)])
        mu = velocity[2]+p["systemic_km_s"]
        flux = (radius*math.exp(-radius/geometry.scale_kpc)*radial_step*2*np.pi/nphi
                /normalization*(1+p["asymmetry"]*np.cos(angles-math.radians(geometry.asymmetry_phase_deg))))
        channel = np.diff(ndtr((edge[:, None]-mu)/p["dispersion_km_s"]), axis=0)*flux
        # Independent gather at all pixels touched by this ring (no four-corner scatter).
        wx = np.maximum(1-np.abs(xyz[0, :, None]-extent_x)/instrument.pixel_kpc, 0)
        wy = np.maximum(1-np.abs(xyz[1, :, None]-extent_y)/instrument.pixel_kpc, 0)
        assignment = wx*wy
        used = np.any(assignment > 0, axis=0)
        image[:, used] += channel @ assignment[:, used]
    images = image.reshape(instrument.nchannel, size, size)
    # Construct the beam separately from Instrument.kernel.
    if instrument.beam_sigma_kpc == 0:
        beam = np.zeros((2*h+1, 2*h+1))
        beam[h, h] = 1
    else:
        coords = np.arange(-h, h+1)*instrument.pixel_kpc
        beam = np.exp(-(coords[:, None]**2+coords[None, :]**2)/(2*instrument.beam_sigma_kpc**2))
        beam /= beam.sum()
    return np.stack([convolve2d(a, beam, mode="same", boundary="fill") for a in images])[:, h:h+n, h:h+n]


def relative_l1(a, b):
    return float(np.sum(np.abs(a-b))/max(float(np.sum(np.abs(b))), 1e-300))


def fixed_splits(shape):
    """Four disjoint sets partition every voxel without inspecting flux or noise."""
    c, y, x = np.indices(shape)
    hc, hp = c % 3 == 0, (x+2*y) % 3 == 0
    return {"train": ~hc & ~hp, "heldout_channels": hc & ~hp,
            "heldout_pixels": ~hc & hp, "heldout_joint": hc & hp}


def known_noise_sigma(shape, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("known positive noise scale required")
    c, _, x = np.indices(shape)
    return scale*(1+0.15*np.cos(2*np.pi*c/shape[0]))*(1+0.1*x/(shape[2]-1))


def fit_model(data, sigma, train_mask, geometry, instrument, config, expanded,
              fixed=None, quadrature=None, max_nfev=100):
    """Fit only supplied training cells; starts/bounds never use held-out values."""
    fixed = fixed or {}
    names = [k for k in (PARAMETERS if expanded else CIRCULAR_PARAMETERS) if k not in fixed]
    limits = config["parameter_bounds"]
    lo, hi = np.array([limits[k][0] for k in names]), np.array([limits[k][1] for k in names])
    scales = hi-lo
    nr, nphi = quadrature or config["quadrature"]["fit"]
    if (data.shape != sigma.shape or data.shape != train_mask.shape
            or train_mask.dtype != bool or not train_mask.any()
            or not np.isfinite(data[train_mask]).all() or not np.isfinite(sigma).all()
            or np.any(sigma <= 0)):
        raise ValueError("invalid training data/covariance/mask")
    # Deliberately slice and copy here: the optimizer has no held-out response access.
    target, uncertainty = data[train_mask].copy(), sigma[train_mask].copy()

    def parameters(unit):
        p = dict(config["base_parameters"])
        p.update({k: 0.0 for k in PARAMETERS[5:]})
        p.update(zip(names, lo+scales*unit))
        p.update(fixed)
        return p

    def residual(unit):
        model = forward_cube(parameters(unit), geometry, instrument, nr, nphi)
        return (model[train_mask]-target)/uncertainty

    candidates = []
    for start in config["study"]["starts"]:
        initial = dict(zip(PARAMETERS, start))
        u = (np.array([initial[k] for k in names])-lo)/scales
        fit = least_squares(residual, u, bounds=(np.zeros(len(names)), np.ones(len(names))),
                            max_nfev=max_nfev, ftol=1e-7, xtol=1e-7, gtol=1e-7)
        candidates.append((fit, {"training_q": float(2*fit.cost), "success": bool(fit.success),
                                  "status": int(fit.status), "message": fit.message,
                                  "nfev": int(fit.nfev), "parameters": parameters(fit.x)}))
    best_index = int(np.argmin([a[1]["training_q"] for a in candidates]))
    best, _ = candidates[best_index]
    p = parameters(best.x)
    singular = np.linalg.svd(best.jac, compute_uv=False)
    threshold = max(float(singular[0])*1e-8, 1e-10)
    rank = int(np.count_nonzero(singular > threshold))
    norms = np.linalg.norm(best.jac, axis=0)
    column_cosines = []
    for j, name in enumerate(names):
        for k in range(j):
            if norms[j]*norms[k] > 1e-14:
                cosine = float(np.dot(best.jac[:, j], best.jac[:, k])/(norms[j]*norms[k]))
                column_cosines.append({"parameters": [names[k], name], "cosine": cosine})
    column_cosines.sort(key=lambda row: abs(row["cosine"]), reverse=True)
    receipt = {
        "model": "expanded" if expanded else "circular_only", "parameters": p,
        "free_parameters": names, "fixed_parameters": fixed, "selected_start": best_index,
        "starts": [a[1] for a in candidates], "training_q": float(2*best.cost),
        "optimizer_success": bool(best.success), "local_scaled_jacobian_rank": rank,
        "local_scaled_jacobian_singular_values": singular.tolist(),
        "local_scaled_jacobian_column_norms": dict(zip(names, norms.tolist())),
        "largest_local_sensitivity_cosines": column_cosines[:5],
        "bound_contacts": [k for k, u in zip(names, best.x) if min(u, 1-u) < 1e-4],
        "identifiability_note": "Local sensitivity in bound-width-scaled coordinates, not posterior coverage or unique global recovery.",
    }
    return p, forward_cube(p, geometry, instrument, nr, nphi), receipt


def evaluate_prediction(prediction, data, truth, sigma, splits):
    return {name: {
        "n": int(mask.sum()),
        "weighted_data_error_per_voxel": float(np.mean(((prediction[mask]-data[mask])/sigma[mask])**2)),
        "weighted_truth_error_per_voxel": float(np.mean(((prediction[mask]-truth[mask])/sigma[mask])**2)),
    } for name, mask in splits.items()}


def numerical_controls(config):
    """All target-free gates execute before any random injection data generation."""
    g, ins = Geometry(**config["geometry"]), Instrument(**config["instrument"])
    base, gate = dict(config["base_parameters"]), config["gates"]
    records = []

    def upper(name, value, tolerance, **detail):
        records.append({"name": name, "value": float(value), "tolerance": float(tolerance),
                        "passed": bool(np.isfinite(value) and value <= tolerance), **detail})

    p = dict(base, warp_inclination_deg=8.0, warp_position_angle_deg=16.0,
             radial_km_s=22.0, asymmetry=0.3)
    x, y, z, los, weights = projected_particles(p, g, 13, 36)
    radii, angles, _ = ring_nodes(g, 13, 36)
    reference_xyz, reference_los = [], []
    for radius, phi in zip(radii, angles):
        w = (radius/g.radius_max_kpc)**2
        i = math.radians(p["inclination_deg"]+p["warp_inclination_deg"]*w)
        a = math.radians(p["position_angle_deg"]+p["warp_position_angle_deg"]*w)
        rx = np.array([[1, 0, 0], [0, math.cos(i), -math.sin(i)], [0, math.sin(i), math.cos(i)]])
        rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        position = np.array([radius*math.cos(phi), radius*math.sin(phi), 0])
        shape = 1-math.exp(-radius/g.turn_kpc)
        velocity = np.array([p["radial_km_s"]*math.cos(phi)-p["rotation_km_s"]*math.sin(phi),
                             p["radial_km_s"]*math.sin(phi)+p["rotation_km_s"]*math.cos(phi), 0])*shape
        reference_xyz.append(rz @ rx @ position)
        reference_los.append((rz @ rx @ velocity)[2]+p["systemic_km_s"])
    upper("independent_cartesian_position_kpc", np.max(np.abs(np.array([x, y, z]).T-reference_xyz)), gate["geometry_absolute"])
    upper("independent_cartesian_velocity_km_s", np.max(np.abs(los-reference_los)), gate["geometry_absolute"])
    upper("analytic_intrinsic_flux", abs(weights.sum()-1), gate["flux_relative"])

    face = dict(base, inclination_deg=0, radial_km_s=25)
    face_particles = projected_particles(face, g, 12, 32)
    upper("face_on_planar_velocity", np.max(np.abs(face_particles[3]-base["systemic_km_s"])), gate["geometry_absolute"])
    face_other = dict(face, rotation_km_s=155, radial_km_s=-30)
    upper("face_on_full_cube_degeneracy", relative_l1(forward_cube(face, g, ins), forward_cube(face_other, g, ins)), gate["geometry_absolute"])
    edge = dict(base, inclination_deg=90, position_angle_deg=0)
    ex, ey, ez, ev, _ = projected_particles(edge, g, 12, 32)
    er, ep, _ = ring_nodes(g, 12, 32)
    upper("edge_on_projected_minor_axis", np.max(np.abs(ey)), gate["geometry_absolute"])
    upper("edge_on_velocity", np.max(np.abs(ev-base["rotation_km_s"]*(-np.expm1(-er/g.turn_kpc))*np.cos(ep))), gate["geometry_absolute"])
    upper("edge_on_depth", np.max(np.abs(ez-er*np.sin(ep))), gate["geometry_absolute"])
    normal = dict(base, radial_km_s=21, systemic_km_s=0)
    reverse = dict(normal, rotation_km_s=-normal["rotation_km_s"], radial_km_s=-normal["radial_km_s"])
    upper("velocity_sign_channel_reversal", relative_l1(forward_cube(reverse, g, ins), forward_cube(normal, g, ins)[::-1]), gate["geometry_absolute"])
    # Rigid rotation is independently manufactured: v = Omega n cross r.
    a, i, omega = np.deg2rad(base["position_angle_deg"]), np.deg2rad(base["inclination_deg"]), 12.0
    rx, ry, rz0, _, rw = projected_particles(base, g, 12, 32)
    norm = np.array([np.sin(a)*np.sin(i), -np.cos(a)*np.sin(i), np.cos(i)])
    manufactured = omega*np.cross(norm, np.array([rx, ry, rz0]).T)[:, 2]
    scalar = omega*np.sin(i)*(rx*np.cos(a)+ry*np.sin(a))
    upper("rigid_rotation_los", np.max(np.abs(manufactured-scalar)), gate["geometry_absolute"])
    # The actual production velocity law also tends to Omega*R as Rturn -> infinity
    # with V0=Omega*Rturn. expm1 avoids cancellation in this independent limit.
    rigid_g = replace(g, turn_kpc=1e14)
    rigid_p = dict(base, rotation_km_s=omega*rigid_g.turn_kpc, systemic_km_s=0)
    rigid_los = projected_particles(rigid_p, rigid_g, 12, 32)[3]
    upper("production_law_rigid_rotation_limit", np.max(np.abs(rigid_los-scalar)), gate["geometry_absolute"])
    mcube = observe_particles(rx, ry, manufactured, rw, base["dispersion_km_s"], ins)
    scube = observe_particles(rx, ry, scalar, rw, base["dispersion_km_s"], ins)
    upper("rigid_rotation_channel_cube", relative_l1(mcube, scube), gate["geometry_absolute"])

    edges = np.linspace(-100, 100, 31)
    mu = np.array([-70, -0.2, 30.3, 120.0])
    ww = np.array([0.2, 0.1, 0.3, 0.4])
    integrated = project_emission(ww[None, None], mu[None, None], 7, edges, 1)[:, 0]
    analytic = np.diff(ndtr((edges[:, None]-mu)/7), axis=0)*ww
    upper("independent_gaussian_channels", np.max(np.abs(integrated-analytic)), gate["gaussian_reference_absolute_per_unit_flux"])

    reference = direct_reference_cube(p, g, ins, *config["quadrature"]["truth"])
    levels = [forward_cube(p, g, ins, *q) for q in config["quadrature"]["convergence"]]
    errors = [relative_l1(v, reference) for v in levels]
    upper("independent_direct_cube", errors[1], gate["direct_cube_relative_l1"], errors=errors)
    upper("fine_quadrature_error", errors[-1], gate["fine_quadrature_relative_l1"])
    upper("fine_to_coarse_error_ratio", errors[-1]/max(errors[0], 1e-300), gate["fine_error_to_coarse_error_max"])
    double_channels = replace(ins, nchannel=ins.nchannel*2)
    fine_spectrum = forward_cube(p, g, double_channels)
    upper("spectral_rebinning", relative_l1(fine_spectrum.reshape(ins.nchannel, 2, ins.npix, ins.npix).sum(axis=1), levels[1]), gate["spectral_rebin_relative_l1"])
    scaled_g = replace(g, radius_max_kpc=g.radius_max_kpc*1000, scale_kpc=g.scale_kpc*1000, turn_kpc=g.turn_kpc*1000)
    scaled_i = replace(ins, pixel_kpc=ins.pixel_kpc*1000, beam_sigma_kpc=ins.beam_sigma_kpc*1000)
    upper("length_unit_rescaling", relative_l1(forward_cube(p, scaled_g, scaled_i), levels[1]), gate["length_scale_relative_l1"])
    zero = dict(base, warp_inclination_deg=0, warp_position_angle_deg=0, radial_km_s=0, asymmetry=0)
    upper("amplitude_zero_nesting", relative_l1(forward_cube(base, g, ins), forward_cube(zero, g, ins)), gate["geometry_absolute"])

    crop = replace(ins, npix=9, channel_min_km_s=-25, channel_max_km_s=25, nchannel=10)
    narrow, loss = forward_cube(p, g, crop, accounting=True)
    ref_narrow = direct_reference_cube(p, g, crop)
    upper("cropped_direct_cube", relative_l1(narrow, ref_narrow), gate["direct_cube_relative_l1"], accounting=loss)
    upper("loss_balance", abs(loss["intrinsic_flux"]-loss["spectral_loss"]-loss["total_spatial_loss_after_band"]-loss["final_science_flux"]), gate["flux_relative"])
    records.append({"name": "finite_band_and_spatial_losses_retained", "passed": bool(loss["spectral_loss"] > 0.05 and loss["total_spatial_loss_after_band"] > 0.001 and loss["beam_import_from_halo"] > 0), "accounting": loss})
    # Exact center impulse + exact outside-field impulse: independently sum kernel taps.
    tiny = replace(ins, npix=7, beam_half_width=2, nchannel=1, channel_min_km_s=-1000, channel_max_km_s=1000)
    impulse_x = np.array([0., 2.0])
    impulse_y = np.array([0., 0.])
    impulse, iloss = observe_particles(impulse_x, impulse_y, np.zeros(2), np.array([0.6, 0.4]), 7, tiny, True)
    expected = np.zeros((7, 7))
    for xx, weight in zip([3, 7], [0.6, 0.4]):
        for ky in range(5):
            for kx in range(5):
                outx, outy = xx+kx-2, 3+ky-2
                if 0 <= outx < 7 and 0 <= outy < 7:
                    expected[outy, outx] += weight*tiny.kernel[ky, kx]
    upper("outside_field_beam_inscatter", np.max(np.abs(impulse[0]-expected)), gate["beam_boundary_absolute_per_unit_flux"], accounting=iloss)
    full = replace(ins, npix=31, beam_half_width=4, channel_min_km_s=-1000, channel_max_km_s=1000)
    upper("wide_field_band_flux_conservation", abs(forward_cube(p, g, full).sum()-1), gate["flux_relative"])

    # Unresolved spectrum: rotation/radial phase rotation leaves azimuthal integral unchanged.
    r, phi, weight = ring_nodes(g, 48, 2880)
    amp, rad = 100., 30.
    shape = -np.expm1(-r/g.turn_kpc)
    first = np.sin(i)*shape*(amp*np.cos(phi)+rad*np.sin(phi))
    second = np.sin(i)*shape*np.hypot(amp, rad)*np.cos(phi)
    s1 = (np.diff(ndtr((ins.edges[:, None]-first)/7), axis=0)*weight).sum(axis=1)
    s2 = (np.diff(ndtr((ins.edges[:, None]-second)/7), axis=0)*weight).sum(axis=1)
    upper("unresolved_rotation_radial_degeneracy", relative_l1(s1, s2), gate["geometry_absolute"], rotation_radial_pair=[amp, rad], equivalent_circular_speed=float(np.hypot(amp, rad)))
    return {"all_passed": all(r["passed"] for r in records), "controls": records,
            "quadrature_reference_relative_l1": errors,
            "disposition": "THEORY_BENCHMARK_ONLY" if all(r["passed"] for r in records) else "BENCHMARK_FAILED"}
