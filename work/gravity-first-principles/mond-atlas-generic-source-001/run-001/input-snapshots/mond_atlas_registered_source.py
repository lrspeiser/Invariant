"""Conservative, registered flat-disk tracer grids; SOURCE_BLOCKED for 3D use.

Arrays use galaxy major X as axis zero and deprojected minor Y as axis one.
No dynamics, source noise likelihood, instrument deconvolution or gravity fit.
"""
from __future__ import annotations
import numpy as np
from build_mond_atlas_ngc2903_source import linear_cd, sky_vectors


def inclination(geometry):
    if 'inclination_deg' in geometry:
        inc = float(geometry['inclination_deg'])
    else:
        q = 1 - float(geometry['ellipticity'])
        q0 = float(geometry['intrinsic_axis_ratio'])
        if not 0 <= q0 < q <= 1:
            raise ValueError('oblate geometry requires 0 <= q0 < apparent q <= 1')
        inc = float(np.rad2deg(np.arccos(np.sqrt((q*q-q0*q0)/(1-q0*q0)))))
    if not np.isfinite(inc) or not 0 <= inc < 85:
        raise ValueError('source plane requires finite inclination below 85 degrees')
    return inc


def transfer_matrix(p1_header, shift):
    if p1_header['CTYPE1'] != 'RA---TAN' or p1_header['CTYPE2'] != 'DEC--TAN':
        raise ValueError('registered P1 reference must use explicit core TAN')
    shift = np.asarray(shift, float)
    if shift.shape != (2,) or not np.isfinite(shift).all():
        raise ValueError('invalid P1 pixel translation')
    dl, dm = np.deg2rad(linear_cd(p1_header) @ shift)
    center, east, north = sky_vectors(p1_header['CRVAL1'], p1_header['CRVAL2'])
    return np.eye(3) + np.outer(dl*east + dm*north, center)


def source_coordinates(header, xy, geometry, transform=None):
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('finite x/y pixel coordinates required')
    distance = float(geometry['distance_mpc'])*1000
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError('positive distance required')
    if not np.isfinite([geometry['ra_deg'], geometry['dec_deg'], geometry['pa_deg']]).all() or not -90 <= geometry['dec_deg'] <= 90:
        raise ValueError('invalid galaxy sky geometry')
    cd = linear_cd(header)
    projection = header['CTYPE1'][-3:]
    if header['CTYPE1'] != 'RA---'+projection or header['CTYPE2'] != 'DEC--'+projection:
        raise ValueError('explicit celestial axes required')
    plane = np.deg2rad((xy + 1 - [header['CRPIX1'], header['CRPIX2']]) @ cd.T)
    l, m = plane.T
    center, east, north = sky_vectors(header['CRVAL1'], header['CRVAL2'])
    solid = np.full(len(xy), abs(np.linalg.det(cd))*(np.pi/180)**2)
    if projection == 'TAN':
        raw = center + l[:, None]*east + m[:, None]*north
    elif projection == 'SIN':
        radial = 1-l*l-m*m
        if np.any(radial <= 0):
            raise ValueError('SIN outside hemisphere')
        raw = np.sqrt(radial)[:, None]*center + l[:, None]*east + m[:, None]*north
        solid /= np.sqrt(radial)
    else:
        raise ValueError('only core TAN and SIN are supported')
    if transform is not None:
        transform = np.asarray(transform, float)
        if transform.shape != (3, 3) or not np.isfinite(transform).all() or np.linalg.det(transform) <= 0:
            raise ValueError('invalid sky transform')
        raw = raw @ transform.T
        solid *= abs(np.linalg.det(transform))
    norm = np.linalg.norm(raw, axis=1)
    world = raw / norm[:, None]
    solid /= norm**3
    gc, ge, gn = sky_vectors(geometry['ra_deg'], geometry['dec_deg'])
    depth = world @ gc
    if np.any(depth <= 0):
        raise ValueError('source lies outside galaxy tangent hemisphere')
    east_kpc = (world @ ge)/depth*distance
    north_kpc = (world @ gn)/depth*distance
    pa = np.deg2rad(float(geometry['pa_deg']))
    cosi = float(np.cos(np.deg2rad(inclination(geometry))))
    major = east_kpc*np.sin(pa) + north_kpc*np.cos(pa)
    minor = (east_kpc*np.cos(pa) - north_kpc*np.sin(pa))/cosi
    area = solid*distance**2/depth**3
    return major, minor, area, world


def make_grid(grid):
    h, half = float(grid['spacing_kpc']), float(grid['half_width_kpc'])
    ratio = 2*half/h
    if not np.isfinite([h, half]).all() or min(h, half) <= 0 or abs(ratio-round(ratio)) > 1e-9:
        raise ValueError('positive integral grid extent required')
    if not 0 < grid['taper_start_kpc'] < grid['cutoff_kpc'] <= half:
        raise ValueError('invalid finite support/taper')
    if grid['annulus_width_kpc'] <= 0:
        raise ValueError('invalid annulus width')
    if not all(0 < grid[k] <= 1 for k in ('minimum_cell_coverage', 'minimum_annulus_coverage')):
        raise ValueError('invalid coverage thresholds')
    return np.arange(int(round(ratio))+1)*h-half


def rebin_tracer(image, header, good, geometry, grid, conversion=1., transform=None,
                 subdivisions=4, error=None, chunk_size=32768):
    image = np.asarray(image, float)
    good = np.array(good, bool, copy=True)
    if image.ndim != 2 or good.shape != image.shape or not np.isfinite(conversion) or conversion <= 0:
        raise ValueError('invalid image/support/conversion')
    if not isinstance(subdivisions, (int, np.integer)) or subdivisions < 1:
        raise ValueError('positive integer pixel subdivision required')
    if error is not None and np.shape(error) != image.shape:
        raise ValueError('error map shape differs')
    good &= np.isfinite(image)
    iy, ix = np.nonzero(good)
    if len(ix) == 0:
        raise ValueError('no measured support')
    axis = make_grid(grid); n = len(axis); h = grid['spacing_kpc']
    cosi = np.cos(np.deg2rad(inclination(geometry)))
    area_sum = np.zeros(n*n); flux_sum = np.zeros(n*n); error_sum = np.zeros(n*n)
    total_flux = 0.; total_area = 0.
    offsets = (np.arange(subdivisions)+.5)/subdivisions-.5
    for start in range(0, len(ix), chunk_size):
        x = ix[start:start+chunk_size]; y = iy[start:start+chunk_size]
        value = image[y, x]
        for ox in offsets:
            for oy in offsets:
                major, minor, area, _ = source_coordinates(header, np.column_stack((x+ox, y+oy)), geometry, transform)
                area /= subdivisions**2
                total_flux += float(np.sum(value*conversion*area*1e6))
                total_area += float(np.sum(area/cosi))
                bx = np.floor((major-axis[0]+h/2)/h).astype(int)
                by = np.floor((minor-axis[0]+h/2)/h).astype(int)
                inside = (bx >= 0) & (by >= 0) & (bx < n) & (by < n)
                label = bx[inside]*n+by[inside]
                area_sum += np.bincount(label, weights=area[inside]/cosi, minlength=n*n)
                flux_sum += np.bincount(label, weights=value[inside]*conversion*area[inside]*1e6, minlength=n*n)
                if error is not None:
                    error_sum += np.bincount(label, weights=np.asarray(error)[y, x][inside]*conversion*area[inside]*1e6, minlength=n*n)
    area_sum = area_sum.reshape(n, n); flux_sum = flux_sum.reshape(n, n)
    coverage = area_sum/h**2
    mean = np.divide(flux_sum, area_sum*1e6, out=np.full((n,n), np.nan), where=area_sum > 0)
    observed = flux_sum/(h*h*1e6)
    xx, yy = np.meshgrid(axis, axis, indexing='ij'); radius = np.hypot(xx, yy)
    rings = np.floor(radius/grid['annulus_width_kpc']).astype(int); nr = int(rings.max())+1
    ring_area = np.bincount(rings.ravel(), weights=area_sum.ravel(), minlength=nr)
    ring_flux = np.bincount(rings.ravel(), weights=flux_sum.ravel(), minlength=nr)
    ring_coverage = ring_area/(np.bincount(rings.ravel(), minlength=nr)*h*h)
    ring_mean = np.divide(ring_flux, ring_area*1e6, out=np.full(nr, np.nan), where=ring_area > 0)
    qualified = (ring_coverage >= grid['minimum_annulus_coverage']) & np.isfinite(ring_mean)
    if not np.any(qualified):
        raise ValueError('no qualified source annulus')
    index = np.arange(nr)
    fill_profile = np.interp(index, index[qualified], np.maximum(ring_mean[qualified], 0), left=0, right=0)
    trusted = (coverage >= grid['minimum_cell_coverage']) & np.isfinite(mean)
    filled = np.where(trusted, np.maximum(mean, 0), fill_profile[rings])
    taper = np.clip((grid['cutoff_kpc']-radius)/(grid['cutoff_kpc']-grid['taper_start_kpc']), 0, 1)
    zero = np.maximum(observed, 0)*taper; annular = filled*taper
    error_mean = None
    if error is not None:
        error_mean = np.divide(error_sum.reshape(n,n), area_sum*1e6, out=np.full((n,n),np.nan), where=area_sum > 0)
    report = dict(native_supported_pixels=len(ix), pixel_subdivisions=subdivisions,
                  input_signed_integral=total_flux, untapered_in_field_signed_integral=float(flux_sum.sum()),
                  outside_field_signed_integral=total_flux-float(flux_sum.sum()),
                  input_deprojected_covered_area_kpc2=total_area,
                  signed_measured_integral=float(np.sum(observed*taper)*h*h*1e6),
                  negative_projection_added_integral=float(np.sum(zero-observed*taper)*h*h*1e6),
                  conditional_zero_integral=float(zero.sum()*h*h*1e6),
                  conditional_annular_integral=float(annular.sum()*h*h*1e6),
                  observed_area_fraction_inside_cutoff=float(np.mean(np.minimum(coverage,1)[radius < grid['cutoff_kpc']])),
                  maximum_area_coverage=float(coverage.max()), supported_annuli=int(qualified.sum()),
                  missing_flux_is_measured_zero=False, source_noise_likelihood_complete=False,
                  error_map_role='area-weighted native EMOM0 diagnostic only; no propagated covariance' if error is not None else None)
    rows = [dict(radius_kpc=(i+.5)*grid['annulus_width_kpc'], coverage=float(ring_coverage[i]),
                 signed_mean=float(ring_mean[i]) if np.isfinite(ring_mean[i]) else None,
                 accepted=bool(qualified[i]), conditional_fill=float(fill_profile[i])) for i in range(nr)]
    return dict(axis=axis, observed=observed, mean=mean, coverage=coverage, zero=zero, annular=annular, error=error_mean), report, rows
