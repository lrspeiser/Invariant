"""Conditional THINGS selection operators; CPU only, no gravity response access."""
from __future__ import annotations

import re
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import erf

FWHM_SIGMA = np.sqrt(8 * np.log(2))
MAD_NORMAL = 1.482602218505602


def beam_from_history(header):
    records = []
    for line in header.get('HISTORY', []):
        m = re.search(r'AIPS\s+CLEAN BMAJ=\s*([\d.E+-]+) BMIN=\s*([\d.E+-]+) BPA=\s*([\d.E+-]+)', line)
        if m:
            records.append(tuple(float(v) for v in m.groups()))
    if not records or len(set(records)) != 1:
        raise ValueError('unique AIPS restoring beam required')
    major, minor, pa = records[0]
    return dict(major_arcsec=major * 3600, minor_arcsec=minor * 3600, pa_deg=pa)


def beam_covariance(major, minor, pa, dy, dx):
    """Gaussian covariance in array (y,x), from signed north/east pixel scales."""
    if not 0 < minor <= major or dx == 0 or dy == 0:
        raise ValueError('invalid beam or pixel dimensions')
    angle = np.deg2rad(pa)
    # Major axis: (north,east) = (cos PA,sin PA).
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    angular = rotation @ np.diag(np.square([major, minor]) / FWHM_SIGMA**2) @ rotation.T
    transform = np.diag([1 / dy, 1 / dx])
    return transform @ angular @ transform


def gaussian_kernel(covariance, truncate=5):
    covariance = np.asarray(covariance, float)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T):
        raise ValueError('symmetric finite 2D covariance required')
    eig = np.linalg.eigvalsh(covariance)
    if eig.min() <= 0 or truncate < 3:
        raise ValueError('positive covariance and adequate support required')
    radius = int(np.ceil(truncate * np.sqrt(eig.max())))
    grid = np.stack(np.mgrid[-radius:radius+1, -radius:radius+1], axis=-1)
    exponent = np.einsum('...i,ij,...j->...', grid, np.linalg.inv(covariance), grid)
    kernel = np.exp(-0.5 * exponent)
    return kernel / kernel.sum()


def convolve_spatial(cube, kernel):
    data = np.asarray(cube)
    if data.ndim not in (2, 3) or not np.isfinite(data).all():
        raise ValueError('finite plane or spectral cube required')
    k = kernel if data.ndim == 2 else kernel[None]
    return fftconvolve(data, k, mode='same', axes=(-2, -1))


def select_runs(cube, sigma, threshold=2, count=3, support=None):
    cube = np.asarray(cube)
    sigma = np.asarray(sigma)
    if cube.ndim != 3 or not isinstance(count, int) or not 1 <= count <= len(cube):
        raise ValueError('invalid cube or run length')
    if sigma.ndim == 1:
        sigma = sigma[:, None, None]
    if not np.isfinite(sigma).all() or np.any(sigma <= 0) or threshold <= 0:
        raise ValueError('positive finite thresholds required')
    above = np.isfinite(cube) & (cube > threshold * sigma)
    starts = np.logical_and.reduce([above[i:i+len(cube)-count+1] for i in range(count)])
    result = np.zeros_like(above)
    for i in range(count):
        result[i:i+len(starts)] |= starts
    if support is not None:
        result &= np.asarray(support, bool)[None]
    return result


def spectral_matrix(n, branch):
    """Finite filter on a padded white pre-grid, in units of stored channel spacing."""
    if branch == 'boxcar_independent':
        return np.eye(n), np.arange(n, dtype=float), 1.0
    if branch == 'boxcar_hanning_full':
        matrix = np.zeros((n, n+2))
        for i in range(n):
            matrix[i, i:i+3] = [.25, .5, .25]
        return matrix, np.arange(n+2, dtype=float)-1, 1.0
    if branch == 'boxcar_hanning_decimated':
        matrix = np.zeros((n, 2*n+1))
        for i in range(n):
            matrix[i, 2*i:2*i+3] = [.25, .5, .25]
        return matrix, (np.arange(2*n+1, dtype=float)-1)/2, 0.5
    raise ValueError('undeclared spectral branch')


def integrated_gaussian(centers, center, fwhm, width):
    if fwhm <= 0 or width <= 0:
        raise ValueError('positive Gaussian and cell widths required')
    sigma = fwhm/FWHM_SIGMA
    hi = (np.asarray(centers)+width/2-center)/(np.sqrt(2)*sigma)
    lo = (np.asarray(centers)-width/2-center)/(np.sqrt(2)*sigma)
    return sigma*np.sqrt(np.pi/2)*(erf(hi)-erf(lo))/width


def robust_channels(cube, support):
    values = np.asarray(cube)[:, support]
    if values.shape[1] < 10 or not np.isfinite(values).all():
        raise ValueError('insufficient finite calibration samples')
    med = np.median(values, axis=1)
    sigma = MAD_NORMAL*np.median(np.abs(values-med[:, None]), axis=1)
    if np.any(sigma <= 0):
        raise ValueError('nonpositive calibration scale')
    return med, sigma


def covariance_diagnostics(cube, support):
    med, sigma = robust_channels(cube, support)
    z = (cube[:, support]-med[:, None])/sigma[:, None]
    # Untapered, unclipped descriptive moments; sample counts are correlated pixels.
    covariance = np.cov(z, bias=True)
    scale = np.sqrt(np.diag(covariance))
    corr = covariance/scale[:, None]/scale[None]
    return dict(pixels=int(support.sum()), median_scale=float(np.median(sigma)),
                lag_correlations={str(k): float(np.diag(corr, k).mean()) for k in [1, 2, 3, 6, 12]},
                positive_3mad_fraction=float((z > 3).mean()), negative_3mad_fraction=float((z < -3).mean())), covariance


def source_templates(shape, pixel_arcsec, spatial_fwhm, spectral_fwhm, phase,
                     center_parent, branch, operator, output_indices, native_kernel, extra_kernel):
    h, grid, width = spectral_matrix(operator.shape[1], branch)
    pre = h @ integrated_gaussian(grid, center_parent+phase, spectral_fwhm, width)
    post = operator @ pre
    yy, xx = np.indices(shape)
    spatial = np.exp(-4*np.log(2)*pixel_arcsec**2 *
                     ((yy-(shape[0]-1)/2)**2+(xx-(shape[1]-1)/2)**2)/spatial_fwhm**2)
    native_spatial = convolve_spatial(spatial, native_kernel)
    detection_spatial = convolve_spatial(native_spatial, extra_kernel)
    scale = pre[output_indices].max()*detection_spatial.max()
    native = post[:, None, None]*native_spatial[None]/scale
    detection = post[:, None, None]*detection_spatial[None]/scale
    positive_reference = pre[output_indices, None, None]*native_spatial[None]/scale
    return native, detection, positive_reference


def recovery(native_background, detection_background, native_source, detection_source,
             positive_source, sigma, amplitude, flux_factor, baseline_mask=None):
    mask = select_runs(detection_background+amplitude*detection_source, sigma)
    baseline = select_runs(detection_background, sigma) if baseline_mask is None else baseline_mask
    truth = amplitude*positive_source
    total = truth.sum()
    selected = (native_background+amplitude*native_source)[mask].sum()
    baseline_flux = native_background[baseline].sum()
    center = np.unravel_index(np.argmax(detection_source), detection_source.shape)
    return dict(peak_selected=bool(mask[center]),
                true_flux_fraction_retained=float(truth[mask].sum()/total),
                reference_flux_jy_kms=float(total*flux_factor),
                post_continuum_flux_over_reference=float(amplitude*native_source.sum()/total),
                selected_noisy_flux_over_reference=float(selected/total),
                paired_selected_flux_difference_over_reference=float((selected-baseline_flux)/total),
                selected_voxel_fraction=float(mask.mean()))


def conditional_noise(rng, shape, covariance, native_kernel, extra_kernel, detector_scale):
    """Extended spatial draw then valid interior crop; all spectral covariance retained."""
    radius = native_kernel.shape[0]//2 + extra_kernel.shape[0]//2
    extended = (shape[0]+2*radius, shape[1]+2*radius)
    white = rng.normal(size=(len(covariance), *extended))
    eig, vectors = np.linalg.eigh(covariance)
    if eig.min() < -1e-10:
        raise ValueError('non-PSD covariance')
    root = vectors*np.sqrt(np.maximum(eig, 0))
    correlated = (root @ white.reshape(len(covariance), -1)).reshape(white.shape)
    native = convolve_spatial(correlated, native_kernel)
    detector = convolve_spatial(native, extra_kernel)
    composed = fftconvolve(native_kernel, extra_kernel, mode='full')
    scale = detector_scale/np.sqrt(np.diag(covariance).mean()*np.square(composed).sum())
    cut = np.s_[:, radius:radius+shape[0], radius:radius+shape[1]]
    return native[cut]*scale, detector[cut]*scale
