"""Geometric HI-support warnings and fixed-segment background diagnostics.

No source selection, gravity fit, or line-free-channel identification occurs here.
"""
from __future__ import annotations
import numpy as np


def block_fraction(support, factor):
    support = np.asarray(support, bool)
    if support.ndim != 2 or not isinstance(factor, int) or factor < 1 or any(n % factor for n in support.shape):
        raise ValueError('support must tile into complete square blocks')
    ny, nx = support.shape
    return support.reshape(ny//factor, factor, nx//factor, factor).mean(axis=(1, 3))


def dilate_disk(support, radius):
    support = np.asarray(support, bool)
    if support.ndim != 2 or not isinstance(radius, int) or radius < 0:
        raise ValueError('invalid integer disk dilation')
    result = np.zeros_like(support); ny, nx = support.shape
    for dy in range(-min(radius, ny-1), min(radius, ny-1)+1):
        for dx in range(-min(radius, nx-1), min(radius, nx-1)+1):
            if dx*dx+dy*dy > radius*radius: continue
            sy = slice(max(0, -dy), min(ny, ny-dy)); sx = slice(max(0, -dx), min(nx, nx-dx))
            ty = slice(max(0, dy), min(ny, ny+dy)); tx = slice(max(0, dx), min(nx, nx+dx))
            result[ty, tx] |= support[sy, sx]
    return result


def channel_segments(channels, end_fraction):
    if channels < 3 or not 0 < end_fraction < .5:
        raise ValueError('invalid spectral partition')
    count = int(np.floor(channels*end_fraction))
    if count < 1 or channels-2*count < 1: raise ValueError('empty spectral segment')
    return dict(band_ends=[(0, count), (channels-count, channels)],
                band_center=[(count, channels-count)])


def segment_pairs(segments, lag, channels):
    if not isinstance(lag, int) or lag < 1: raise ValueError('positive integer lag required')
    covered = set(); left = []; right = []
    for start, stop in segments:
        if not 0 <= start < stop <= channels: raise ValueError('invalid channel segment')
        if covered.intersection(range(start, stop)): raise ValueError('overlapping segments')
        covered.update(range(start, stop))
        left.extend(range(start, stop-lag)); right.extend(range(start+lag, stop))
    return np.array(left, int), np.array(right, int)


def segment_statistics(standardized, segments, lags):
    values = np.asarray(standardized, float)
    if values.ndim != 2 or not np.isfinite(values).all() or values.shape[1] < 1:
        raise ValueError('finite channel by pixel values required')
    segment_pairs(segments, 1, len(values))  # Validate even when no requested lag fits.
    selected = np.concatenate([np.arange(a, b) for a, b in segments])
    data = values[selected]; q05, median, q95 = np.quantile(data, [.05, .5, .95])
    lag_rows = []
    for lag in lags:
        left, right = segment_pairs(segments, lag, len(values))
        if not len(left):
            lag_rows.append(dict(lag=lag, channel_pairs=0, pixel_pairs=0, product=None, normalized_product=None)); continue
        a, b = values[left], values[right]
        product = float(np.mean(a*b)); denominator = float(np.sqrt(np.mean(a*a)*np.mean(b*b)))
        lag_rows.append(dict(lag=lag, channel_pairs=len(left), pixel_pairs=a.size,
            product=product, normalized_product=product/denominator if denominator > 0 else None))
    return dict(channels=len(selected), pixels=values.shape[1], mean_square=float(np.mean(data*data)),
        median=float(median), q05=float(q05), q95=float(q95),
        upper_to_lower_tail_ratio=float((q95-median)/(median-q05)) if median > q05 else None,
        above_median_plus_3_fraction=float(np.mean(data > median+3)),
        below_median_minus_3_fraction=float(np.mean(data < median-3)), lags=lag_rows)


def spectral_diagnostics(cube, calibration, validation, end_fraction, lags):
    """Only selected spatial values enter; test values never set centering/scales."""
    cube = np.asarray(cube); calibration = np.asarray(calibration, bool); validation = np.asarray(validation, bool)
    if cube.ndim != 3 or calibration.shape != cube.shape[1:] or validation.shape != calibration.shape or np.any(calibration & validation):
        raise ValueError('invalid disjoint spatial partition')
    cal, held = cube[:, calibration].astype(float), cube[:, validation].astype(float)
    if cal.shape[1] < 2 or held.shape[1] < 1 or not np.isfinite(cal).all() or not np.isfinite(held).all():
        raise ValueError('insufficient finite selected background values')
    mean = cal.mean(axis=1); std = cal.std(axis=1)
    if np.any(std <= 0): raise ValueError('zero calibration channel variance')
    bands = channel_segments(len(cube), end_fraction)
    result = dict(channel_segments=bands, calibration_pixels=cal.shape[1], validation_pixels=held.shape[1])
    for role, data in [('calibration', cal), ('validation', held)]:
        standardized = (data-mean[:, None])/std[:, None]
        result[role] = {band:segment_statistics(standardized, segments, lags) for band, segments in bands.items()}
    return result


def support_overlap(mask, fraction, expanded):
    mask = np.asarray(mask, bool); fraction = np.asarray(fraction, float); expanded = np.asarray(expanded, bool)
    if mask.shape != fraction.shape or mask.shape != expanded.shape or not mask.any():
        raise ValueError('invalid support overlap mask')
    return dict(pixels=int(mask.sum()), direct_pixels=int(np.sum(mask & (fraction > 0))),
        expanded_pixels=int(np.sum(mask & expanded)),
        direct_pixel_fraction=float(np.mean(fraction[mask] > 0)),
        native_positive_area_fraction=float(np.mean(fraction[mask])),
        expanded_pixel_fraction=float(np.mean(expanded[mask])))
