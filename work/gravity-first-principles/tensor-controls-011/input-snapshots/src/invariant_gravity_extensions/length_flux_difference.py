"""Direct length-minus-zero flux for the existing static length action.

An exact logarithmic rearrangement replaces subtraction of the leading
coefficients. No quadrature or physical interpolation between laws is added.
"""
from __future__ import annotations

import numpy as np


def excess_derivative_change(spec, x, h):
    """E'(x+h)-E'(x), retaining a small nonnegative shift h explicitly."""
    x, h = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(h, dtype=float))
    if np.any(x < 0) or np.any(h < 0) or not np.all(np.isfinite([x, h])):
        raise ValueError('finite nonnegative invariants required')
    log_v = np.log(x+spec.epsilon**2)
    shift = np.log1p(h/(x+spec.epsilon**2))
    a, b = spec.shape*log_v, spec.shape*shift
    sigmoid = np.exp(-np.logaddexp(0, -a))
    difference = np.empty_like(b)
    small = b < 50
    difference[small] = np.log1p(sigmoid[small]*np.expm1(b[small]))
    # For a large positive shift, subtraction of softplus values is well
    # conditioned relative to the accompanying -shift/4 term. Avoid exp(b).
    difference[~small] = np.logaddexp(0, (a+b)[~small])-np.logaddexp(0, a[~small])
    log_ratio = -.25*shift-(1+.75/spec.shape)*difference
    return spec.excess_derivatives(x)[0]*np.expm1(log_ratio)


def length_flux_difference(spec, gradient, hessian, gradient_hessian_norm_squared,
                           gradient_laplacian, length, a0=1.):
    """Return J(ell)-J(0) in Cartesian components, including reaction terms."""
    p, H, dH2, dlap = [np.asarray(v, dtype=float) for v in
                      (gradient, hessian, gradient_hessian_norm_squared, gradient_laplacian)]
    if (p.ndim < 1 or p.shape[0] != 3 or H.shape != (3, *p.shape)
            or dH2.shape != p.shape or dlap.shape != p.shape
            or any(np.any(~np.isfinite(v)) for v in (p, H, dH2, dlap))
            or not np.isfinite(length) or length < 0 or not np.isfinite(a0) or a0 <= 0):
        raise ValueError('consistent finite Cartesian fields and physical units required')
    if length == 0:
        return np.zeros_like(p)
    x = np.sum(p*p, axis=0)/a0**2
    h = length**2*np.sum(H*H, axis=(0, 1))/a0**2
    u = x+h
    _, ph, k1, k2, fraction = spec.partials(x, h)
    complement = np.divide(h, u, out=np.zeros_like(u), where=u > 0)
    # P_x-1 = E'(u) - (h/u)*(u K'(u)).
    delta_px = excess_derivative_change(spec, x, h)-complement*k1
    dx = 2*np.einsum('ij...,j...->i...', H, p)/a0**2
    dh = length**2*dH2/a0**2
    dph = np.divide((k1+fraction*k2)*dx+fraction*k2*dh, u,
                     out=np.zeros_like(p), where=u > 0)
    value = delta_px*p-length**2*(np.einsum('ij...,j...->i...', H, dph)+ph*dlap)
    if not np.all(np.isfinite(value)):
        raise FloatingPointError('nonfinite length flux difference')
    return value
