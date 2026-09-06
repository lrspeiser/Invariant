"""Finite-volume line emission and separable correlated-noise building blocks.

All 3D arrays are already in observer coordinates. This module deliberately has
no WCS or mass-to-motion inference: those must be validated in a source adapter.
An HI spectral cube is never used as a spatial-density cube by this interface.
"""
from __future__ import annotations

import numpy as np


def gaussian_cdf(x):
    """Normal CDF approximation (absolute error < 8e-8)."""
    a = np.asarray(x, float)/np.sqrt(2.)
    t = 1/(1+.3275911*np.abs(a))
    erf = 1-(((((1.061405429*t-1.453152027)*t+1.421413741)*t-.284496736)*t
               +.254829592)*t)*np.exp(-a*a)
    return .5*(1+np.sign(a)*erf)


def project_emission(emissivity_observer_zyx, los_velocity_km_s, dispersion_km_s,
                     channel_edges_km_s, path_element):
    """Integrate every physical depth element into finite spectral channels.

    Emissivity is per unit physical path length; output sums to integrated
    emission only when channels contain the complete line. No renormalizing
    truncated line wings. Self-absorption is not modeled.
    """
    emissivity = np.asarray(emissivity_observer_zyx, float)
    velocity = np.asarray(los_velocity_km_s, float)
    sigma = np.broadcast_to(np.asarray(dispersion_km_s,float), emissivity.shape)
    edges = np.asarray(channel_edges_km_s,float)
    if (emissivity.ndim != 3 or velocity.shape != emissivity.shape or
            not np.isfinite(emissivity).all() or (emissivity < 0).any() or
            not np.isfinite(velocity).all() or not np.isfinite(sigma).all() or
            (sigma<=0).any() or path_element<=0 or not np.isfinite(path_element) or
            edges.ndim!=1 or len(edges)<2 or not np.isfinite(edges).all() or (np.diff(edges)<=0).any()):
        raise ValueError("invalid observer-volume emission input")
    cube = np.zeros((len(edges)-1,)+emissivity.shape[1:])
    for z in range(emissivity.shape[0]):
        cdf = gaussian_cdf((edges[:,None,None]-velocity[z])/sigma[z])
        cube += np.diff(cdf,axis=0)*emissivity[z]*path_element
    return cube


def spatial_beam(cube, kernel):
    """Linear zero-padded convolution, with the kernel centered on its middle pixel."""
    cube,kernel=np.asarray(cube,float),np.asarray(kernel,float)
    if (cube.ndim!=3 or kernel.ndim!=2 or any(n%2==0 for n in kernel.shape) or
            not np.isfinite(kernel).all() or (kernel<0).any() or kernel.sum()<=0):
        raise ValueError("odd, finite nonnegative beam kernel required")
    kernel=kernel/kernel.sum()
    shape=tuple(cube.shape[i+1]+kernel.shape[i]-1 for i in range(2))
    result=np.fft.irfft2(np.fft.rfft2(cube,s=shape,axes=(-2,-1))*
                        np.fft.rfft2(kernel,s=shape),s=shape,axes=(-2,-1))
    y,x=(n//2 for n in kernel.shape)
    return result[:,y:y+cube.shape[1],x:x+cube.shape[2]]


def spectral_response(cube, response):
    """Apply a measured channel-mixing matrix; conserve flux if columns sum to one."""
    response=np.asarray(response,float)
    if (response.ndim!=2 or response.shape[1]!=cube.shape[0] or
            not np.isfinite(response).all()):
        raise ValueError("invalid spectral response matrix")
    return np.einsum("ij,jyx->iyx",response,cube)


def correlated_score(residual_channel_pixel, channel_covariance, spatial_covariance):
    """Exact Gaussian score for a supplied separable covariance on retained pixels.

    The caller applies one fixed spatial mask to all channels and subsets the
    spatial covariance accordingly. Data-dependent selection, a nonseparable
    covariance or missing channels at varying pixels require a different model.
    No p-value or validated astronomical goodness-of-fit is implied by this score.
    """
    residual=np.asarray(residual_channel_pixel,float)
    cc,cs=np.asarray(channel_covariance,float),np.asarray(spatial_covariance,float)
    if (residual.ndim!=2 or cc.shape!=(residual.shape[0],)*2 or cs.shape!=(residual.shape[1],)*2
            or not np.isfinite(residual).all() or not np.isfinite(cc).all() or not np.isfinite(cs).all()
            or not np.allclose(cc,cc.T,rtol=1e-12,atol=1e-14)
            or not np.allclose(cs,cs.T,rtol=1e-12,atol=1e-14)):
        raise ValueError("invalid separable covariance or residual dimensions")
    lc,ls=np.linalg.cholesky(cc),np.linalg.cholesky(cs)
    white=np.linalg.solve(ls,np.linalg.solve(lc,residual).T).T
    nchan,nsky=residual.shape
    logdet=2*nsky*np.log(np.diag(lc)).sum()+2*nchan*np.log(np.diag(ls)).sum()
    q=float(np.sum(white**2))
    return dict(quadratic_form=q,log_determinant=float(logdet),
                gaussian_log_likelihood=float(-.5*(q+logdet+residual.size*np.log(2*np.pi))))
