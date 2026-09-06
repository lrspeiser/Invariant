"""Finite native-filter covariance and separable coarse Gaussian simulations."""
from __future__ import annotations
import numpy as np
from mond_atlas_preprocessing import finite_convolution_axis
from mond_atlas_native_spectral import continuum_operator,spectral_covariance


def gaussian_kernel(sigma,truncate=4.):
    if not np.isfinite(sigma) or sigma<0:raise ValueError('invalid Gaussian width')
    if sigma==0:return np.array([1.])
    radius=int(truncate*sigma+.5);x=np.arange(-radius,radius+1,dtype=float)
    kernel=np.exp(-.5*x*x/sigma**2);return kernel/kernel.sum()


def native_to_coarse_operator(length,factor,extra_sigma,native_sigma=0.):
    if not isinstance(length,int) or not isinstance(factor,int) or factor<1 or length%factor:raise ValueError('non-tiling linear operator')
    block=np.zeros((length//factor,length))
    for i in range(length//factor):block[i,i*factor:(i+1)*factor]=1/factor
    operator=finite_convolution_axis(block,gaussian_kernel(extra_sigma),1)
    if native_sigma>0:operator=finite_convolution_axis(operator,gaussian_kernel(native_sigma),1)
    return operator


def normalized_covariance(operator):
    covariance=operator@operator.T
    scale=float(covariance[len(covariance)//2,len(covariance)//2])
    if scale<=0:raise ValueError('invalid covariance scale')
    covariance=(covariance+covariance.T)/(2*scale)
    return covariance,np.linalg.cholesky(covariance)


def selected_channel_covariance(provenance,hanning):
    if not provenance['direct_channel_mapping']:raise ValueError('unresolved spectral history')
    n=provenance['parent_channel_count'];operator=continuum_operator(n,provenance['continuum_fit_parent_indices_zero_based'],provenance['parent_channel_indices_zero_based'],provenance['polynomial_order'])
    indices=provenance['retained_continuum_fit_stored_indices']
    if not indices:raise ValueError('no candidate channels')
    operator=operator[indices];covariance=operator@spectral_covariance(n,hanning)@operator.T
    std=np.sqrt(np.diag(covariance));covariance=covariance/std[:,None]/std[None,:]
    covariance=(covariance+covariance.T)/2
    return covariance,np.linalg.cholesky(covariance)


def draw_separable(rng,batch,channel_factor,y_factor,x_factor):
    nc,ny,nx=len(channel_factor),len(y_factor),len(x_factor)
    white=rng.standard_normal((batch,nc,ny,nx))
    spatial=(y_factor@white)@x_factor.T
    return np.einsum('ij,bjkl->bikl',channel_factor,spatial,optimize=True)


def center_outer_statistic(draws,inner,outer):
    def mad(values):
        center=np.median(values,axis=-1,keepdims=True)
        return np.median(np.abs(values-center),axis=-1)
    a=mad(draws[...,inner]);b=mad(draws[...,outer])
    if np.any(b<=0):raise ValueError('zero simulated outer MAD')
    return np.median(a/b,axis=1)
