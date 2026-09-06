"""Finite Gaussian smoothing matching the old native-to-coarse preprocessing."""
from __future__ import annotations
import numpy as np


def finite_convolution_axis(array,kernel,axis):
    array=np.asarray(array,float);kernel=np.asarray(kernel,float)
    if array.ndim!=2 or kernel.ndim!=1 or len(kernel)%2!=1 or not np.isfinite(array).all() or not np.isfinite(kernel).all():
        raise ValueError('finite image and odd one-dimensional kernel required')
    moved=np.moveaxis(array,axis,-1);n=moved.shape[-1];size=1<<(n+len(kernel)-2).bit_length()
    full=np.fft.irfft(np.fft.rfft(moved,n=size,axis=-1)*np.fft.rfft(kernel,n=size),n=size,axis=-1)
    radius=len(kernel)//2
    return np.moveaxis(full[...,radius:radius+n],-1,axis)


def gaussian_plane_float32(array,sigma,truncate=4.):
    if not np.isfinite(sigma) or sigma<0 or truncate<=0:raise ValueError('invalid Gaussian width')
    data=np.asarray(array,np.float32)
    if sigma==0:return data.copy()
    radius=int(truncate*sigma+.5);x=np.arange(-radius,radius+1,dtype=float)
    kernel=np.exp(-.5*x*x/(sigma*sigma));kernel/=kernel.sum()
    # scipy gaussian_filter on (channel,y,x) with sigma_channel=0 applies y then x,
    # storing each intermediate in the input/output dtype.
    for axis in (0,1):data=finite_convolution_axis(data,kernel,axis).astype(np.float32)
    return data


def block_mean(array,factor):
    array=np.asarray(array)
    if array.ndim!=2 or not isinstance(factor,int) or factor<1 or any(n%factor for n in array.shape):raise ValueError('non-tiling block factor')
    ny,nx=array.shape
    return array.reshape(ny//factor,factor,nx//factor,factor).mean(axis=(1,3))


def original_offset_mask(east,north,config):
    yy,xx=np.indices(east.shape);lo,hi=config['offset_background_annulus_arcsec'];r=np.hypot(east,north)
    side=config['offset_block_side_coarse_pixels']
    return (r>lo)&(r<hi)&(xx>6)&(yy>6)&(xx<east.shape[1]-7)&(yy<east.shape[0]-7)&(((xx//side+yy//side)%2)==config['offset_calibration_parity'])
