"""Controlled smooth-cube generation and a fixed consecutive-channel mask."""
from __future__ import annotations
import numpy as np
from mond_atlas_smoothing_null import gaussian_kernel
from mond_atlas_preprocessing import finite_convolution_axis


def consecutive_mask(cube,threshold=2.,count=3):
    cube=np.asarray(cube)
    if cube.ndim!=3 or not np.isfinite(cube).all() or not isinstance(count,int) or not 1<=count<=len(cube):raise ValueError('invalid detection cube')
    above=cube>threshold
    starts=np.ones((len(cube)-count+1,*cube.shape[1:]),bool)
    for offset in range(count):starts &= above[offset:offset+len(starts)]
    mask=np.zeros_like(above)
    for offset in range(count):mask[offset:offset+len(starts)] |= starts
    return mask


def spatial_filter_cube(cube,kernel):
    cube=np.asarray(cube,float)
    if cube.ndim!=3:raise ValueError('channel by spatial array required')
    nc,ny,nx=cube.shape
    # Finite y then x convolution, preserving channel identity.
    transformed=finite_convolution_axis(cube.transpose(0,2,1).reshape(nc*nx,ny),kernel,1).reshape(nc,nx,ny).transpose(0,2,1)
    return finite_convolution_axis(transformed.reshape(nc*ny,nx),kernel,1).reshape(nc,ny,nx)


def response_kernel(instrument):
    pixel=instrument['pixel_arcsec'];native=instrument['native_circular_fwhm_arcsec'];target=instrument['detection_circular_fwhm_arcsec']
    if target<native:raise ValueError('cannot smooth to a narrower beam')
    k0=gaussian_kernel(native/(2.354820045*pixel),instrument['spatial_kernel_truncate_sigma'])
    k1=gaussian_kernel(np.sqrt(target**2-native**2)/(2.354820045*pixel),instrument['spatial_kernel_truncate_sigma'])
    return np.convolve(k0,k1)


def noise_cube(rng,instrument,mode):
    kernel=response_kernel(instrument);radius=len(kernel)//2;nc=instrument['channels'];ny,nx=instrument['spatial_shape']
    if mode not in instrument['spectral_branches']:raise ValueError('undeclared spectral branch')
    parent_nc=2*nc+1 if mode=='decimated_hanning_channels' else nc+2 if mode=='hanning_channels' else nc
    white=rng.normal(size=(parent_nc,ny+2*radius,nx+2*radius))
    smoothed=spatial_filter_cube(white,kernel)[:,radius:radius+ny,radius:radius+nx]
    # Independent filters along x and y give standard deviation sum(k^2).
    smoothed/=np.sum(kernel*kernel)
    if mode!='independent_channels':
        spectral=np.asarray(instrument['hanning_kernel'],float)
        length=parent_nc-2
        smoothed=sum(spectral[i]*smoothed[i:i+length] for i in range(3))/np.sqrt(np.sum(spectral*spectral))
        if mode=='decimated_hanning_channels':smoothed=smoothed[::2]
    return smoothed


def source_template(instrument,spatial_fwhm,spectral_fwhm,mode):
    nc=instrument['channels'];ny,nx=instrument['spatial_shape'];pixel=instrument['pixel_arcsec']
    yy,xx=np.indices((ny,nx));center=(nc//2,ny//2,nx//2)
    spatial=np.exp(-4*np.log(2)*pixel**2*((xx-center[2])**2+(yy-center[1])**2)/spatial_fwhm**2)
    if mode=='decimated_hanning_channels':
        parent=np.exp(-4*np.log(2)*((np.arange(2*nc+1)-(2*center[0]+1))/2)**2/spectral_fwhm**2)
        spectral=np.convolve(parent,instrument['hanning_kernel'],mode='valid')[::2]
    else:
        spectral=np.exp(-4*np.log(2)*(np.arange(nc)-center[0])**2/spectral_fwhm**2)
        if mode=='hanning_channels':spectral=np.convolve(spectral,instrument['hanning_kernel'],mode='same')
    cube=spatial_filter_cube(spectral[:,None,None]*spatial[None,:,:],response_kernel(instrument))
    # Numerical Fourier roundoff must not create negative physical injected flux.
    if cube.min() < -1e-12:raise ArithmeticError('negative source after positive response')
    cube=np.maximum(cube,0);cube/=cube.max()
    return cube,center


def recovery_metrics(cube,template,amplitude,center,detection):
    mask=consecutive_mask(cube,detection['threshold_sigma'],detection['consecutive_channels'])
    source=template*amplitude;flux=float(source.sum())
    return dict(peak_selected=bool(mask[center]),selected_voxel_fraction=float(mask.mean()),
        known_source_flux=float(flux),true_flux_fraction_retained=float(source[mask].sum()/flux),
        measured_selected_flux_over_true=float(cube[mask].sum()/flux))
