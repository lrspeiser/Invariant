"""Supplied point-emissivity quadrature to fixed native aperture means."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv,json,warnings
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss
from astropy.io import fits
from astropy.wcs import WCS
from mond_atlas_native_selection import beam_from_history,beam_covariance,spectral_matrix
from mond_atlas_native_spectral import continuum_operator
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'work/gravity-first-principles/mond-atlas-native-emission-001'
BRANCHES=('boxcar_independent','boxcar_hanning_full','boxcar_hanning_decimated')

def normal_interval(lower,upper):
    """Stable normalized Gaussian CDF interval; no line-center sampling."""
    lower,upper=np.broadcast_arrays(lower,upper)
    return np.where(lower>=0,ndtr(-lower)-ndtr(-upper),ndtr(upper)-ndtr(lower))

def load_instrument():
    history=json.loads((ROOT/'work/gravity-first-principles/mond-atlas-native-spectral-001/NGC2976.json').read_text(encoding='utf-8'));p=history['provenance'];h=fits.getheader(ROOT/history['source_path']);beam=beam_from_history(h)
    assert p['parent_channel_count']==63 and p['parent_channel_indices_zero_based']==list(range(11,53))
    assert h['CTYPE3']=='VELO-HEL' and h['VELREF']==258 and h['CDELT3']<0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');wcs=WCS(h)
    assert wcs.wcs.ctype[2]=='VRAD' and str(wcs.wcs.cunit[2])=='m / s'
    velocity=(h['CRVAL3']+(np.arange(42)+1-h['CRPIX3'])*h['CDELT3'])/1000
    actual=wcs.all_pix2world(np.column_stack([np.full(42,h['CRPIX1']-1),np.full(42,h['CRPIX2']-1),np.arange(42),np.zeros(42)]),0)[:,2]/1000
    assert np.max(abs(actual-velocity))<1e-10
    path=ROOT/'work/gravity-first-principles/mond-atlas-observation-admission-001/run004/frozen-geometric-apertures.csv'
    with path.open(newline='',encoding='utf-8') as f:apertures=list(csv.DictReader(f))
    for row in apertures:
        for k in ('x_start','x_stop','y_start','y_stop'):row[k]=int(row[k])
        assert row['x_stop']-row['x_start']==12 and row['y_stop']-row['y_start']==12
    assert len(apertures)==15
    covariance=beam_covariance(beam['major_arcsec'],beam['minor_arcsec'],beam['pa_deg'],h['CDELT2']*3600,h['CDELT1']*3600)
    return dict(apertures=apertures,beam=beam,beam_covariance_yx_pixel2=covariance,beam_area_pixels=2*np.pi*np.sqrt(np.linalg.det(covariance)),parent_channels=63,stored_indices=np.arange(11,53),velocity_km_s=velocity,velocity_increment_km_s=h['CDELT3']/1000,first_parent_velocity_km_s=velocity[0]-11*h['CDELT3']/1000,continuum=continuum_operator(63,p['continuum_fit_parent_indices_zero_based'],p['parent_channel_indices_zero_based'],p['polynomial_order']),header_original_ctype=h['CTYPE3'],header_original_velref=h['VELREF'],interpreted_ctype=str(wcs.wcs.ctype[2]),interpreted_specsys=wcs.wcs.specsys,wcs_warnings=[str(v.message) for v in caught])

def _beam_geometry(instrument,truncate_sigma):
    C=np.asarray(instrument['beam_covariance_yx_pixel2']);sy=np.sqrt(C[0,0]);sx=np.sqrt(C[1,1]);slope=C[0,1]/C[1,1];conditional=np.sqrt(C[0,0]-C[0,1]**2/C[1,1]);halo=float(truncate_sigma*np.sqrt(np.linalg.eigvalsh(C).max()))
    integrand=lambda x:np.exp(-.5*(x/sx)**2)/(np.sqrt(2*np.pi)*sx)*normal_interval((-halo-slope*x)/conditional,(halo-slope*x)/conditional)
    norm=quad(integrand,-halo,halo,epsabs=1e-13,epsrel=1e-13)[0]
    return sx,slope,conditional,halo,norm

def aperture_weights(xy_pixel,instrument,quadrature_order=24,truncate_sigma=6.,apertures=None):
    """Return aperture×emitter mean Jy/beam per emitter Jy density."""
    xy=np.asarray(xy_pixel,float)
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():raise ValueError('finite N by2 native x,y coordinates required')
    if quadrature_order<8 or truncate_sigma<5:raise ValueError('insufficient beam integration')
    apertures=instrument['apertures'] if apertures is None else apertures;sx,slope,conditional,halo,norm=_beam_geometry(instrument,truncate_sigma);nodes,weights=leggauss(quadrature_order);result=np.zeros((len(apertures),len(xy)))
    for j,ap in enumerate(apertures):
        # Native integer pixel centers imply these continuous aperture edges.
        xlo=np.maximum(ap['x_start']-.5-xy[:,0],-halo);xhi=np.minimum(ap['x_stop']-.5-xy[:,0],halo);ylo=np.maximum(ap['y_start']-.5-xy[:,1],-halo);yhi=np.minimum(ap['y_stop']-.5-xy[:,1],halo);valid=(xhi>xlo)&(yhi>ylo);k=np.flatnonzero(valid)
        for start in range(0,len(k),2048):
            ids=k[start:start+2048];half=(xhi[ids]-xlo[ids])/2;x=(xhi[ids]+xlo[ids])[:,None]/2+half[:,None]*nodes
            conditional_probability=normal_interval((ylo[ids,None]-slope*x)/conditional,(yhi[ids,None]-slope*x)/conditional);density=np.exp(-.5*(x/sx)**2)/(np.sqrt(2*np.pi)*sx);probability=half*np.sum(density*conditional_probability*weights,axis=1)/norm
            area=(ap['x_stop']-ap['x_start'])*(ap['y_stop']-ap['y_start']);result[j,ids]=probability*instrument['beam_area_pixels']/area
    return result

def spectral_density(v_los_km_s,line_sigma_km_s,instrument,branch,systemic_km_s=0.):
    velocity=np.asarray(v_los_km_s,float);sigma=np.broadcast_to(np.asarray(line_sigma_km_s,float),velocity.shape)
    if velocity.ndim!=1 or not np.isfinite(velocity).all() or not np.isfinite(sigma).all() or np.any(sigma<=0):raise ValueError('finite one-dimensional velocity and positive intrinsic sigma required')
    H,grid,width=spectral_matrix(instrument['parent_channels'],branch);centers=instrument['first_parent_velocity_km_s']+grid*instrument['velocity_increment_km_s'];dv=abs(instrument['velocity_increment_km_s'])*width;mu=velocity+systemic_km_s
    bins=normal_interval((centers[:,None]-dv/2-mu)/sigma,(centers[:,None]+dv/2-mu)/sigma)/dv
    parent=H@bins;stored=parent[instrument['stored_indices']];post=instrument['continuum']@parent
    return dict(parent=parent,positive_stored=stored,post_continuum=post,pregrid_integrated_fraction=np.sum(bins*dv,axis=0))

def render_spectra(xy_pixel,v_los_km_s,flux_jy_km_s,line_sigma_km_s,instrument=None,branch='boxcar_independent',systemic_km_s=0.,emission_multiplier=1.,weights=None,quadrature_order=24,truncate_sigma=6.):
    """Predict mean mJy/native beam: arrays are aperture×native stored channel."""
    instrument=load_instrument() if instrument is None else instrument;flux=np.asarray(flux_jy_km_s,float)
    if flux.ndim!=1 or len(flux)!=len(xy_pixel) or len(flux)!=len(v_los_km_s) or np.any(flux<0) or not np.isfinite(flux).all() or not np.isfinite(emission_multiplier) or emission_multiplier<=0:raise ValueError('nonnegative finite emitter flux Jy km/s required')
    W=aperture_weights(xy_pixel,instrument,quadrature_order,truncate_sigma) if weights is None else np.asarray(weights,float)
    if W.shape!=(len(instrument['apertures']),len(flux)) or not np.isfinite(W).all() or np.any(W<0):raise ValueError('invalid aperture weights')
    spectral=spectral_density(v_los_km_s,line_sigma_km_s,instrument,branch,systemic_km_s);weighted=W*(flux*emission_multiplier)[None,:]
    return dict(spectra_mjy_beam=1000*weighted@spectral['post_continuum'].T,positive_stored_mjy_beam=1000*weighted@spectral['positive_stored'].T,parent_mjy_beam=1000*weighted@spectral['parent'].T,velocity_km_s=instrument['velocity_km_s'].copy(),input_flux_jy_km_s=float(flux.sum()*emission_multiplier),source_weighted_pregrid_flux_fraction=float(np.dot(flux,spectral['pregrid_integrated_fraction'])/flux.sum()) if flux.sum()>0 else 1.,branch=branch)
