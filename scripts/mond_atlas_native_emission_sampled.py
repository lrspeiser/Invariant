"""Sampled CLEAN pixel means; exact grouped emission linearity."""
from mond_atlas_native_emission import load_instrument,spectral_density,BRANCHES,_beam_geometry,ROOT,PACKAGE
import numpy as np

def aperture_weights(xy_pixel,instrument,truncate_sigma=6.,apertures=None):
    xy=np.asarray(xy_pixel,float)
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all() or truncate_sigma<5:raise ValueError('finite N by2 native positions and halo>=5 sigma required')
    apertures=instrument['apertures'] if apertures is None else apertures;_,_,_,halo,norm=_beam_geometry(instrument,truncate_sigma);inv=np.linalg.inv(instrument['beam_covariance_yx_pixel2']);result=np.zeros((len(apertures),len(xy)))
    for a,ap in enumerate(apertures):
        yy,xx=np.mgrid[ap['y_start']:ap['y_stop'],ap['x_start']:ap['x_stop']];xx=xx.ravel();yy=yy.ravel();eligible=np.flatnonzero((xy[:,0]>=xx.min()-halo)&(xy[:,0]<=xx.max()+halo)&(xy[:,1]>=yy.min()-halo)&(xy[:,1]<=yy.max()+halo))
        for start in range(0,len(eligible),1024):
            ids=eligible[start:start+1024];dx=xx[None,:]-xy[ids,0,None];dy=yy[None,:]-xy[ids,1,None];square=inv[0,0]*dy*dy+2*inv[0,1]*dx*dy+inv[1,1]*dx*dx;kernel=np.exp(-.5*square)*((abs(dx)<=halo)&(abs(dy)<=halo))/norm;result[a,ids]=kernel.mean(axis=1)
    return result

def grouped_aperture_flux(xy_pixel,flux_jy_km_s,group_ids,instrument,group_count=None,truncate_sigma=6.):
    """Sum projected vertical nodes exactly; returns aperture×group Jy km/s weights."""
    flux=np.asarray(flux_jy_km_s,float);groups=np.asarray(group_ids)
    if flux.shape!=(len(xy_pixel),) or groups.shape!=flux.shape or groups.dtype.kind not in 'iu' or not np.isfinite(flux).all() or np.any(flux<0) or np.any(groups<0):raise ValueError('finite nonnegative flux and nonnegative integer groups required')
    count=int(groups.max())+1 if group_count is None and len(groups) else int(group_count or 0)
    if len(groups) and groups.max()>=count:raise ValueError('group index out of bounds')
    result=np.zeros((len(instrument['apertures']),count))
    for a,ap in enumerate(instrument['apertures']):
        weights=aperture_weights(xy_pixel,instrument,truncate_sigma,apertures=[ap])[0];result[a]=np.bincount(groups,weights=weights*flux,minlength=count)
    return result

def render_grouped_spectra(grouped_flux_jy_km_s,v_los_km_s,line_sigma_km_s,instrument,branch='boxcar_independent',systemic_km_s=0.,emission_multiplier=1.):
    grouped=np.asarray(grouped_flux_jy_km_s,float);v=np.asarray(v_los_km_s,float);sigma=np.broadcast_to(np.asarray(line_sigma_km_s,float),v.shape)
    if grouped.shape!=(len(instrument['apertures']),len(v)) or np.any(grouped<0) or not np.isfinite(grouped).all() or not np.isfinite(emission_multiplier) or emission_multiplier<=0 or not np.isfinite(v).all() or not np.isfinite(sigma).all() or np.any(sigma<=0):raise ValueError('invalid grouped emissivity or line parameters')
    active=np.any(grouped>0,axis=0);g=grouped[:,active]*emission_multiplier;spec=spectral_density(v[active],sigma[active],instrument,branch,systemic_km_s)
    return dict(spectra_mjy_beam=1000*g@spec['post_continuum'].T,positive_stored_mjy_beam=1000*g@spec['positive_stored'].T,parent_mjy_beam=1000*g@spec['parent'].T,velocity_km_s=instrument['velocity_km_s'].copy(),branch=branch,active_groups=int(active.sum()),inactive_zero_weight_groups=int((~active).sum()))

def render_spectra(xy_pixel,v_los_km_s,flux_jy_km_s,line_sigma_km_s,instrument=None,branch='boxcar_independent',systemic_km_s=0.,emission_multiplier=1.,weights=None,truncate_sigma=6.):
    inst=load_instrument() if instrument is None else instrument;flux=np.asarray(flux_jy_km_s,float)
    if flux.shape!=(len(xy_pixel),) or len(v_los_km_s)!=len(flux) or np.any(flux<0) or not np.isfinite(flux).all():raise ValueError('nonnegative finite integrated emitter flux required')
    W=aperture_weights(xy_pixel,inst,truncate_sigma) if weights is None else np.asarray(weights,float)
    result=render_grouped_spectra(W*flux[None,:],v_los_km_s,line_sigma_km_s,inst,branch,systemic_km_s,emission_multiplier);result['input_flux_jy_km_s']=float(flux.sum()*emission_multiplier);return result
