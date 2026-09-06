"""Source-only conditional baryon maps, with missing coverage retained.

No rotation curves or cube velocities are imported. Pixel-area approximation is
evaluated at centers with the TAN/SIN spherical Jacobian. Output units are
face-on Msun/pc2 (stars stored as Lsun/pc2; CO as face-on K km/s).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT,read_json,digest,write_json,write_csv
from mond_atlas_image_io import read_primary_image


def sky_vectors(ra_deg,dec_deg):
    ra,dec=np.deg2rad([ra_deg,dec_deg])
    center=np.array([np.cos(dec)*np.cos(ra),np.cos(dec)*np.sin(ra),np.sin(dec)])
    east=np.array([-np.sin(ra),np.cos(ra),0.]);north=np.cross(center,east)
    return center,east,north


def linear_cd(header):
    if any(k.startswith(('PV1_','PV2_','CPDIS','D2IM','PC1_','PC2_')) for k in header):
        raise ValueError('distorted or PC-matrix WCS not implemented')
    if any(float(header.get(k,0))!=0 for k in ('CROTA1','CROTA2')):
        raise ValueError('rotated CDELT convention not implemented')
    if float(header.get('LONPOLE',180))!=180 or float(header.get('LATPOLE',90))!=90:
        raise ValueError('nondefault pole')
    if any(header.get('CUNIT'+str(i),'deg').lower() not in ('deg','degree','degrees') for i in (1,2)):
        raise ValueError('degree coordinates required')
    keys=('CD1_1','CD1_2','CD2_1','CD2_2')
    if all(k in header for k in keys):cd=np.array([header[k] for k in keys]).reshape(2,2)
    elif not any(k in header for k in keys):cd=np.diag([header['CDELT1'],header['CDELT2']])
    else:raise ValueError('partial CD matrix')
    if not np.isfinite(cd).all() or np.linalg.det(cd)==0:raise ValueError('singular WCS')
    return cd


def pixel_geometry(header,shape,geometry):
    cd=linear_cd(header);projection=header['CTYPE1'][-3:]
    if header['CTYPE1']!='RA---'+projection or header['CTYPE2']!='DEC--'+projection or projection not in ('TAN','SIN'):
        raise ValueError('only explicit standard TAN and SIN projections')
    yy,xx=np.indices(shape);p=np.stack((xx+1-header['CRPIX1'],yy+1-header['CRPIX2']),axis=-1)
    plane=np.deg2rad(p@cd.T);l,m=plane[...,0],plane[...,1]
    c,e,n=sky_vectors(header['CRVAL1'],header['CRVAL2']);radius2=l*l+m*m
    if projection=='TAN':
        denom=np.sqrt(1+radius2);v=(c+l[...,None]*e+m[...,None]*n)/denom[...,None]
        jac=(1+radius2)**(-1.5)
    else:
        if np.any(radius2>=1):raise ValueError('SIN outside visible hemisphere')
        root=np.sqrt(1-radius2);v=root[...,None]*c+l[...,None]*e+m[...,None]*n;jac=1/root
    gc,ge,gn=sky_vectors(geometry['ra_deg'],geometry['dec_deg']);depth=v@gc
    if np.any(depth<=0):raise ValueError('galaxy projection outside tangent hemisphere')
    east=(v@ge)/depth*geometry['distance_mpc']*1000
    north=(v@gn)/depth*geometry['distance_mpc']*1000
    pa=np.deg2rad(geometry['pa_deg']);cosi=np.cos(np.deg2rad(geometry['inclination_deg']))
    major=east*np.sin(pa)+north*np.cos(pa)
    minor=(east*np.cos(pa)-north*np.sin(pa))/cosi
    # Solid angle -> projected area at fixed distance; no depth inference.
    projected_area=abs(np.linalg.det(cd))*(np.pi/180)**2*(geometry['distance_mpc']*1000)**2*jac
    return major,minor,projected_area,cosi


def rebin_source(image,header,good,geometry,grid,conversion=1.,error=None):
    x,y,area,cosi=pixel_geometry(header,image.shape,geometry)
    h=grid['spacing_kpc'];half=grid['half_width_kpc'];axis=np.arange(-half,half+h/2,h)
    edges=np.r_[axis-h/2,axis[-1]+h/2];shape=(len(axis),len(axis))
    def hist(weights):return np.histogram2d(x[good],y[good],bins=(edges,edges),weights=weights[good])[0]
    good=np.asarray(good,bool)&np.isfinite(image)
    measured_area=hist(area/cosi);signed_mass=hist(image*conversion*area*1e6)
    observed=signed_mass/(h*h*1e6)
    mean=np.divide(signed_mass,measured_area*1e6,out=np.full(shape,np.nan),where=measured_area>0)
    coverage=measured_area/(h*h)
    xx,yy=np.meshgrid(axis,axis,indexing='ij');rr=np.hypot(xx,yy)
    rings=np.floor(rr/grid['annulus_width_kpc']).astype(int)
    nring=int(np.max(rings))+1
    ring_area=np.bincount(rings.ravel(),weights=measured_area.ravel(),minlength=nring)
    ring_mass=np.bincount(rings.ravel(),weights=signed_mass.ravel(),minlength=nring)
    ring_all_area=np.bincount(rings.ravel(),minlength=nring)*h*h
    ring_cov=ring_area/np.maximum(ring_all_area,1e-30)
    ring_mean=np.divide(ring_mass,ring_area*1e6,out=np.full(nring,np.nan),where=ring_area>0)
    qualified=np.isfinite(ring_mean)&(ring_cov>=grid['minimum_annulus_coverage'])
    indices=np.arange(nring)
    if not qualified.any():raise ValueError('no sufficiently covered source annulus')
    profile=np.interp(indices,indices[qualified],np.maximum(ring_mean[qualified],0),left=0,right=0)
    trusted=(coverage>=grid['minimum_cell_coverage'])&np.isfinite(mean)
    filled=np.where(trusted,np.maximum(mean,0),profile[rings])
    measured=np.maximum(observed,0)
    taper=np.clip((grid['cutoff_kpc']-rr)/(grid['cutoff_kpc']-grid['taper_start_kpc']),0,1)
    filled*=taper;measured*=taper
    # The recorded linear taper is intentional; no force data choose it.
    error_map=None
    if error is not None:
        weighted_error=hist(np.where(np.isfinite(error),error,0)*conversion*area*1e6)
        error_map=np.divide(weighted_error,measured_area*1e6,out=np.full(shape,np.nan),where=measured_area>0)
    report=dict(observed_area_fraction_inside_cutoff=float(np.sum(np.minimum(coverage,1)[rr<grid['cutoff_kpc']])/np.sum(rr<grid['cutoff_kpc'])),
        cells_with_half_coverage=int(np.sum(trusted&(rr<grid['cutoff_kpc']))),cells_inside_cutoff=int(np.sum(rr<grid['cutoff_kpc'])),
        signed_measured_integral=float(np.sum(observed*taper)*h*h*1e6),
        negative_projection_added_integral=float(np.sum((measured-observed*taper))*h*h*1e6),
        conditional_zero_integral=float(np.sum(measured)*h*h*1e6),
        conditional_annular_integral=float(np.sum(filled)*h*h*1e6),
        supported_annuli=int(np.sum(qualified)),
        extrapolated_outside_last_supported_annulus=False,
        missing_flux_is_measured_zero=False,
        maximum_area_coverage=float(np.max(coverage)))
    rows=[dict(radius_kpc=(i+.5)*grid['annulus_width_kpc'],coverage=float(ring_cov[i]),
         signed_mean=float(ring_mean[i]) if np.isfinite(ring_mean[i]) else None,
         accepted=bool(qualified[i]),conditional_fill=float(profile[i])) for i in range(nring)]
    return dict(axis=axis,observed=observed,mean=mean,coverage=coverage,annular=filled,zero=measured,error=error_map),report,rows


def build(protocol_path,output,private):
    config=read_json(protocol_path)
    if output.exists() or private.exists():raise FileExistsError('immutable output already exists')
    output.mkdir(parents=True);private.mkdir(parents=True)
    assets=read_json(ROOT/'work/gravity-first-principles/stellar-co-acquisition-001/receipt.json')['files']
    assets={a['role']:a for a in assets if a['name']==config['object_id']}
    hi=next(a for a in read_json(ROOT/'work/gravity-first-principles/things-observable-acquisition-003/receipt.json')['files'] if a['name']==config['object_id'] and a['resolution']=='NA' and a['moment']==0)
    used=[assets[k] for k in ('STELLAR_MASS_MAP','STELLAR_ICA_MASK','CO21_MOM0','CO21_EMOM0')]+[hi]
    for a in used:
        if digest(ROOT/a['file'])!=a['sha256']:raise ValueError('source hash mismatch '+a['file'])
    transfer=read_json(ROOT/config['p5_transfer_receipt'])
    if not transfer['pass_gate'] or transfer['p1_pixel_translation']!=[-3.,-1.]:raise ValueError('P5 transfer contract')
    star,sh=read_primary_image(ROOT/assets['STELLAR_MASS_MAP']['file']);mask,mh=read_primary_image(ROOT/assets['STELLAR_ICA_MASK']['file'])
    if star.shape!=mask.shape or any(sh.get(k)!=mh.get(k) for k in ('CRVAL1','CRVAL2','CRPIX1','CRPIX2','CD1_1','CD1_2','CD2_1','CD2_2')):raise ValueError('ICA mask coordinates differ')
    sh=sh.copy();sh['CRPIX1']+=config['p5_crpix_add'][0];sh['CRPIX2']+=config['p5_crpix_add'][1]
    atomic,hh=read_primary_image(ROOT/hi['file'])
    import re
    beam=next(re.search(r'BMAJ=\s*([\d.E+-]+)\s+BMIN=\s*([\d.E+-]+)',t) for t in hi['beam_and_blanking_history'] if 'CLEAN BMAJ=' in t)
    major,minor=[float(t)*3600 for t in beam.groups()]
    nu=hh.get('RESTFREQ',1420405750)/1e9
    factor=.001*1222000/(nu*nu*major*minor)*1.823e18/1.248e20*config['conversions']['helium_factor_hi']
    co,ch=read_primary_image(ROOT/assets['CO21_MOM0']['file']);err,eh=read_primary_image(ROOT/assets['CO21_EMOM0']['file'])
    if co.shape!=err.shape or any(ch.get(k)!=eh.get(k) for k in ('CRPIX1','CRPIX2','CRVAL1','CRVAL2','CDELT1','CDELT2')):raise ValueError('CO error coordinates differ')
    if sh.get('BUNIT').lower()!='mjy/sr' or hh.get('BUNIT')!='JY/B*M/S' or ch.get('BUNIT')!='K KM/S':raise ValueError('source units changed')
    inputs=[('stellar_luminosity',star,sh,(mask==0)&np.isfinite(star),config['conversions']['stellar_lsun_pc2_per_mjy_sr'],None),
            ('atomic_helium',atomic,hh,np.isfinite(atomic)&(atomic!=0),factor,None),
            ('co21',co,ch,np.isfinite(co)&np.isfinite(err)&(err>0),1.,err)]
    packed={};reports={};allrows=[]
    for name,img,h,good,conv,error in inputs:
        arrays,report,rows=rebin_source(img,h,good,config['geometry'],config['source_grid'],conv,error)
        reports[name]=report
        for k,v in arrays.items():
            if v is not None:packed[name+'_'+k]=v
        allrows.extend(dict(component=name,**r) for r in rows)
    target=private/'source-maps.npz';np.savez_compressed(target,**packed)
    write_csv(output/'source-annuli.csv',allrows)
    write_json(output/'source-audit.json',dict(status='CONDITIONAL_SOURCE_MAPS_NOT_OBSERVED_3D',object_id=config['object_id'],
        components=reports,protocol=config,protocol_sha256=digest(protocol_path),
        source_assets=used,source_packet=str(target.relative_to(ROOT)),source_packet_sha256=digest(target),
        bindings=[dict(path=str(p.relative_to(ROOT)),sha256=digest(p)) for p in (Path(__file__),ROOT/'scripts/mond_atlas_image_io.py',ROOT/'scripts/mond_atlas_common.py',ROOT/config['geometry_source'],ROOT/config['p5_transfer_receipt'])],
        kinematic_inputs_consumed=[],source_uncertainty_likelihood_complete=False))
    print(reports,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/mond_atlas_ngc2903_field_v1.json')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--private',type=Path,required=True)
    args=parser.parse_args();build(args.protocol.resolve(),args.output.resolve(),args.private.resolve())
