"""Independent Astropy WCS + finite-difference projected-area photometry replay."""
from pathlib import Path
import json,hashlib,math
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.integrate import quad
from scipy.special import ndtr

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
P=HERE.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
bindings=json.loads((P/'run001/bindings.json').read_text())
assert all(sha(ROOT/k)==v for k,v in bindings.items())
summary=json.loads((P/'run001/summary.json').read_text())['rows']
results=[]
for galaxy,radii in [('NGC2976',[1,3,6]),('NGC3198',[3,10,28])]:
    cfg=json.loads((ROOT/f'configs/mond_atlas_{galaxy.lower()}_source_v1.json').read_text(encoding='utf8'))
    data,h=fits.getdata(ROOT/cfg['assets']['CO21_MOM0']['file'],header=True)
    err,eh=fits.getdata(ROOT/cfg['assets']['CO21_EMOM0']['file'],header=True)
    data=data.squeeze();err=err.squeeze();wcs=WCS(h).celestial
    g=cfg['geometry'];distance=g['distance_mpc']*1e6
    cosi=math.cos(math.radians(g['inclination_deg'])) if 'inclination_deg' in g else math.sqrt(((1-g['ellipticity'])**2-g['intrinsic_axis_ratio']**2)/(1-g['intrinsic_axis_ratio']**2))
    ra0,dec0=np.deg2rad([g['ra_deg'],g['dec_deg']]);pa=math.radians(g['pa_deg'])
    def projected(x,y):
        ra,dec=np.deg2rad(wcs.all_pix2world(x,y,0))
        dra=ra-ra0
        denom=np.sin(dec0)*np.sin(dec)+np.cos(dec0)*np.cos(dec)*np.cos(dra)
        return distance*np.cos(dec)*np.sin(dra)/denom,distance*(np.cos(dec0)*np.sin(dec)-np.sin(dec0)*np.cos(dec)*np.cos(dra))/denom
    yy,xx=np.indices(data.shape,dtype=float);x=xx.ravel();y=yy.ravel()
    good=np.isfinite(data.ravel())&np.isfinite(err.ravel())&(err.ravel()>0)
    value=np.where(good,data.ravel(),0.);sigma=np.where(good,err.ravel(),0.)
    for sub in [1,4]:
        weights=np.zeros((3,len(x)))
        for ox in (np.arange(sub)+.5)/sub-.5:
            for oy in (np.arange(sub)+.5)/sub-.5:
                east,north=projected(x+ox,y+oy)
                major=east*np.sin(pa)+north*np.cos(pa)
                minor=(east*np.cos(pa)-north*np.sin(pa))/cosi
                radial=np.sqrt(major**2+minor**2)/1000
                # Numerical Jacobian of independent celestial WCS and gnomonic projection.
                eps=.01
                ep,np_=projected(x+ox+eps,y+oy);em,nm=projected(x+ox-eps,y+oy)
                eq,nq=projected(x+ox,y+oy+eps);er,nr=projected(x+ox,y+oy-eps)
                area=np.abs((ep-em)*(nq-nr)-(eq-er)*(np_-nm))/(4*eps**2)
                for j,radius in enumerate(radii):weights[j]+=area*(radial<=radius)/sub**2
        for radius,weight in zip(radii,weights):
            row=next(r for r in summary if r['galaxy']==galaxy and r['radius_kpc']==radius and r['subdivisions']==sub)
            ww=weight*good
            ref=dict(covered_projected_area_pc2=float(ww.sum()),signed_co_luminosity=float(ww@value),
                positive_clipped_luminosity=float(ww@np.maximum(value,0)),independent_pixel_sigma=float(np.sqrt(np.sum((ww*sigma)**2))),
                fully_correlated_sigma_bound=float(np.sum(ww*sigma)))
            relative={k:abs(ref[k]/row[k]-1) for k in ref}
            assert max(relative.values())<1e-6
            assert abs(row['clipping_increment']+row['parts']['negative']['signed_luminosity'])<1e-7*max(1,abs(row['signed_co_luminosity']))
            results.append(dict(galaxy=galaxy,radius_kpc=radius,subdivisions=sub,reference=ref,relative_errors=relative,
                saved_coverage_fraction=row['coverage_fraction'],raw_coverage_unclipped=True,
                full_finite_footprint=(int(np.count_nonzero(ww))==int(good.sum())),
                finite_intensity_invalid_error_pixels=int((np.isfinite(data.ravel())&~good).sum())))
expectation_errors=[]
for mu in [-3,-.5,0,.5,3]:
    for sigma in [.2,1,2]:
        analytic=sigma*math.exp(-.5*(mu/sigma)**2)/math.sqrt(2*math.pi)+mu*ndtr(mu/sigma)
        numerical=quad(lambda z:z*math.exp(-.5*((z-mu)/sigma)**2)/(sigma*math.sqrt(2*math.pi)),0,np.inf,epsabs=1e-10)[0]
        expectation_errors.append(abs(analytic-numerical))
assert max(expectation_errors)<1e-9
assert all(sha(ROOT/k)==v for k,v in bindings.items())
receipt=dict(status='INDEPENDENT_PHOTOMETRY_REPLAY_PASSED',method='Astropy celestial WCS, spherical trigonometry, numerical projected-area Jacobian; no source_coordinates or production weights imports',
    source_hashes_reverified=len(bindings),apertures_replayed=results,
    maximum_photometry_relative_error=max(max(r['relative_errors'].values()) for r in results),
    independent_positive_expectation_max_abs=max(expectation_errors),
    observed_velocity_arrays_accessed=False,gravity_scored=False,
    reviewer_script_sha256=sha(Path(__file__)))
with (HERE/'receipt.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(receipt['maximum_photometry_relative_error'],receipt['independent_positive_expectation_max_abs'])
