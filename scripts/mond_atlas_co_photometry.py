"""Actual signed molecular photometry; no source reconstruction or gravity fit."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from mond_atlas_image_io import read_primary_image
from mond_atlas_registered_source import source_coordinates,inclination
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'work/gravity-first-principles/mond-atlas-co-photometry-001'
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def controls():
    errors=[]
    for mu in [0,.5,3]:
        for sigma in [.2,1,2]:
            analytic=sigma*np.exp(-.5*(mu/sigma)**2)/np.sqrt(2*np.pi)+mu*ndtr(mu/sigma)
            numerical=quad(lambda z:max(mu+sigma*z,0)*np.exp(-z*z/2)/np.sqrt(2*np.pi),-mu/sigma,12,epsabs=1e-11)[0]
            errors.append(abs(analytic-numerical));assert errors[-1]<1e-9
    rng=np.random.default_rng(927);a=np.maximum(rng.normal(size=1000000),0)
    assert abs(a.mean()-1/np.sqrt(2*np.pi))<5*a.std()/np.sqrt(len(a))
    return dict(max_expectation_error=max(errors),zero_signal_simulation_mean=float(a.mean()),analytic=1/np.sqrt(2*np.pi))

def weights(header,shape,geometry,radii,subdivisions):
    yy,xx=np.indices(shape);pixels=np.column_stack([xx.ravel(),yy.ravel()]);w=np.zeros((len(radii),len(pixels)))
    offsets=(np.arange(subdivisions)+.5)/subdivisions-.5
    for dy in offsets:
        for dx in offsets:
            x,y,area,_=source_coordinates(header,pixels+[dx,dy],geometry)
            r=np.hypot(x,y)
            for i,radius in enumerate(radii):w[i]+=area*1e6*(r<=radius)/subdivisions**2
    return w

def run():
    out=P/'run001';out.mkdir(exist_ok=False);bindings={}
    configs=[]
    for galaxy,radii in [('NGC2976',[1,3,6]),('NGC3198',[3,10,28])]:
        path=ROOT/f'configs/mond_atlas_{galaxy.lower()}_source_v1.json';cfg=json.loads(path.read_text(encoding='utf-8'));configs.append((cfg,radii));bindings[path.relative_to(ROOT).as_posix()]=digest(path)
        for name in ['CO21_MOM0','CO21_EMOM0']:
            asset=cfg['assets'][name];p=ROOT/asset['file'];assert digest(p)==asset['sha256'];bindings[asset['file']]=asset['sha256']
    for p in [Path(__file__),P/'PREFLIGHT.md',ROOT/'scripts/mond_atlas_registered_source.py',ROOT/'scripts/mond_atlas_image_io.py']:bindings[p.relative_to(ROOT).as_posix()]=digest(p)
    save(out/'bindings.json',bindings);save(out/'controls.json',controls());rows=[]
    for cfg,radii in configs:
        data,h=read_primary_image(ROOT/cfg['assets']['CO21_MOM0']['file']);err,eh=read_primary_image(ROOT/cfg['assets']['CO21_EMOM0']['file'])
        assert data.shape==err.shape and h['BUNIT']==eh['BUNIT']=='K KM/S'
        for key in ['CTYPE1','CTYPE2','CRVAL1','CRVAL2','CRPIX1','CRPIX2','CDELT1','CDELT2','CD1_1','CD1_2','CD2_1','CD2_2']:assert h.get(key)==eh.get(key)
        v=data.ravel();sigma=err.ravel();valid=np.isfinite(v)&np.isfinite(sigma)&(sigma>0)
        vv=np.where(valid,v,0);ss=np.where(valid,sigma,0);snr=np.divide(vv,ss,out=np.zeros_like(vv),where=ss>0)
        conversion=cfg['conversions']['alpha_co10_including_helium']/cfg['conversions']['co21_to_co10']
        for n in [1,4]:
            allw=weights(h,data.shape,cfg['geometry'],radii,n)
            for radius,weight in zip(radii,allw):
                w=weight*valid;area=w.sum();signed=float(w@vv);positive=float(w@np.maximum(vv,0));sig=float(np.linalg.norm(w*ss));upper=float(w@ss)
                nominal_area=np.pi*(radius*1000)**2*np.cos(np.deg2rad(inclination(cfg['geometry'])))
                parts={name:dict(area_fraction=float(w[mask].sum()/area),signed_luminosity=float(w[mask]@vv[mask])) for name,mask in [('negative',vv<0),('snr_below1',snr<1),('snr_atleast3',snr>=3)]}
                row=dict(galaxy=cfg['object_id'],radius_kpc=radius,subdivisions=n,valid_native_pixels=int(np.count_nonzero(w)),covered_projected_area_pc2=float(area),nominal_aperture_projected_area_pc2=float(nominal_area),coverage_fraction=float(area/nominal_area),native_image_aperture_area_pc2=float(weight.sum()),signed_co_luminosity=signed,positive_clipped_luminosity=positive,clipping_increment=positive-signed,clipping_increment_fraction_of_signed=(positive/signed-1 if signed!=0 else None),zero_signal_clipping_reference=upper/np.sqrt(2*np.pi),independent_pixel_sigma=sig,fully_correlated_sigma_bound=upper,fixed_conversion_msun_per_luminosity=conversion,signed_mass_proxy=signed*conversion,parts=parts)
                assert abs(row['signed_mass_proxy']/conversion-signed)<max(1,abs(signed))*1e-12
                rows.append(row)
    save(out/'summary.json',dict(status='SOURCE_PHOTOMETRY_DIAGNOSTIC_NOT_MASS_POSTERIOR',rows=rows,raw_source_modified=False,observed_motion_accessed=False))
    print(json.dumps([dict(galaxy=r['galaxy'],r=r['radius_kpc'],coverage=r['coverage_fraction'],clipping_fraction=r['clipping_increment_fraction_of_signed']) for r in rows if r['subdivisions']==4],indent=2))

if __name__=='__main__':run()
