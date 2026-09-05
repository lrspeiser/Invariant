"""Astrometrically gated, common-beam projected matter pilot; no 3D inversion."""
import csv,hashlib,json,shutil,urllib.request,warnings
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from astropy.io import fits
from astropy.wcs import WCS
from astropy import log
from run_gravity_cube_pilot import beam_values
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'work/gravity-first-principles/ngc2903-matter-001';D.mkdir(exist_ok=False)
P=ROOT/'work/private/ngc2903-matter-001';P.mkdir(exist_ok=False)
shutil.copy2(__file__,D/'runner.py')
def read(p):return json.loads((ROOT/p).read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False))
registration=dict(name='NGC2903',source_selection='Only object passing stellar-gaia-alignment-001 strict holdout gate.',
 target_fwhm_arcsec=48,support_threshold=.98,
 stellar_mask='All nonzero ICA mask labels excluded, including negative recursive-ICA labels, per publisher README.',
 stellar_mass_to_light=[.4,.6,.8],co_alpha10_including_helium=[2.,4.35,8.],co_r21=[.5,.65,1.],
 uncertainties='Parameter ranges are illustrative source-conversion sensitivity, not confidence intervals. CO propagated error uses fully correlated-noise upper bound, not independence.',
 primary_sources=['https://irsa.ipac.caltech.edu/data/SPITZER/S4G/docs/P5_README.html','https://www.iram.fr/ILPA/LP001/README'],
 limits=['Projected surface mass only, not volume density.','CO integration windows partly depend on HI mean velocity, so tracer products are not fully independent.','HI and stellar measurement uncertainty not fully propagated.','No missing source area is filled with measured zero density.'])
save(D/'registration.json',registration)
assets={a['role']:a for a in read('work/gravity-first-principles/stellar-co-acquisition-001/receipt.json')['files'] if a['name']=='NGC2903'}
alignment=read('work/gravity-first-principles/stellar-gaia-alignment-001/NGC2903.json')
assert alignment['status']=='PASS' and alignment['selected_wcs']=='linear_tan'
url='https://irsa.ipac.caltech.edu/data/SPITZER/S4G/galaxies/NGC2903/P5/NGC2903.nonstellar.fits';dustfile=P/'NGC2903.nonstellar.fits'
with urllib.request.urlopen(url,timeout=45) as response:dustfile.write_bytes(response.read())
def load(file):
 with fits.open(file) as hd:a=np.squeeze(hd[0].data).astype(float);h=hd[0].header.copy()
 w=WCS(h).celestial;w.sip=None
 return a,h,w
log.setLevel('ERROR');warnings.filterwarnings('ignore',module='astropy')
star,sh,sw=load(ROOT/assets['STELLAR_MASS_MAP']['file']);mask,_,_=load(ROOT/assets['STELLAR_ICA_MASK']['file'])
dust,_,_=load(dustfile);p1,_,w1=load(ROOT/alignment['image_file'])
yy,xx=np.mgrid[:star.shape[0],:star.shape[1]];ra,dec=sw.all_pix2world(xx,yy,0);px,py=w1.all_world2pix(ra,dec,0)
observed=map_coordinates(p1,[py,px],order=1,mode='constant',cval=np.nan)
valid=(mask==0)&np.isfinite(star)&np.isfinite(dust)&np.isfinite(observed)
bright=valid&(observed>np.nanpercentile(observed[valid],80))
combined=star+dust
relative=float(np.sqrt(np.mean((combined[bright]-observed[bright])**2))/np.sqrt(np.mean(observed[bright]**2)))
correlation=float(np.corrcoef(combined[bright],observed[bright])[0,1])
alignment_pass=relative<.05 and correlation>.99
save(D/'p1-p5-registration.json',dict(relative_flux_rms=relative,correlation=correlation,pass_gate=alignment_pass,
 dust_url=url,dust_sha256=hashlib.sha256(dustfile.read_bytes()).hexdigest(),scope='P5 stellar+nonstellar reconstruction against Gaia-validated P1 in common TAN coordinates.'))
if not alignment_pass:
 save(D/'result.json',dict(status='BLOCKED_BY_RELATIVE_REGISTRATION',relative_flux_rms=relative,correlation=correlation));raise SystemExit('P1/P5 transfer did not pass')
packet=dict(np.load(ROOT/'work/private/conditional-cube-pilot-001/NGC2903.npz'))
g=next(a['geometry'] for a in read('work/gravity-first-principles/conditional-cube-pilot-001/data-audit.json') if a['name']=='NGC2903')
targetra=packet['east']/np.cos(np.deg2rad(g['dec']))/3600+g['ra'];targetdec=packet['north']/3600+g['dec']
def matched(a,w,good,beam,conversion=1.):
 # Correct native elliptical beam in sky coordinates, then sample common WCS.
 major,minor,angle=beam;angle=np.deg2rad(angle)
 u=np.array([np.sin(angle),np.cos(angle)]);v=np.array([np.cos(angle),-np.sin(angle)])
 sky=(np.eye(2)*48**2-np.outer(u,u)*major**2-np.outer(v,v)*minor**2)/2.354820045**2
 inverse=np.linalg.inv(w.pixel_scale_matrix*3600);cov=inverse@sky@inverse.T
 assert np.linalg.eigvalsh(cov).min()>0
 size=int(np.ceil(5*np.sqrt(np.linalg.eigvalsh(cov).max())))
 y,x=np.mgrid[-size:size+1,-size:size+1];points=np.stack([x,y],axis=-1);inv=np.linalg.inv(cov)
 kernel=np.exp(-.5*np.einsum('...i,ij,...j->...',points,inv,points));kernel/=kernel.sum()
 support=fftconvolve(good.astype(float),kernel,mode='same')
 smoothed=fftconvolve(np.where(good,a*conversion,0),kernel,mode='same')/np.maximum(support,1e-12)
 x,y=w.all_world2pix(targetra,targetdec,0)
 return map_coordinates(smoothed,[y,x],order=1,mode='constant',cval=np.nan),map_coordinates(support,[y,x],order=1,mode='constant',cval=0)
stellar,ss=matched(star,sw,(mask==0)&np.isfinite(star),(1.7,1.7,0),704.04)
moment=next(a for a in read('work/gravity-first-principles/things-observable-acquisition-003/receipt.json')['files'] if a['name']=='NGC2903' and a['resolution']=='NA' and a['moment']==0)
hi,hh,hw=load(ROOT/moment['file']);beam=beam_values(hh)
# MOM0 is Jy/beam m/s, temperature conversion uses beam dimensions in arcsec.
nu=hh.get('RESTFREQ',1420405750)/1e9
hiconversion=.001*1222000/(nu**2*beam[0]*beam[1])*1.823e18/1.248e20*1.36
atomic,hs=matched(hi,hw,np.isfinite(hi)&(hi>0),beam,hiconversion)
co,ch,cw=load(ROOT/assets['CO21_MOM0']['file']);err,_,_=load(ROOT/assets['CO21_EMOM0']['file'])
cb=(ch['BMAJ']*3600,ch['BMIN']*3600,ch.get('BPA',0));cg=np.isfinite(co)&np.isfinite(err)&(err>0)
molecular,cs=matched(co,cw,cg,cb);coerror,_=matched(err,cw,cg,cb)
joint=(ss>=.98)&(hs>=.98)&(cs>=.98)&np.isfinite(stellar)&np.isfinite(atomic)&np.isfinite(molecular)
selected=packet['train_mask']|packet['test_mask'];admitted=joint&selected
stars=.6*np.maximum(stellar,0);h2=4.35/.65*np.maximum(molecular,0);total=stars+atomic+h2
# Signed CO and its error are retained separately, including nondetections.
co_upper=np.maximum(molecular,0)+3*coerror
lower=.4*np.maximum(stellar,0)+atomic+2.*np.maximum(molecular-3*coerror,0)
upper=.8*np.maximum(stellar,0)+atomic+8./.5*co_upper
np.savez_compressed(P/'projected-matter.npz',stellar_luminosity=stellar,atomic_with_helium=atomic,
 co21_intensity=molecular,co_error_upper_bound=coerror,stars_nominal=stars,h2_nominal=h2,total_nominal=total,
 sensitivity_lower=lower,sensitivity_upper=upper,joint_mask=joint,admitted_mask=admitted)
rows=[]
for y,x in zip(*np.where(admitted)):
 rows.append(dict(east_arcsec=float(packet['east'][y,x]),north_arcsec=float(packet['north'][y,x]),
  sigma_star_nominal=float(stars[y,x]),sigma_atomic_with_helium=float(atomic[y,x]),co21_signed=float(molecular[y,x]),
  co21_error_bound=float(coerror[y,x]),sigma_h2_nominal=float(h2[y,x]),sigma_total_nominal=float(total[y,x]),
  conditional_lower=float(lower[y,x]),conditional_upper=float(upper[y,x])))
if rows:
 with (D/'selected-surface-matter.csv').open('w',newline='') as file:
  writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
result=dict(status='COMPLETE_PROJECTED_SOURCE_PILOT',joint_selected_pixels=int(admitted.sum()),geometric_selected_pixels=int(selected.sum()),
 stellar_supported_pixels=int(np.sum(selected&(ss>=.98))),hi_supported_pixels=int(np.sum(selected&(hs>=.98))),co_supported_pixels=int(np.sum(selected&(cs>=.98))),
 nominal_median_atomic_fraction=float(np.median((atomic/np.maximum(total,1e-12))[admitted])) if rows else None,
 molecular_upper_limit_positions=int(np.sum(admitted&(molecular<3*coerror))),
 nominal_units='Solar masses per square parsec projected on sky; no cos(inclination) deprojection.',
 stellar_conversion='704.04 Lsun/pc2 per MJy/sr times assumed mass-to-light ratio; stellar map is flux, not preconverted mass.',
 total_volume_density_measured=False,uncertainty_model_complete=False,registration=registration)
save(D/'result.json',result);print(json.dumps(result,indent=2))
