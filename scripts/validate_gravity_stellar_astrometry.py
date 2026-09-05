"""Compare conflicting stellar WCS interpretations against independent Gaia stars."""
import concurrent.futures,csv,hashlib,io,json,shutil,urllib.parse,urllib.request,warnings
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter,maximum_filter
from scipy.spatial import cKDTree
from astropy.io import fits
from astropy.wcs import WCS,NoConvergence
from astropy import log
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'work/gravity-first-principles/stellar-gaia-alignment-001';D.mkdir(exist_ok=False)
P=ROOT/'work/private/gaia-stellar-alignment-001';P.mkdir(exist_ok=True)
shutil.copy2(__file__,D/'runner.py')
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False))
def read(p):return json.loads((ROOT/p).read_text())
geo={a['name']:a['geometry'] for a in read('work/gravity-first-principles/conditional-cube-pilot-001/data-audit.json')}
assets=read('work/gravity-first-principles/stellar-co-acquisition-001/receipt.json')['files']
save(D/'registration.json',dict(catalog='Gaia DR3 ESA TAP; G<18, parallax/error>5, RUWE<1.4, proper motion <50 mas/yr; cone .2 deg, brightest 150',
 epoch='Propagate to 2005; conservative source-epoch bracket 2000..2010, maximum .25 arcsec from proper-motion cutoff.',
 detection='Image-only high-pass local peaks >8 global robust sigma, 5-pixel maxima, positive 3x3 centroid.',
 selection='Catalog sorted by RA, alternating calibration/validation stars. Choose TAN or SIP on calibration clipped nearest-peak distance (12 arcsec).',
 gate='At least 4 calibration and 4 validation stars; validation median <1 arcsec and 90th percentile <2 arcsec; no fitted coordinate offset.',
 scope='Stellar astrometric check, no velocity or gravity targets used. S4G original P1 flux maps are checked; transfer to P5 requires a separate relative-registration check.'))

def acquire(name):
 g=geo[name]
 q=f"SELECT TOP 150 source_id,ra,dec,pmra,pmdec,ref_epoch,phot_g_mean_mag FROM gaiadr3.gaia_source WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{g['ra']},{g['dec']},0.2)) AND phot_g_mean_mag<18 AND parallax_over_error>5 AND ruwe<1.4 ORDER BY phot_g_mean_mag"
 data=urllib.parse.urlencode({'REQUEST':'doQuery','LANG':'ADQL','FORMAT':'csv','QUERY':q}).encode()
 # The NGC3198 pilot query is identical; retain its exact receipt.
 file=P/(name+'.csv')
 if not file.exists():
  with urllib.request.urlopen('https://gea.esac.esa.int/tap-server/tap/sync',data=data,timeout=45) as response:file.write_bytes(response.read())
 existing=next(a for a in assets if a['name']==name and a['role'] in ('STELLAR_MASS_MAP','STELLAR_IRAC1_FLUX'))
 if existing['role']=='STELLAR_MASS_MAP':
  url=f'https://irsa.ipac.caltech.edu/data/SPITZER/S4G/galaxies/{name}/P1/{name}.phot.1.fits';image=P/(name+'.P1.fits')
  if not image.exists():
   with urllib.request.urlopen(url,timeout=45) as response:image.write_bytes(response.read())
 else:url=existing['url'];image=ROOT/existing['file']
 return dict(name=name,query=q,catalog_file=str(file.relative_to(ROOT)),catalog_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
  image_url=url,image_file=str(image.relative_to(ROOT)),image_sha256=hashlib.sha256(image.read_bytes()).hexdigest())

def check(a):
 with fits.open(ROOT/a['image_file']) as hd:image=np.squeeze(hd[0].data).astype(float);header=hd[0].header.copy()
 finite=np.isfinite(image);safe=np.where(finite,image,0);high=safe-gaussian_filter(safe,3)
 noise=1.4826*np.median(abs(high[finite]-np.median(high[finite])))
 mask=finite&(high==maximum_filter(high,5))&(high>8*max(noise,1e-8))
 yy,xx=np.where(mask);peaks=[]
 for y,x in zip(yy,xx):
  if min(y,x)<4 or y>=image.shape[0]-4 or x>=image.shape[1]-4:continue
  patch=np.maximum(high[y-1:y+2,x-1:x+2],0);weight=patch.sum()
  sy,sx=np.mgrid[y-1:y+2,x-1:x+2]
  if weight>0:peaks.append([float((sx*patch).sum()/weight),float((sy*patch).sum()/weight)])
 tree=cKDTree(peaks);cat=list(csv.DictReader((ROOT/a['catalog_file']).read_text().splitlines()))
 cat=[c for c in cat if np.hypot(float(c['pmra']),float(c['pmdec']))<50]
 cat.sort(key=lambda c:float(c['ra']))
 ra=np.array([float(c['ra'])+float(c['pmra'])*(2005-float(c['ref_epoch']))/3.6e6/np.cos(np.deg2rad(float(c['dec']))) for c in cat])
 dec=np.array([float(c['dec'])+float(c['pmdec'])*(2005-float(c['ref_epoch']))/3.6e6 for c in cat])
 w=WCS(header).celestial;plain=w.deepcopy();plain.sip=None
 linear=np.column_stack(plain.all_world2pix(ra,dec,0));scale=float(np.sqrt(abs(np.linalg.det(plain.pixel_scale_matrix)))*3600)
 inside=(linear[:,0]>16)&(linear[:,1]>16)&(linear[:,0]<image.shape[1]-17)&(linear[:,1]<image.shape[0]-17)
 ra,dec=ra[inside],dec[inside];linear=linear[inside];identifiers=np.array([c['source_id'] for c in cat])[inside]
 try:distorted=np.column_stack(w.all_world2pix(ra,dec,0,maxiter=100))
 except NoConvergence as e:
  distorted=e.best_solution
  for indices in (e.divergent,e.slow_conv):
   if indices is not None:distorted[indices]=np.nan
 modes={}
 for label,xy in [('linear_tan',linear),('header_sip',distorted)]:
  valid=np.all(np.isfinite(xy),axis=1);distance=np.ones(len(xy))*12
  distance[valid]=np.minimum(tree.query(xy[valid])[0]*scale,12)
  modes[label]=distance
 calibration=np.arange(len(ra))%2==0;validation=~calibration
 if not min(calibration.sum(),validation.sum()):return dict(**a,status='INSUFFICIENT_STARS',stars=len(ra))
 choice=min(modes,key=lambda label:np.mean(modes[label][calibration]))
 d=modes[choice][validation];passed=bool(min(calibration.sum(),validation.sum())>=4 and np.median(d)<1 and np.quantile(d,.9)<2)
 return dict(**a,status='PASS' if passed else 'FAIL_OR_INSUFFICIENT',selected_wcs=choice,stars=int(len(ra)),peak_count=len(peaks),
  validation_median_arcsec=float(np.median(d)),validation_p90_arcsec=float(np.quantile(d,.9)),
  modes={key:dict(calibration_mean_arcsec=float(np.mean(v[calibration])),validation_median_arcsec=float(np.median(v[validation])),validation_p90_arcsec=float(np.quantile(v[validation],.9))) for key,v in modes.items()},
  matches=[dict(source_id=str(i),calibration=bool(c),linear_distance_arcsec=float(l),sip_distance_arcsec=float(s)) for i,c,l,s in zip(identifiers,calibration,modes['linear_tan'],modes['header_sip'])])

log.setLevel('ERROR');warnings.filterwarnings('ignore',module='astropy')
results=[];errors=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 tasks={pool.submit(acquire,name):name for name in geo}
 for task in concurrent.futures.as_completed(tasks):
  name=tasks[task]
  try:
   a=task.result();row=check(a);results.append(row);save(D/(name+'.json'),row)
   print(name,row['status'],row.get('selected_wcs'),row.get('stars'),row.get('validation_median_arcsec'),flush=True)
  except Exception as e:errors.append(dict(name=name,error=repr(e)));print('FAIL',name,repr(e),flush=True)
save(D/'result.json',dict(status='COMPLETE' if not errors else 'INCOMPLETE',objects=results,errors=errors))
