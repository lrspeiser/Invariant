"""Full3D native-position export of the frozen HI emitter quadrature."""
from pathlib import Path
import json,hashlib
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'work/gravity-first-principles/mond-atlas-hi-column-001';PRIVATE=ROOT/'work/private/mond-atlas-hi-column-001'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
def main():
 out=P/'projection001';out.mkdir(exist_ok=False);priv=PRIVATE/'projection001';priv.mkdir(parents=True,exist_ok=False)
 summary=read(P/'run001/summary.json');g=read(ROOT/'work/gravity-first-principles/mond-atlas-generic-source-001/run-002/summary.json')['cases'][0]['geometry']
 cubepath=ROOT/'work/private/things-observable-12gal-003/NGC_2976_NA_CUBE_THINGS.FITS';h=fits.getheader(cubepath);w=WCS(h).celestial;nativeframe=w.pixel_to_world(0,0).frame;origin=SkyCoord(g['ra_deg']*u.deg,g['dec_deg']*u.deg);pa=np.deg2rad(g['pa_deg']);inc=np.deg2rad(g['inclination_deg']);D=g['distance_mpc']*1000
 def project(x,y,z):
  E=x*np.sin(pa)+y*np.cos(pa)*np.cos(inc)-z*np.cos(pa)*np.sin(inc);N=x*np.cos(pa)-y*np.sin(pa)*np.cos(inc)+z*np.sin(pa)*np.sin(inc)
  sky=SkyCoord(lon=E/D*u.rad,lat=N/D*u.rad,frame=origin.skyoffset_frame()).transform_to(nativeframe)
  px,py=w.world_to_pixel(sky)
  return np.column_stack([px,py]),sky
 def reference(sky):
  ra,dec=sky.ra.rad,sky.dec.rad;ra0=np.deg2rad(h['CRVAL1']);dec0=np.deg2rad(h['CRVAL2']);delta=ra-ra0
  xi=np.cos(dec)*np.sin(delta);eta=np.sin(dec)*np.cos(dec0)-np.cos(dec)*np.sin(dec0)*np.cos(delta)
  return np.column_stack([np.rad2deg(xi)/h['CDELT1']+h['CRPIX1']-1,np.rad2deg(eta)/h['CDELT2']+h['CRPIX2']-1])
 xyz=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]],float);xy,sky=project(*xyz.T);err=float(np.max(abs(xy-reference(sky))));assert err<1e-6
 save(out/'pre-array-controls.json',dict(manual_SIN_error_pixel=err,cube_values_read=False,source_geometry=g,native_source_header_sha256=hashlib.sha256(h.tostring().encode()).hexdigest(),script_sha256=sha(Path(__file__)),addendum_sha256=sha(P/'PROJECTION_ADDENDUM.md')))
 rows=[]
 for asset in summary['assets']:
  if asset['role']!='factorized intrinsic integrated HI flux':continue
  path=ROOT/asset['path'];assert sha(path)==asset['sha256']
  with np.load(path) as d:
   n=len(d['x_major_kpc']);nz=len(d['z_kpc']);x=np.repeat(d['x_major_kpc'],nz);y=np.repeat(d['y_minor_kpc'],nz);z=np.tile(d['z_kpc'],n);flux=(d['flux_jy_km_s'][:,None]*d['z_weight'][None,:]).ravel();r=np.hypot(x,y);phi=np.arctan2(y,x);total=float(d['flux_jy_km_s'].sum())
  xy=np.empty((len(x),2));maxerror=0.
  for start in range(0,len(x),65536):
   end=min(len(x),start+65536);q,sky=project(x[start:end],y[start:end],z[start:end]);xy[start:end]=q
   if start==0:maxerror=float(np.max(abs(q[:128]-reference(sky[:128]))))
  supported=(r>=.05)&(r<=6);assert np.isfinite(xy).all() and abs(flux.sum()/total-1)<1e-10 and maxerror<1e-6
  pp=priv/path.name;np.savez_compressed(pp,xy_pixel=xy,flux_jy_km_s=flux,radius_kpc=r,phi_rad=phi,z_kpc=z,force_table_supported=supported)
  rows.append(dict(path=pp.relative_to(ROOT).as_posix(),sha256=sha(pp),source_packet=asset['path'],source_sha256=asset['sha256'],emitters=len(flux),integrated_flux_jy_km_s=float(flux.sum()),flux_relative_error=float(abs(flux.sum()/total-1)),manual_SIN_error_pixel=maxerror,force_unsupported_nodes=int((~supported).sum()),force_unsupported_flux_fraction=float(flux[~supported].sum()/total),coordinate_unit='zero-based native FITS pixels',flux_unit='Jy km/s per3Dnode, pureHI, noPBcorrection',LOS_velocity_present=False))
 save(out/'summary.json',dict(status='ACTUAL_3D_EMITTER_POSITIONS_EXPORTED',assets=rows,observed_response_access=False,force_at_unsupported_nodes_not_invented=True,private_bytes=sum((ROOT/r['path']).stat().st_size for r in rows)))
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
