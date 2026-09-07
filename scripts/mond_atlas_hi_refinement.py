"""Factorized actual-source HI refinement with streamed native3Dpositions."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from scipy.special import roots_laguerre
from scipy.interpolate import RegularGridInterpolator
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from threadpoolctl import threadpool_limits
from mond_atlas_hi_column_centroid import moment_matrix
from mond_atlas_source_resolution import cell_projection_matrix
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'work/gravity-first-principles/mond-atlas-hi-refinement-001';PRIVATE=ROOT/'work/private/mond-atlas-hi-refinement-001'
HE=1.36;D=3.611;COEF=235631.09406987214;HEIGHT=.2
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
def projector():
 g=read(ROOT/'work/gravity-first-principles/mond-atlas-generic-source-001/run-002/summary.json')['cases'][0]['geometry'];h=fits.getheader(ROOT/'work/private/things-observable-12gal-003/NGC_2976_NA_CUBE_THINGS.FITS');w=WCS(h).celestial;native=w.pixel_to_world(0,0).frame;origin=SkyCoord(g['ra_deg']*u.deg,g['dec_deg']*u.deg);pa=np.deg2rad(g['pa_deg']);inc=np.deg2rad(g['inclination_deg'])
 def project(x,y,z,manual=False):
  E=x*np.sin(pa)+y*np.cos(pa)*np.cos(inc)-z*np.cos(pa)*np.sin(inc);N=x*np.cos(pa)-y*np.sin(pa)*np.cos(inc)+z*np.sin(pa)*np.sin(inc)
  sky=SkyCoord(lon=E/(D*1000)*u.rad,lat=N/(D*1000)*u.rad,frame=origin.skyoffset_frame()).transform_to(native)
  if not manual:return np.column_stack(w.world_to_pixel(sky))
  da=sky.ra.rad-np.deg2rad(h['CRVAL1']);dec=sky.dec.rad;dec0=np.deg2rad(h['CRVAL2']);xi=np.cos(dec)*np.sin(da);eta=np.sin(dec)*np.cos(dec0)-np.cos(dec)*np.sin(dec0)*np.cos(da)
  return np.column_stack([np.rad2deg(xi)/h['CDELT1']+h['CRPIX1']-1,np.rad2deg(eta)/h['CDELT2']+h['CRPIX2']-1])
 return project,g

def load_planar(path):
 with np.load(path) as z:return {k:z[k].copy() for k in ['x_major_kpc','y_minor_kpc','mass_HI_msun','flux_jy_km_s','radius_kpc','phi_rad'] if k in z.files}
def iter_native_batches(packet,vertical_order=48,max_nodes=65536):
 """Yieldxy,flux,planar_index arrays; packetdictorNPZpath. Neverdropsourceflux."""
 d=load_planar(packet) if isinstance(packet,(str,Path)) else packet
 z,w=roots_laguerre(vertical_order);z=np.r_[-z[::-1],z]*HEIGHT;w=np.r_[w[::-1],w]*.5;nz=len(z);batch=max(1,max_nodes//nz);project,_=projector()
 for start in range(0,len(d['x_major_kpc']),batch):
  stop=min(start+batch,len(d['x_major_kpc']));ids=np.repeat(np.arange(start,stop),nz);x=np.repeat(d['x_major_kpc'][start:stop],nz);y=np.repeat(d['y_minor_kpc'][start:stop],nz);zz=np.tile(z,stop-start)
  yield dict(xy_pixel=project(x,y,zz),flux_jy_km_s=(d['flux_jy_km_s'][start:stop,None]*w[None,:]).ravel(),planar_index=ids)

def independent_cell(S,axis,i,j,cells,step):
 interp=RegularGridInterpolator((axis,axis),S,bounds_error=False,fill_value=0);a,b=cells[i]-step/2,cells[i]+step/2;c,e=cells[j]-step/2,cells[j]+step/2;ex=np.r_[a,axis[(axis>a)&(axis<b)],b];ey=np.r_[c,axis[(axis>c)&(axis<e)],e];nodes=[];weights=[]
 for edges in [ex,ey]:
  ns=[];ws=[]
  for lo,hi in zip(edges[:-1],edges[1:]):ns.extend([(lo+hi)/2-(hi-lo)/(2*np.sqrt(3)),(lo+hi)/2+(hi-lo)/(2*np.sqrt(3))]);ws.extend([(hi-lo)/2]*2)
  nodes.append(ns);weights.append(ws)
 X,Y=np.meshgrid(*nodes,indexing='ij');q=interp(np.stack([X,Y],axis=-1))*np.outer(*weights);m=q.sum();return m,float((q*X).sum()/m),float((q*Y).sum()/m)

def build(S,axis,step):
 cells=np.arange(round(16/step))*step-8+step/2;h=axis[1]-axis[0];A=cell_projection_matrix(cells,step,axis,h,0);M=moment_matrix(cells,step,axis,h);plane=A@S@A.T;mx=M@S@A.T;my=A@S@M.T;keep=plane>0;X,Y=np.meshgrid(cells,cells,indexing='ij');x=mx[keep]/plane[keep];y=my[keep]/plane[keep];mass=plane[keep]*step*step*1e6;values=RegularGridInterpolator((axis,axis),S,bounds_error=False,fill_value=0)(np.c_[x,y]);positive=bool((values>0).all());inside=bool(np.all(abs(x-X[keep])<=step/2+1e-10)&np.all(abs(y-Y[keep])<=step/2+1e-10));inds=np.flatnonzero(keep);err=[];me=[]
 for idx in inds[np.linspace(0,len(inds)-1,min(128,len(inds)),dtype=int)]:
  i,j=np.unravel_index(idx,plane.shape);m,cx,cy=independent_cell(S,axis,i,j,cells,step);err.extend([abs(cx-mx[i,j]/plane[i,j]),abs(cy-my[i,j]/plane[i,j])]);me.append(abs(m-plane[i,j]*step*step)/m)
 return dict(x_major_kpc=x,y_minor_kpc=y,mass_HI_msun=mass,flux_jy_km_s=mass/(COEF*D*D),radius_kpc=np.hypot(x,y),phi_rad=np.arctan2(y,x)),dict(step_kpc=step,planar_nodes=len(x),all_positive_source_locations=positive,all_centroids_inside_cell=inside,moment_reference_max_error_kpc=float(max(err)),independent_mass_relative_error=float(max(me)),min_radius_kpc=float(np.hypot(x,y).min()),max_radius_kpc=float(np.hypot(x,y).max()),HI_mass_msun=float(mass.sum()),flux_jy_km_s=float(mass.sum()/(COEF*D*D)))

def main():
 t=time.monotonic();out=P/'run001';out.mkdir(exist_ok=False);priv=PRIVATE/'run001';priv.mkdir(parents=True,exist_ok=False)
 source=ROOT/'work/private/mond-atlas-source-resolution-001/run-001/atomic_helium-h0p2-f4.npz';old=ROOT/'work/private/mond-atlas-hi-column-001/centroid002/f4-emission-0.0625.npz';receipt=read(ROOT/'work/gravity-first-principles/mond-atlas-hi-column-001/completion-receipt.json');assert sha(old)==receipt['private_assets'][old.relative_to(ROOT).as_posix()]['sha256']
 deps=[Path(__file__),P/'PREFLIGHT.md',source,old,ROOT/'scripts/mond_atlas_hi_column_centroid.py',ROOT/'scripts/mond_atlas_source_resolution.py',ROOT/'work/gravity-first-principles/mond-atlas-generic-source-001/run-002/summary.json'];bound={p.relative_to(ROOT).as_posix():sha(p) for p in deps};save(out/'bindings.json',bound)
 # Analytic triangular source has unitmass andzero global firstmoments.
 a=np.arange(-2,2.01,.25);S=np.outer(np.maximum(1-abs(a),0),np.maximum(1-abs(a),0));c=np.arange(-1.9375,2,.125);A=cell_projection_matrix(c,.125,a,.25,0);M=moment_matrix(c,.125,a,.25);assert abs((A@S@A.T).sum()*.125**2-1)<1e-10;assert max(abs((M@S@A.T).sum()),abs((A@S@M.T).sum()))<1e-10
 project,g=projector();q=np.array([[0,0,0],[1,2,.3],[-2,1,-.5]],float);pe=float(np.max(abs(project(*q.T)-project(*q.T,manual=True))));assert pe<1e-6
 zerr=[]
 for order in [24,48]:
  z,w=roots_laguerre(order);zerr.append(max(abs(w.sum()-1),abs(w@z-1),abs(w@(z*z)-2)))
 assert max(zerr)<1e-10;save(out/'pre-source-controls.json',dict(passed=True,native_projection_error_pixel=pe,exponential_moment_errors=zerr))
 with np.load(source) as z:S=z['intrinsic_effective_surface']/HE;axis=z['latent_axis']
 full=S.sum()*(axis[1]-axis[0])**2*1e6;rows=[];assets=[];batches=[]
 for step in [.0625,.03125]:
  d,r=build(S,axis,step);r['mass_relative_error']=abs(r['HI_mass_msun']/full-1);assert r['all_positive_source_locations'] and r['all_centroids_inside_cell'] and r['moment_reference_max_error_kpc']<1e-10 and r['independent_mass_relative_error']<1e-8 and r['mass_relative_error']<1e-10
  if step==.0625:
   with np.load(old) as z:replay=max(float(np.max(abs(d[k]-z[k]))) for k in ['x_major_kpc','y_minor_kpc','mass_HI_msun','flux_jy_km_s'])
   assert replay<1e-12;r['old_fine_array_replay_max_error']=replay
  z24,w24=roots_laguerre(24);z48,w48=roots_laguerre(48);path=priv/f'HI-planar-{step}.npz';np.savez_compressed(path,**d,z24_kpc=np.r_[-z24[::-1],z24]*HEIGHT,z24_weight=np.r_[w24[::-1],w24]*.5,z48_kpc=np.r_[-z48[::-1],z48]*HEIGHT,z48_weight=np.r_[w48[::-1],w48]*.5);assets.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),step_kpc=step));rows.append(r)
  for order in [24,48]:
   total=0.;n=0;first=None
   for b in iter_native_batches(d,order):
    assert np.isfinite(b['xy_pixel']).all();total+=b['flux_jy_km_s'].sum();n+=len(b['planar_index'])
    if first is None:first=b
   error=abs(total/r['flux_jy_km_s']-1);assert error<1e-10
   batches.append(dict(step_kpc=step,vertical_order_per_side=order,expanded_nodes=n,flux_relative_error=float(error),persisted_full_nodes=False))
  print(r,flush=True)
 assert all(sha(ROOT/k)==v for k,v in bound.items());save(out/'summary.json',dict(status='ACTUAL_FINER_HI_EMITTER_INPUT_PASS',cases=rows,assets=assets,batch_checks=batches,geometry=g,source_mass_msun=float(full),seconds=time.monotonic()-t,observed_response_access=False,spectral_convergence_claimed=False,private_bytes=sum((ROOT/a['path']).stat().st_size for a in assets)))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
