"""Actual positive HI source columns and conservative factorized emission packets."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import simpson,quad
from scipy.special import roots_laguerre
from scipy.ndimage import map_coordinates
from threadpoolctl import threadpool_limits
from mond_atlas_source_resolution import cell_projection_matrix
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'work/gravity-first-principles/mond-atlas-hi-column-001';PRIVATE=ROOT/'work/private/mond-atlas-hi-column-001'
HE=1.36;D=3.611;ELL=.25;HEIGHT=.2;NU=1.42040575;AS=206264.80624709636
COEF=1e12/AS**2*(1222000/NU**2)*(1.823e18/1.248e20)*np.pi/(4*np.log(2))
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,default=lambda x:x.item())
def smooth(r,sigma,target):
 d=target[:,None]-r;plus=target[:,None]+r
 kernel=lambda x:np.exp(-.5*(x/ELL)**2)/(np.sqrt(2*np.pi)*ELL)
 km,kp=kernel(d),kernel(plus)
 return simpson((km+kp)*sigma,x=r,axis=1),simpson((-d*km-plus*kp)/ELL**2*sigma,x=r,axis=1)
def controls():
 r=np.arange(0,8+.00625,.0125);targets=np.array([0,.5,1.,2.]);sigma=np.exp(-r*r/(2*.7**2));s,ds=smooth(r,sigma,targets)
 width=np.sqrt(.7**2+ELL**2);truth=.7/width*np.exp(-targets**2/(2*width**2));gradient=-targets/width**2*truth
 error=max(np.max(abs(s-truth)),np.max(abs(ds-gradient)));assert error<1e-8
 constant,zero=smooth(r,np.ones_like(r),targets);assert max(np.max(abs(constant-1)),np.max(abs(zero)))<1e-8
 qerr=max(abs(quad(lambda u:np.exp(-u*u/(2*.7**2))*(np.exp(-(t-u)**2/(2*ELL**2))+np.exp(-(t+u)**2/(2*ELL**2)))/(np.sqrt(2*np.pi)*ELL),0,8,epsabs=1e-12)[0]-v) for t,v in zip(targets,s));assert qerr<1e-8
 axis=np.arange(-2,2.01,.25);source=np.outer(np.maximum(1-abs(axis),0),np.maximum(1-abs(axis),0));cells=np.arange(-1.9375,2,.125);A=cell_projection_matrix(cells,.125,axis,.25,0);plane=A@source@A.T;mass=plane.sum()*.125**2;assert abs(mass-1)<1e-10
 zchecks=[]
 for order in [12,24]:
  z,w=roots_laguerre(order);zchecks.append(dict(order=order,normalization_error=abs(w.sum()-1),mean_abs_z_error=abs(np.dot(w,z*HEIGHT)-HEIGHT),second_moment_error=abs(np.dot(w,(z*HEIGHT)**2)-2*HEIGHT**2)))
 assert max(v for row in zchecks for k,v in row.items() if k!='order')<1e-10
 # Invert beam intensity -> column -> mass and independently angular integrated flux.
 beamarea=np.pi/(4*np.log(2))*7.407*6.42384;factor=.001*1222000/(NU*NU*7.407*6.42384)*1.823e18/1.248e20*HE
 flux=1000*.001*2.25/beamarea;mass=1000*factor*2.25*(D*1e6/AS)**2/HE
 conversion_error=abs(mass/(COEF*D*D*flux)-1);assert conversion_error<1e-10
 return dict(passed=True,analytic_gaussian_max_error=float(error),independent_quad_error=float(qerr),cell_mass_error=float(abs(mass/(COEF*D*D*flux)-1)),conversion_error=conversion_error,zchecks=zchecks,coefficient=COEF,relative_vs_THINGS_236000=COEF/236000-1,relative_vs_235600=COEF/235600-1)
def sample(surface,axis,r,nphi):
 phi=(np.arange(nphi)+.5)*2*np.pi/nphi
 x=r[:,None]*np.cos(phi);y=r[:,None]*np.sin(phi)
 interp=RegularGridInterpolator((axis,axis),surface,bounds_error=False,fill_value=0)
 v=interp(np.stack([x,y],axis=-1));return v.mean(axis=1),v,phi

def main():
 t=time.monotonic();out=P/'run001';out.mkdir(exist_ok=False);priv=PRIVATE/'run001';priv.mkdir(parents=True,exist_ok=False)
 sources=read(ROOT/'work/gravity-first-principles/mond-atlas-spatial-program-001/source-bindings.json');bound={}
 for c in sources['source_cases']:
  for x in c['components']:
   if x['id']=='atomic_helium':bound[x['path']]=x['sha256']
 deps=[Path(__file__),P/'PREFLIGHT.md',ROOT/'scripts/mond_atlas_source_resolution.py',ROOT/'configs/mond_atlas_ngc2976_source_v1.json',ROOT/'work/gravity-first-principles/mond-atlas-generic-source-001/run-002/summary.json',ROOT/'work/gravity-first-principles/mond-atlas-observation-admission-001/first-score-protocol.json']
 frozen={p.relative_to(ROOT).as_posix():sha(p) for p in deps};save(out/'pre-source-bindings.json',dict(source_files=bound,dependencies=frozen));save(out/'controls.json',controls())
 summaries=[];assets=[];target=np.arange(1,241)*.025
 for path,expected in bound.items():
  assert sha(ROOT/path)==expected
  with np.load(ROOT/path) as z:surface=z['intrinsic_effective_surface'].copy()/HE;axis=z['latent_axis'].copy();coverage=z['coverage'].copy();observed_axis=z['observed_axis'].copy()
  assert np.isfinite(surface).all() and (surface>=0).all();h=axis[1]-axis[0];full=surface.sum()*h*h*1e6;case='f4' if '-f4.' in path else 'f1';profiles=[]
  for dr,np_ in [(.025,256),(.0125,512)]:
   rr=np.arange(round(8/dr)+1)*dr;column,angular,phi=sample(surface,axis,rr,np_);ss,der=smooth(rr,column,target);profiles.append((rr,column,ss,der))
  rr,column,ss,der=profiles[-1];_,angular,phi=sample(surface,axis,target,512)
  inner=(target>=.75)&(target<=2.5);reference=max(np.sqrt(np.mean(der[inner]**2)),1e-15);diff=profiles[0][3]-der
  derivative_relative=float(np.sqrt(np.mean(diff[inner]**2))/reference);maxpoint=float(np.max(abs(diff[inner])/np.maximum(abs(der[inner]),.1*np.median(abs(der[inner])))))
  column_relative=float(np.sqrt(np.mean((profiles[0][2][inner]-ss[inner])**2))/np.sqrt(np.mean(ss[inner]**2)))
  sp,_=smooth(rr,column,target+1e-4);sm,_=smooth(rr,column,target-1e-4);fd=float(np.sqrt(np.mean(((sp-sm)/.0002-der)[inner]**2))/reference)
  polar=simpson(2*np.pi*rr*column,x=rr)*1e6;smfull,_=smooth(rr,column,rr);smoothmass=simpson(2*np.pi*rr*smfull,x=rr)*1e6
  # Separate scipy grid interpolator, same actual source, random source-only coordinates.
  rng=np.random.default_rng(9063101);xy=rng.uniform(-5,5,(80,2));pred=RegularGridInterpolator((axis,axis),surface,bounds_error=False,fill_value=0)(xy);ref=map_coordinates(surface,((xy-axis[0])/h).T,order=1,mode='constant',cval=0,prefilter=False);interp_error=float(np.max(abs(pred-ref)))
  arrays=dict(radius_kpc=target,sigma_HI_msun_pc2=ss,d_sigma_HI_dR_msun_pc2_kpc=der,pressure_variance_km2_s2=np.array([25,100,225]),pressure_msun_pc2_km2_s2=ss[:,None]*np.array([25,100,225]),pressure_gradient=der[:,None]*np.array([25,100,225]),raw_radius_kpc=rr,raw_sigma_HI_msun_pc2=column,phi_rad=phi,angular_sigma_HI_msun_pc2=angular,angular_HI_normalized_weight=np.divide(angular,angular.sum(axis=1)[:,None],out=np.zeros_like(angular),where=angular.sum(axis=1)[:,None]>0),helium_factor=np.array(HE),distance_mpc=np.array(D),mass_flux_coefficient=np.array(COEF))
  pack=priv/f'{case}-column.npz';np.savez_compressed(pack,**arrays);assets.append(dict(path=pack.relative_to(ROOT).as_posix(),sha256=sha(pack),role='HI pressure and angular weighting'))
  emission=[]
  for step,order in [(.125,12),(.0625,24)]:
   centers=np.arange(round(16/step))*step-8+step/2;A=cell_projection_matrix(centers,step,axis,h,0);plane=A@surface@A.T;mass=plane*step*step*1e6;xx,yy=np.meshgrid(centers,centers,indexing='ij');keep=mass>0;z,w=roots_laguerre(order);z=np.r_[-z[::-1],z]*HEIGHT;weights=np.r_[w[::-1],w]*.5
   pathout=priv/f'{case}-emission-{step}.npz';np.savez_compressed(pathout,x_major_kpc=xx[keep],y_minor_kpc=yy[keep],cell_width_kpc=np.array(step),radius_kpc=np.hypot(xx[keep],yy[keep]),ring_index_025kpc=np.floor(np.hypot(xx[keep],yy[keep])/.025).astype(np.int32),mass_HI_msun=mass[keep],flux_jy_km_s=mass[keep]/(COEF*D*D),z_kpc=z,z_weight=weights,HI_scaling_factors=np.array([.8,1.,1.2]))
   assets.append(dict(path=pathout.relative_to(ROOT).as_posix(),sha256=sha(pathout),role='factorized intrinsic integrated HI flux'))
   emission.append(dict(spacing_kpc=step,vertical_order_per_side=order,planar_emitters=int(keep.sum()),total_HI_mass_msun=float(mass.sum()),total_flux_jy_km_s=float(mass.sum()/(COEF*D*D)),relative_mass_error=float(abs(mass.sum()/full-1)),outside_nominal_6kpc_mass_fraction=float(mass[np.hypot(xx,yy)>6].sum()/full),vertical_weight_sum=float(weights.sum())))
  passed=column_relative<.01 and derivative_relative<.02 and maxpoint<.1 and fd<1e-5 and abs(polar/full-1)<.001 and interp_error<1e-10 and all(x['relative_mass_error']<1e-10 for x in emission)
  summaries.append(dict(case=case,source_path=path,full_HI_mass_msun=float(full),full_atomic_helium_mass_msun=float(full*HE),full_flux_jy_km_s=float(full/(COEF*D*D)),column_refinement_relative=column_relative,derivative_refinement_relative=derivative_relative,derivative_max_normalized_point_change=maxpoint,derivative_finite_difference_relative=fd,polar_mass_relative_error=float(abs(polar/full-1)),smoothing_changes_radial_mass_fraction=float(smoothmass/full-1),independent_interpolator_max_error=interp_error,emission=emission,passed=passed))
 save(out/'summary.json',dict(status='ACTUAL_SOURCE_PACKET_CHECKED' if all(x['passed'] for x in summaries) else 'BENCHMARK_FAILED',observed_admission='SOURCE_BLOCKED',cases=summaries,assets=assets,observed_responses_opened=False,seconds=time.monotonic()-t,new_private_bytes=sum((ROOT/x['path']).stat().st_size for x in assets)))
 assert all(sha(ROOT/k)==v for k,v in frozen.items());assert time.monotonic()-t<300
 print(json.dumps(summaries,indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
