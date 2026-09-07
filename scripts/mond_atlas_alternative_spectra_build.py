from mond_atlas_actual_spectra import ROOT,load_instrument
from mond_atlas_actual_spectra_build import save,digest
from mond_atlas_hi_refinement import load_planar,iter_native_batches
from mond_atlas_native_emission_sampled import grouped_aperture_flux
from scipy.interpolate import PchipInterpolator
import numpy as np,json,csv,gzip,time
from pathlib import Path
P=ROOT/'work/gravity-first-principles/mond-atlas-alternative-spectra-001';D=ROOT/'work/private/mond-atlas-alternative-spectra-001'
S=ROOT/'work/gravity-first-principles/mond-atlas-source-sensitivity-001'
CASES={'common30':'common30_gaussian_approximation','common30_stars_h0.4':'common30_gaussian_approximation','missing_zero':'missing_zero','missing_annular':'missing_annular'}
def readgz(path):
 with gzip.open(path,'rt',newline='') as f:return list(csv.DictReader(f))
def run():
 out=P/'cache001';out.mkdir(exist_ok=False);priv=D/'cache001';priv.mkdir(parents=True,exist_ok=True);started=time.monotonic();fr=readgz(S/'fields001/radial-fields.csv.gz');pr=readgz(S/'hi001/pressure-profiles.csv.gz');inst=load_instrument();assets=[]
 paths=[ROOT/f'work/private/mond-atlas-alternative-hi-001/run001/{case}-HI-planar-{step}.npz' for case in ['common30','missing_zero','missing_annular'] for step in [.0625,.03125]]
 deps=[ROOT/'work/gravity-first-principles/mond-atlas-alternative-hi-001/run001/summary.json',Path(__file__),P/'PREFLIGHT.md',S/'fields001/source-contract.json',S/'fields001/radial-fields.csv.gz',S/'hi001/pressure-profiles.csv.gz',ROOT/'scripts/mond_atlas_hi_refinement.py',ROOT/'scripts/mond_atlas_native_emission_sampled.py']+paths;save(out/'bindings.json',{str(p.relative_to(ROOT)):digest(p) for p in deps})
 for case,branch in CASES.items():
  sourcecase='common30' if case.startswith('common30') else case
  for step in [.0625,.03125]:
   path=ROOT/f'work/private/mond-atlas-alternative-hi-001/run001/{sourcecase}-HI-planar-{step}.npz';planar=load_planar(path);r=planar['radius_kpc'];phi=planar['phi_rad'];p=sorted([x for x in pr if x['branch']==branch and float(x['radial_spacing_kpc'])==.00625],key=lambda x:float(x['r_kpc']));rr=np.array([float(x['r_kpc']) for x in p]);inside=(r>=rr.min())&(r<=rr.max());den=PchipInterpolator(rr,[float(x['raw_sigma_hi']) for x in p],extrapolate=False)(r);gradient=PchipInterpolator(rr,[float(x['smoothed_gradient']) for x in p],extrapolate=False)(r)
   arrays=dict(radius_kpc=r,phi_rad=phi,pressure_gradient=gradient,sin_inclination=np.sin(np.deg2rad(53.86233095)),planar_flux=planar['flux_jy_km_s']);force={}
   for grid in ['nf','lf']:
    vectors=[]
    for comp in ['stellar_luminosity','atomic_helium','co21']:
     rows=sorted([x for x in fr if x['case']==case and x['grid']==grid and x['component']==comp and int(x['azimuth_nodes'])==2048],key=lambda x:float(x['r_kpc']));rad=np.array([float(x['r_kpc']) for x in rows]);inside&=(r>=rad.min())&(r<=rad.max());vals=np.array([float(x['gbar_inward']) if x['gbar_inward'] else np.nan for x in rows]);assert np.isfinite(vals).all();vectors.append(PchipInterpolator(rad,vals,extrapolate=False)(r))
    force[grid]=np.array(vectors)
   inside&=np.isfinite(den)&(den>0)&np.isfinite(gradient);arrays.update(force_supported=inside,den_h0p1=den,force_h0p1_newton=force['nf'],force_h0p1_newton_plus_log=force['nf']+force['lf'])
   for order in [24,48]:
    grouped=np.zeros((15,len(r)));flux=0.;count=0
    for batch in iter_native_batches(planar,vertical_order=order,max_nodes=65536):
     grouped+=grouped_aperture_flux(batch['xy_pixel'],batch['flux_jy_km_s'],batch['planar_index'],inst,group_count=len(r));flux+=batch['flux_jy_km_s'].sum();count+=len(batch['planar_index'])
    closure=abs(flux/planar['flux_jy_km_s'].sum()-1);assert closure<1e-8;label=f'p{str(step).replace(".","p")}_z{order}';target=priv/f'{case}__{label}.npz';np.savez_compressed(target,**arrays,grouped_flux=grouped);assets.append(dict(case=case,label=label,cache=str(target.relative_to(ROOT)),sha256=digest(target),source_packet=str(path.relative_to(ROOT)),planar_nodes=len(r),projected_nodes=count,unsupported_groups=int((~inside).sum()),unsupported_weighted_flux_max_jy_km_s=float(grouped[:,~inside].sum(1).max()),closure=closure,total_flux_jy_km_s=float(planar['flux_jy_km_s'].sum())));print(case,label,'done',flush=True)
 save(out/'summary.json',dict(assets=assets,seconds=time.monotonic()-started,observed_source_spectra_read=0));assert all(digest(ROOT/p)==h for p,h in json.loads((out/'bindings.json').read_text()).items())
if __name__=='__main__':run()
