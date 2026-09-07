"""Matched alternative HI emission packets, without observed response access."""
from pathlib import Path
import json,time
import numpy as np
from scipy.special import roots_laguerre
from threadpoolctl import threadpool_limits
from mond_atlas_hi_refinement import build,iter_native_batches,projector,sha,HE,D,COEF,HEIGHT
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-alternative-hi-001'
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
    start=time.monotonic();out=P/'run001';out.mkdir(exist_ok=False);priv=R/'work/private/mond-atlas-alternative-hi-001/run001';priv.mkdir(parents=True,exist_ok=False)
    contract=R/'work/gravity-first-principles/mond-atlas-source-sensitivity-001/fields001/source-contract.json'
    cases=json.loads(contract.read_text())['cases'];selected={k:next(a for a in cases[k] if a['component']=='atomic_helium') for k in ['common30','missing_zero','missing_annular']}
    deps=[Path(__file__),P/'PREFLIGHT.md',contract]+[R/'scripts'/f for f in ['mond_atlas_hi_refinement.py','mond_atlas_hi_column_centroid.py','mond_atlas_source_resolution.py']]+[R/'work/gravity-first-principles/mond-atlas-generic-source-001/run-002/summary.json']
    bindings={p.relative_to(R).as_posix():sha(p) for p in deps}
    for a in selected.values(): assert sha(R/a['path'])==a['sha256'];bindings[a['path']]=a['sha256']
    # Only the native header is read, never source-region spectra.
    native=R/'work/private/things-observable-12gal-003/NGC_2976_NA_CUBE_THINGS.FITS'
    from astropy.io import fits
    import hashlib
    header_hash=hashlib.sha256(fits.getheader(native).tostring().encode()).hexdigest()
    save(out/'bindings.json',bindings)
    project,g=projector();points=np.array([[0,0,0],[1,2,.3],[-2,1,-.5]],float);perr=float(np.max(abs(project(*points.T)-project(*points.T,manual=True))));assert perr<1e-6
    vertical={};verr=[]
    for order in [24,48]:
        z,w=roots_laguerre(order);zz=np.r_[-z[::-1],z]*HEIGHT;ww=np.r_[w[::-1],w]*.5
        vertical[f'z{order}_kpc']=zz;vertical[f'z{order}_weight']=ww
        err=max(abs(ww.sum()-1),abs(ww@abs(zz)-HEIGHT),abs(ww@(zz**2)-2*HEIGHT**2));verr.append(float(err));assert err<1e-10
    save(out/'controls.json',dict(native_header_sha256=header_hash,native_projection_error_pixel=perr,vertical_moment_errors=verr,geometry=g))
    records=[];assets=[]
    for case,a in selected.items():
        with np.load(R/a['path']) as z:axis=z['latent_axis'];S=z['intrinsic_effective_surface']/HE
        assert np.isfinite(S).all() and (S>=0).all()
        total=S.sum()*(axis[1]-axis[0])**2*1e6
        for step in [.0625,.03125]:
            d,row=build(S,axis,step);row.update(case=case,source_path=a['path'],source_sha256=a['sha256'],mass_relative_error=float(abs(d['mass_HI_msun'].sum()/total-1)))
            assert row['all_positive_source_locations'] and row['all_centroids_inside_cell'] and row['moment_reference_max_error_kpc']<1e-10 and row['independent_mass_relative_error']<1e-8 and row['mass_relative_error']<1e-10
            rad=d['radius_kpc'];flux=d['flux_jy_km_s'];row['flux_outside_R6_jy_km_s']=float(flux[rad>6].sum());row['flux_outside_force_table_jy_km_s']=float(flux[(rad<.05)|(rad>6.025)].sum());row['fraction_outside_R6']=row['flux_outside_R6_jy_km_s']/float(flux.sum())
            row['stream_checks']=[];indices=np.unique(np.linspace(0,len(rad)-1,128,dtype=int));sample={k:v[indices] for k,v in d.items()}
            for order in [24,48]:
                full_error=float(np.max(abs(flux*vertical[f'z{order}_weight'].sum()-flux))/flux.max());assert full_error<1e-10
                summed=0;nodes=0
                for b in iter_native_batches(sample,order):assert np.isfinite(b['xy_pixel']).all();summed+=float(b['flux_jy_km_s'].sum());nodes+=len(b['planar_index'])
                stream_error=abs(summed/sample['flux_jy_km_s'].sum()-1);assert stream_error<1e-10
                row['stream_checks'].append(dict(order_per_side=order,sampled_planar_nodes=len(indices),native_nodes=nodes,flux_relative_error=float(stream_error),full_factorized_flux_error=full_error))
            path=priv/f'{case}-HI-planar-{step}.npz';np.savez_compressed(path,**d,**vertical);assets.append(dict(case=case,step_kpc=step,path=path.relative_to(R).as_posix(),sha256=sha(path),bytes=path.stat().st_size));records.append(row);print(case,step,row['planar_nodes'],row['fraction_outside_R6'],flush=True)
    assert all(sha(R/k)==v for k,v in bindings.items())
    save(out/'summary.json',dict(status='ALTERNATIVE_HI_PLANAR_AND_VERTICAL_CONSERVATION_PASS',cases=records,assets=assets,helium_factor=HE,distance_mpc=D,HI_mass_flux_coefficient=COEF,height_kpc=HEIGHT,seconds=time.monotonic()-start,observed_source_spectra_read=0,spectral_convergence_claimed=False,tail_aperture_bound_claimed=False,iterator='mond_atlas_hi_refinement.iter_native_batches'))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
