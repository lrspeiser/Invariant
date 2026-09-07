"""Sampled pixel reference tests, grouping and fixed source refinement."""
from mond_atlas_native_emission_sampled import *
from mond_atlas_native_emission import aperture_weights as integrated_weights,_beam_geometry
from mond_atlas_native_emission_controls import synthetic,save
from pathlib import Path
import csv,json,hashlib,time
from scipy.integrate import quad

def controls(inst):
    xy=np.array([[468.,444.],[470.37,446.28]]);W=aperture_weights(xy,inst);C=inst['beam_covariance_yx_pixel2'];inv=np.linalg.inv(C);_,_,_,halo,norm=_beam_geometry(inst,6);reference=np.zeros_like(W)
    for a,ap in enumerate(inst['apertures']):
        for j,(x,y) in enumerate(xy):
            values=[]
            for yy in range(ap['y_start'],ap['y_stop']):
                for xx in range(ap['x_start'],ap['x_stop']):
                    d=np.array([yy-y,xx-x]);values.append(np.exp(-d@inv@d/2)/norm if max(abs(d))<=halo else 0)
            reference[a,j]=sum(values)/len(values)
    giant=[dict(x_start=400,x_stop=540,y_start=380,y_stop=520)];full=aperture_weights(xy,inst,apertures=giant);mass=full*140**2/inst['beam_area_pixels'];assert np.max(abs(mass-1))<1e-8
    ap=inst['apertures'][0];tiles=[dict(x_start=x,x_stop=x+6,y_start=y,y_stop=y+6) for x in [ap['x_start'],ap['x_start']+6] for y in [ap['y_start'],ap['y_start']+6]];assert np.max(abs(aperture_weights(xy,inst,apertures=tiles).mean(axis=0)-W[0]))<1e-12
    # Retain distinct heights/positions and combine only equal velocity groups.
    xyz=np.vstack([xy,xy+[.3,.7],xy+[-.3,-.7]]);flux=np.array([.1,.2,.2,.1,.2,.2]);group=np.array([0,1,0,1,0,1]);v=np.array([0.,17.]);sigma=np.array([6.,8.]);grouped=grouped_aperture_flux(xyz,flux,group,inst);errors=[];uniterrors=[]
    for branch in BRANCHES:
        expanded=render_spectra(xyz,v[group],flux,sigma[group],inst,branch);compressed=render_grouped_spectra(grouped,v,sigma,inst,branch)
        errors.append(float(np.max(abs(expanded['spectra_mjy_beam']-compressed['spectra_mjy_beam']))))
        scaling=render_grouped_spectra(grouped*2,v,sigma,inst,branch);uniterrors.append(float(np.max(abs(scaling['spectra_mjy_beam']-2*compressed['spectra_mjy_beam']))))
    # Independently integrate exact Gaussian spectral bins then apply copied numeric A.
    mu=17.;sd=8.;dv=abs(inst['velocity_increment_km_s']);parent=[]
    for c in range(63):
        center=inst['first_parent_velocity_km_s']+c*inst['velocity_increment_km_s'];parent.append(quad(lambda u:np.exp(-.5*((u-mu)/sd)**2)/(np.sqrt(2*np.pi)*sd),center-dv/2,center+dv/2,epsabs=1e-13)[0]/dv)
    predicted=render_spectra(xy[:1],[mu],[.7],sd,inst);expected=1000*.7*reference[:,0,None]*(inst['continuum']@parent)[None,:];single=float(np.max(abs(predicted['spectra_mjy_beam']-expected)));assert single<1e-10 and max(errors)<1e-10
    return dict(status='PASS_SAMPLED_PIXEL_CONTROLS',independent_pixel_loop_error=float(np.max(abs(W-reference))),sampled_mass_closure=float(np.max(abs(mass-1))),grouped_spectrum_error=max(errors),flux_units_linearity_error=max(uniterrors),independent_emitter_spectrum_error=single,old_integrated_pixel_max_difference=float(np.max(abs(W-integrated_weights(xy,inst)))))

def run():
    start=time.monotonic();out=PACKAGE/'sampled-run001';out.mkdir(exist_ok=False);private=ROOT/'work/private/mond-atlas-native-emission-001/sampled-run001';private.mkdir(parents=True,exist_ok=True)
    try:
        inst=load_instrument();deps=[Path(__file__),ROOT/'scripts/mond_atlas_native_emission_sampled.py',ROOT/'scripts/mond_atlas_native_emission.py',ROOT/'scripts/mond_atlas_native_emission_controls.py',PACKAGE/'SAMPLED_PIXEL_PREFLIGHT.md',PACKAGE/'PREFLIGHT.md'];save(out/'bindings.json',{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in deps});save(out/'controls.json',controls(inst));comparisons=[];results={};conventions=[]
        for kind in ['rotation','warp','streaming','asymmetric']:
            prior={}
            for nr,na in [(128,256),(256,512),(512,1024)]:
                xy,v,flux=synthetic(kind,nr,na);W=aperture_weights(xy,inst)
                if nr==128:
                    old=integrated_weights(xy,inst);wide=aperture_weights(xy,inst,7);conventions.append(dict(kind=kind,sampled_integrated_max_abs=float(np.max(abs(W-old))),sampled_integrated_flux_relative=float(np.max(abs((W-old)@flux))/np.max(W@flux)),halo_max_abs=float(np.max(abs(wide-W)))))
                for branch in BRANCHES:
                    rendered=render_spectra(xy,v,flux,6,inst,branch,weights=W);spec=rendered['spectra_mjy_beam'];results[f'{kind}_{nr}_{branch}']=spec
                    if branch in prior:
                        old=prior[branch];l1=np.sum(abs(spec-old),axis=1)/np.sum(abs(spec),axis=1);centroid=np.sum(spec*inst['velocity_km_s'],axis=1)/np.sum(spec,axis=1);centroidold=np.sum(old*inst['velocity_km_s'],axis=1)/np.sum(old,axis=1);delta=abs(centroid-centroidold)
                        for a in range(15):comparisons.append(dict(kind=kind,branch=branch,coarse_nr=nr//2,fine_nr=nr,aperture=a,profile_L1=float(l1[a]),centroid_km_s=float(delta[a]),passed=bool(l1[a]<.01 and delta[a]<.5)))
                    prior[branch]=spec
                assert abs(flux.sum()-1)<1e-8;print(kind,nr,'complete',flush=True)
        np.savez_compressed(private/'synthetic-spectra.npz',**results)
        with (out/'refinement.csv').open('w',newline='',encoding='utf-8') as f:writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
        final=[r for r in comparisons if r['fine_nr']==512];summary=dict(status='SAMPLED_PIXEL_EMITTER_ADAPTER_BENCHMARKED',final_gate_pass=all(r['passed'] for r in final),lower_failed_count=sum(not r['passed'] for r in comparisons if r['fine_nr']==256),final_failed_count=sum(not r['passed'] for r in final),max_final_profile_L1=max(r['profile_L1'] for r in final),max_final_centroid_km_s=max(r['centroid_km_s'] for r in final),convention_checks=conventions,private_spectra= str((private/'synthetic-spectra.npz').relative_to(ROOT)),private_sha256=hashlib.sha256((private/'synthetic-spectra.npz').read_bytes()).hexdigest(),observed_source_spectra_read=0,actual_source_quadrature_admitted=False,elapsed_seconds=time.monotonic()-start);save(out/'summary.json',summary);print(json.dumps(summary,indent=2))
    except Exception as exc:save(out/'failure.json',dict(error=repr(exc)));raise
if __name__=='__main__':run()
