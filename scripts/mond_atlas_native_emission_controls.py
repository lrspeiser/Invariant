"""Source-free admission controls and frozen supplied-emitter refinement."""
from mond_atlas_native_emission import *
import hashlib,time
from scipy.integrate import quad

def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def synthetic(kind,nr,na):
    radius=(np.arange(nr)+.5)*160/nr;phi=(np.arange(na)+.5)*2*np.pi/na;r,p=np.meshgrid(radius,phi,indexing='ij');incl=np.deg2rad(60+(10*r/(r+70) if kind=='warp' else 0));pa=np.deg2rad(20*r/(r+70)) if kind=='warp' else np.zeros_like(r);xx=r*np.cos(p);yy=r*np.sin(p)*np.cos(incl)
    x=511+xx*np.cos(pa)-yy*np.sin(pa);y=512+xx*np.sin(pa)+yy*np.cos(pa);v=60*np.tanh(r/70)*np.sin(incl)*np.cos(p)
    if kind=='streaming':v+=20*np.sin(incl)*np.sin(p)
    flux=r*np.exp(-r/70)*(1+.4*np.cos(p) if kind=='asymmetric' else 1);flux/=flux.sum()
    return np.column_stack((x.ravel(),y.ravel())),v.ravel(),flux.ravel()

def controls(inst):
    center=np.array([[468.,444.],[468.25,444.75]]);W=aperture_weights(center,inst);C=inst['beam_covariance_yx_pixel2'];sx=np.sqrt(C[1,1]);slope=C[0,1]/C[1,1];cs=np.sqrt(C[0,0]-C[0,1]**2/C[1,1]);halo=6*np.sqrt(np.linalg.eigvalsh(C).max())
    norm=quad(lambda x:np.exp(-x*x/(2*sx*sx))/(np.sqrt(2*np.pi)*sx)*(ndtr((halo-slope*x)/cs)-ndtr((-halo-slope*x)/cs)),-halo,halo,epsabs=1e-13,epsrel=1e-13)[0];errors=[]
    for j,ap in enumerate(inst['apertures']):
        for i,(x,y) in enumerate(center):
            lo=max(ap['x_start']-.5-x,-halo);hi=min(ap['x_stop']-.5-x,halo);yl=max(ap['y_start']-.5-y,-halo);yh=min(ap['y_stop']-.5-y,halo)
            probability=0 if lo>=hi or yl>=yh else quad(lambda u:np.exp(-u*u/(2*sx*sx))/(np.sqrt(2*np.pi)*sx)*(ndtr((yh-slope*u)/cs)-ndtr((yl-slope*u)/cs)),lo,hi,epsabs=1e-13,epsrel=1e-13)[0]/norm
            errors.append(abs(W[j,i]-probability*inst['beam_area_pixels']/144))
    giant=[dict(x_start=400,x_stop=540,y_start=380,y_stop=520)];full=aperture_weights(center,inst,48,apertures=giant);mass=full*140**2/inst['beam_area_pixels'];assert np.max(abs(mass-1))<1e-8
    ap=inst['apertures'][0];tiles=[dict(x_start=x,x_stop=x+6,y_start=y,y_stop=y+6) for x in [ap['x_start'],ap['x_start']+6] for y in [ap['y_start'],ap['y_start']+6]];tiled=aperture_weights(center,inst,apertures=tiles).mean(axis=0);assert np.max(abs(tiled-W[0]))<1e-10
    spectral=spectral_density(np.array([0.,17.]),np.array([3.,11.]),inst,'boxcar_independent');lineerrors=[]
    for n,(v,sigma) in enumerate([(0,3),(17,11)]):
        for c in [0,20,31,62]:
            vc=inst['first_parent_velocity_km_s']+c*inst['velocity_increment_km_s'];dv=abs(inst['velocity_increment_km_s']);expected=quad(lambda u:np.exp(-.5*((u-v)/sigma)**2)/(np.sqrt(2*np.pi)*sigma),vc-dv/2,vc+dv/2,epsabs=1e-13)[0]/dv;lineerrors.append(abs(expected-spectral['parent'][c,n]))
    outputs=[]
    for branch in BRANCHES:
        a=render_spectra(center,[0,17],[.3,.7],6,inst,branch);b=render_spectra(center,[0,17],[.6,1.4],6,inst,branch);zero=render_spectra(center,[0,17],[0,0],6,inst,branch);assert np.max(abs(b['spectra_mjy_beam']-2*a['spectra_mjy_beam']))<1e-12 and np.max(abs(zero['spectra_mjy_beam']))==0
        outputs.append(dict(branch=branch,spectral_window_fraction=a['source_weighted_pregrid_flux_fraction']))
    translated=dict(inst,apertures=[{**ap,'x_start':ap['x_start']+31,'x_stop':ap['x_stop']+31,'y_start':ap['y_start']-19,'y_stop':ap['y_stop']-19} for ap in inst['apertures']]);assert np.max(abs(aperture_weights(center+[31,-19],translated)-W))<1e-12
    assert max(errors)<1e-10 and max(lineerrors)<1e-12
    return dict(status='PASS',independent_beam_probability_error=max(errors),enclosing_flux_closure=float(np.max(abs(mass-1))),pixel_tiling_error=float(np.max(abs(tiled-W[0]))),line_CDF_quadrature_error=max(lineerrors),translation=True,zero_flux=True,flux_linearity=True,branches=outputs)

def run():
    started=time.monotonic();out=PACKAGE/'run001';out.mkdir(exist_ok=False)
    try:
        inst=load_instrument();deps=[Path(__file__),ROOT/'scripts/mond_atlas_native_emission.py',PACKAGE/'PREFLIGHT.md',PACKAGE/'REFINEMENT_ADDENDUM.md',ROOT/'work/gravity-first-principles/mond-atlas-native-spectral-001/NGC2976.json',ROOT/'work/gravity-first-principles/mond-atlas-observation-admission-001/first-score-protocol.json',ROOT/'work/gravity-first-principles/mond-atlas-observation-admission-001/run004/frozen-geometric-apertures.csv',ROOT/'scripts/mond_atlas_native_selection.py',ROOT/'scripts/mond_atlas_native_spectral.py',ROOT/'docs/OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md']
        save(out/'bindings.json',{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in deps});save(out/'instrument.json',{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in inst.items()});save(out/'controls.json',controls(inst));comparisons=[];results={};nodeclosures=[];beamchecks=[]
        for kind in ['rotation','warp','streaming','asymmetric']:
            prior={}
            for nr,na in [(32,64),(64,128),(128,256)]:
                xy,v,flux=synthetic(kind,nr,na);W=aperture_weights(xy,inst);nodeclosures.append(dict(kind=kind,nr=nr,nodes=len(flux),sum_flux=float(flux.sum())))
                if nr==64:
                    refined=aperture_weights(xy,inst,48);larger=aperture_weights(xy,inst,48,7);beamchecks.append(dict(kind=kind,quadrature_max_abs=float(np.max(abs(W-refined))),halo_max_abs=float(np.max(abs(larger-refined)))))
                for branch in BRANCHES:
                    rendered=render_spectra(xy,v,flux,6,inst,branch,weights=W);spec=rendered['spectra_mjy_beam'];positive=rendered['positive_stored_mjy_beam'];results[f'{kind}_{nr}_{branch}']=spec;results[f'{kind}_{nr}_{branch}_positive']=positive
                    if branch in prior:
                        old=prior[branch];l1=np.sum(abs(spec-old),axis=1)/np.sum(abs(spec),axis=1);centroid=np.sum(spec*inst['velocity_km_s'],axis=1)/np.sum(spec,axis=1);oldcentroid=np.sum(old*inst['velocity_km_s'],axis=1)/np.sum(old,axis=1);error=abs(centroid-oldcentroid)
                        for a in range(15):comparisons.append(dict(kind=kind,branch=branch,coarse_nr=nr//2,fine_nr=nr,aperture=a,profile_L1=float(l1[a]),centroid_km_s=float(error[a]),passed=bool(l1[a]<.01 and error[a]<.5)))
                    prior[branch]=spec
                print(kind,nr,'complete',flush=True)
        np.savez_compressed(out/'synthetic-spectra.npz',**results)
        with (out/'refinement.csv').open('w',newline='',encoding='utf-8') as f:writer=csv.DictWriter(f,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
        final=[r for r in comparisons if r['fine_nr']==128];summary=dict(status='SUPPLIED_EMITTER_ADAPTER_BENCHMARKED',final_gate_pass=all(r['passed'] for r in final),coarse_failed_count=sum(not r['passed'] for r in comparisons if r['fine_nr']==64),final_failed_count=sum(not r['passed'] for r in final),max_final_profile_L1=max(r['profile_L1'] for r in final),max_final_centroid_km_s=max(r['centroid_km_s'] for r in final),node_flux_closure=max(abs(r['sum_flux']-1) for r in nodeclosures),beamchecks=beamchecks,parent_channels=63,stored_channels=42,apertures=15,observed_source_spectra_read=0,actual_emitter_source_admitted=False,elapsed_seconds=time.monotonic()-started);save(out/'summary.json',summary);save(out/'node-closure.json',nodeclosures);print(json.dumps(summary,indent=2))
    except Exception as e:save(out/'failure.json',dict(error=repr(e)));raise
if __name__=='__main__':run()
