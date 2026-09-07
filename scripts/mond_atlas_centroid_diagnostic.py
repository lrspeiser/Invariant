"""Positive precontinuum centroid diagnostic; retains all historical failed gates."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.integrate import quad
from mond_atlas_actual_spectra import ActualSpectrumModel
from mond_atlas_native_selection import spectral_matrix
R=Path(__file__).resolve().parents[1];P=R/'work/gravity-first-principles/mond-atlas-centroid-diagnostic-001';OLD=R/'work/gravity-first-principles/mond-atlas-actual-spectra-001'
PAIRS={'planar':('p0p0625_z24','p0p03125_z24'),'vertical':('p0p0625_z24','p0p0625_z48'),'joint':('p0p0625_z24','p0p03125_z48'),'fine_planar_vertical':('p0p03125_z24','p0p03125_z48'),'fine_vertical_planar':('p0p0625_z48','p0p03125_z48')}
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
def matrix(n,branch):
    if branch=='boxcar_independent':return np.eye(n),np.arange(n,dtype=float),1.
    if branch=='boxcar_hanning_full':
        grid=np.arange(-1,n+1,dtype=float);H=np.array([np.where(abs(grid-i)<1.1,np.where(grid==i,.5,.25),0.) for i in range(n)])
        return H,grid,1.
    if branch=='boxcar_hanning_decimated':
        grid=np.arange(-1,2*n,dtype=float)/2;H=np.array([np.where(abs(grid-i)<.6,np.where(grid==i,.5,.25),0.) for i in range(n)])
        return H,grid,.5
    raise ValueError(branch)
def interval(low,high):return np.where(low>=0,ndtr(-low)-ndtr(-high),ndtr(high)-ndtr(low))
def centroid(s,v):return s@v/s.sum(axis=-1)
def controls():
    result={}
    for branch in ['boxcar_independent','boxcar_hanning_full','boxcar_hanning_decimated']:
        H,g,w=matrix(63,branch);reference,rg,rw=spectral_matrix(63,branch)
        assert np.array_equal(H,reference) and np.array_equal(g,rg) and w==rw and np.all(H>=0) and np.all(H.sum(axis=1)==1)
    error=0.
    for low,high in [(-7.,-5.),(-1.7,.8),(.2,3.),(5.,7.)]:
        direct=quad(lambda x:np.exp(-x*x/2)/np.sqrt(2*np.pi),low,high,epsabs=1e-14)[0]
        error=max(error,abs(float(interval(np.array(low),np.array(high)))-direct))
    assert error<1e-12;result['independent_scalar_gaussian_bin_error']=error
    v=np.array([-1.,0.,1.]);positive=np.array([.3,.4,.3]);assert centroid(positive,v)==0
    a=np.array([1.,-1.999,1.]);b=a+np.array([0,0,.001]);jump=float(centroid(b,v)-centroid(a,v))
    assert abs(jump-.5)<1e-10;result['signed_cancellation_example_0p001_perturbation_centroid_jump']=jump
    result['independent_H_matches_all_three_operators']=True
    return result

def generate(model,systemic,sigma):
    inst=model.instrument;H,grid,width=matrix(inst['parent_channels'],model.branch);dv=abs(inst['velocity_increment_km_s']);width*=dv
    centers=inst['first_parent_velocity_km_s']+grid*inst['velocity_increment_km_s'];active=np.any(model.grouped_flux>0,axis=0);flux=model.grouped_flux[:,active];mu=model.velocity[active]+systemic
    pre=np.zeros((15,len(grid)))
    for start in range(0,len(mu),4096):
        low=(centers[:,None]-width/2-mu[None,start:start+4096])/sigma;high=low+width/sigma
        bins=interval(low,high)/width;pre+=1000*flux[:,start:start+4096]@bins.T
    parent=pre@H.T;positive=parent[:,inst['stored_indices']];signed=parent@inst['continuum'].T;total=flux.sum(axis=1)*1000
    return dict(positive=positive,signed=signed,velocity=inst['velocity_km_s'],capture=positive.sum(axis=1)*dv/total,parent_capture=parent.sum(axis=1)*dv/total,invalid_groups=model.invalid_count,unknown_max=model.unknown_envelope([systemic,sigma,1]).max(axis=1))

def run():
    start=time.monotonic();out=P/'run002';out.mkdir(exist_ok=False);save(out/'manufactured-controls.json',controls())
    summary=read(OLD/'run002/summary.json');assets=read(OLD/'cache001/summary.json')['assets'];failed=read(OLD/'failed-centroid-diagnostic.json')['failures'];assert len(failed)==42
    with (OLD/'run002/cases.csv').open(newline='') as f:cases={r['tag']:r for r in csv.DictReader(f)}
    tags=sorted({r['case'] for r in failed});packet=R/summary['private_packet'];assert digest(packet)==summary['private_sha256']
    deps=[Path(__file__),P/'PREFLIGHT.md',OLD/'run002/summary.json',OLD/'run002/cases.csv',OLD/'failed-centroid-diagnostic.json',OLD/'cache001/summary.json',packet,R/'scripts/mond_atlas_actual_spectra.py',R/'scripts/mond_atlas_native_selection.py',R/'scripts/mond_atlas_native_emission.py',R/'scripts/mond_atlas_native_emission_sampled.py',R/'work/gravity-first-principles/mond-atlas-native-spectral-001/NGC2976.json']+[R/a['cache'] for a in assets]
    save(out/'bindings.json',{str(p.relative_to(R)):digest(p) for p in deps})
    data={};diagnostics=[];maxreplay=0.;mincapture=1.;maxcapture=0.
    with np.load(packet) as old:
        for tag in tags:
            case=cases[tag]
            for asset in assets:
                path=R/asset['cache'];assert digest(path)==asset['sha256']
                model=ActualSpectrumModel(path,height=case['height'],model=case['gravity'],pressure_reference=float(case['pressure_reference']),spin=int(case['spin']),branch=case['branch'])
                d=generate(model,float(case['systemic']),float(case['sigma']));ref=old[asset['label']+'__'+tag];error=float(np.max(abs(ref-d['signed'])));maxreplay=max(maxreplay,error);assert error<1e-10
                positive=d['positive'];cs=centroid(d['signed'],d['velocity']);cp=centroid(positive,d['velocity'])
                assert np.all(positive>=0) and np.all(positive.sum(axis=1)>0) and np.all(np.isfinite(cs)) and np.all(cp>=min(d['velocity'])) and np.all(cp<=max(d['velocity']))
                assert np.all(d['capture']>=0) and np.all(d['capture']<=1+1e-12) and np.all(d['parent_capture']<=1+1e-12)
                d.update(signed_centroid=cs,positive_centroid=cp);data[(tag,asset['label'])]=d
                for ap in range(15):
                    signed=d['signed'][ap];pos=positive[ap];signedtotal=float(signed.sum());kappa=float(abs(signed).sum()/abs(signedtotal))
                    diagnostics.append(dict(case=tag,cache=asset['label'],aperture=ap,signed_centroid_km_s=float(cs[ap]),positive_centroid_km_s=float(cp[ap]),signed_to_positive_integral=signedtotal/float(pos.sum()),signed_cancellation_kappa=kappa,stored_window_capture=float(d['capture'][ap]),parent_window_capture=float(d['parent_capture'][ap]),signed_outside_window=bool(cs[ap]<min(d['velocity']) or cs[ap]>max(d['velocity'])),unknown_emitter_max_mjy_beam=float(d['unknown_max'][ap]),invalid_groups=d['invalid_groups']))
            print(tag,'four caches complete',flush=True)
    comparisons=[];identityerror=0.
    for original in failed:
        tag=original['case'];lo,hi=PAIRS[original['comparison']];ap=int(original['aperture']);a=data[(tag,lo)];b=data[(tag,hi)];sa=a['signed'][ap];sb=b['signed'][ap];v=a['velocity'];cb=b['signed_centroid'][ap]
        direct=float(a['signed_centroid'][ap]-cb);identity=float((sa-sb)@(v-cb)/sa.sum());identityerror=max(identityerror,abs(direct-identity));bound=float(abs((sa-sb)*(v-cb)).sum()/abs(sa.sum()))
        positive_delta=float(abs(a['positive_centroid'][ap]-b['positive_centroid'][ap]));replayed_L1=float(abs(sa-sb).sum()/abs(sb).sum())
        assert abs(abs(direct)-float(original['centroid_km_s']))<1e-8 and abs(replayed_L1-float(original['profile_L1']))<1e-10
        comparisons.append(dict(**original,original_failure_retained=True,signed_replay_centroid_delta_km_s=abs(direct),positive_centroid_delta_km_s=positive_delta,positive_delta_below_0p5_diagnostic_only=positive_delta<.5,signed_difference_identity_km_s=identity,signed_difference_triangle_bound_km_s=bound,coarse_cancellation_kappa=float(abs(sa).sum()/abs(sa.sum())),fine_cancellation_kappa=float(abs(sb).sum()/abs(sb.sum())),coarse_capture=float(a['capture'][ap]),fine_capture=float(b['capture'][ap]),coarse_positive_centroid_km_s=float(a['positive_centroid'][ap]),fine_positive_centroid_km_s=float(b['positive_centroid'][ap]),coarse_signed_centroid_km_s=float(a['signed_centroid'][ap]),fine_signed_centroid_km_s=float(cb)))
    assert identityerror<1e-8
    for filename,rows in [('all-four-cache-diagnostics.csv',diagnostics),('retained-failure-comparisons.csv',comparisons)]:
        with (out/filename).open('x',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(status='POST_HOC_SOURCE_ONLY_DIAGNOSTIC_COMPLETE_OLD_FAILURES_RETAINED',old_failed_comparisons=len(comparisons),distinct_failed_cases=len(tags),recomputed_profiles=len(diagnostics),positive_comparisons_below_0p5=sum(x['positive_delta_below_0p5_diagnostic_only'] for x in comparisons),max_positive_centroid_difference_km_s=max(x['positive_centroid_delta_km_s'] for x in comparisons),max_old_signed_centroid_difference_km_s=max(float(x['centroid_km_s']) for x in comparisons),failed_row_cancellation_kappa_range=[min(min(x['coarse_cancellation_kappa'],x['fine_cancellation_kappa']) for x in comparisons),max(max(x['coarse_cancellation_kappa'],x['fine_cancellation_kappa']) for x in comparisons)],failed_row_capture_range=[min(min(x['coarse_capture'],x['fine_capture']) for x in comparisons),max(max(x['coarse_capture'],x['fine_capture']) for x in comparisons)],max_saved_signed_spectrum_replay_error_mjy_beam=maxreplay,max_centroid_identity_error_km_s=identityerror,observed_source_spectra_read=0,old_gates_changed=False,seconds=time.monotonic()-start)
    save(out/'summary.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':run()
