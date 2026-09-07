"""Prospective full-design positive-centroid check, preserving original failures."""
from mond_atlas_centroid_diagnostic import *
import cupy as cp
from cupyx.scipy.special import ndtr as gpu_ndtr

def gpu_generate(model,systemic,sigma):
    inst=model.instrument;H,grid,width=matrix(inst['parent_channels'],model.branch);dv=abs(inst['velocity_increment_km_s']);width*=dv
    centers=cp.asarray(inst['first_parent_velocity_km_s']+grid*inst['velocity_increment_km_s']);active=np.any(model.grouped_flux>0,axis=0);flux=cp.asarray(model.grouped_flux[:,active]);mu=cp.asarray(model.velocity[active]+systemic)
    low=(centers[:,None]-width/2-mu[None,:])/sigma;high=low+width/sigma
    probability=cp.where(low>=0,gpu_ndtr(-low)-gpu_ndtr(-high),gpu_ndtr(high)-gpu_ndtr(low))
    parent=cp.asnumpy((1000*flux@(probability/width).T)@cp.asarray(H).T);positive=parent[:,inst['stored_indices']];signed=parent@inst['continuum'].T;total=model.grouped_flux[:,active].sum(axis=1)*1000
    return dict(positive=positive,signed=signed,velocity=inst['velocity_km_s'],capture=positive.sum(axis=1)*dv/total,parent_capture=parent.sum(axis=1)*dv/total)

def run_all():
    started=time.monotonic();out=P/'all-cases001';out.mkdir(exist_ok=False)
    save(out/'manufactured-controls.json',controls())
    assets=read(OLD/'cache001/summary.json')['assets'];oldsummary=read(OLD/'run002/summary.json');packet=R/oldsummary['private_packet'];assert digest(packet)==oldsummary['private_sha256']
    with (OLD/'run002/cases.csv').open(newline='') as f:cases=list(csv.DictReader(f))
    assert len(cases)==192
    deps=[Path(__file__),R/'scripts/mond_atlas_centroid_diagnostic.py',P/'PREFLIGHT.md',P/'ALL_CASES_PREFLIGHT.md',OLD/'run002/summary.json',OLD/'run002/cases.csv',OLD/'cache001/summary.json',packet,R/'scripts/mond_atlas_actual_spectra.py',R/'scripts/mond_atlas_native_emission.py',R/'scripts/mond_atlas_native_emission_sampled.py',R/'scripts/mond_atlas_native_selection.py']+[R/a['cache'] for a in assets]
    save(out/'bindings.json',{str(x.relative_to(R)):digest(x) for x in deps})
    checks=[]
    for branch in ['boxcar_independent','boxcar_hanning_full','boxcar_hanning_decimated']:
        a=assets[0];m=ActualSpectrumModel(R/a['cache'],branch=branch);cpu=generate(m,0,10);gpu=gpu_generate(m,0,10)
        error=max(float(abs(cpu[k]-gpu[k]).max()) for k in ['positive','signed']);assert error<1e-10;checks.append(dict(branch=branch,max_absolute_error_mjy_beam=error))
    save(out/'gpu-cpu-controls.json',checks)
    data={};moments=[];maxerror=0.
    with np.load(packet) as old:
        # Hold each physical source model for all its frozen nuisance cases.
        groups={}
        for c in cases:groups.setdefault((c['height'],c['gravity'],float(c['pressure_reference']),int(c['spin']),c['branch']),[]).append(c)
        for n,(key,group) in enumerate(groups.items()):
            height,gravity,pressure,spin,branch=key
            for a in assets:
                assert digest(R/a['cache'])==a['sha256'];m=ActualSpectrumModel(R/a['cache'],height=height,model=gravity,pressure_reference=pressure,spin=spin,branch=branch)
                for c in group:
                    tag=c['tag'];d=gpu_generate(m,float(c['systemic']),float(c['sigma']));ref=old[a['label']+'__'+tag];error=float(abs(ref-d['signed']).max());maxerror=max(maxerror,error);assert error<1e-10
                    pc=centroid(d['positive'],d['velocity']);sc=centroid(d['signed'],d['velocity']);assert np.isfinite(pc).all() and np.isfinite(sc).all() and np.all(d['positive']>=0) and np.all(pc>=min(d['velocity'])) and np.all(pc<=max(d['velocity'])) and np.all(d['capture']>=0) and np.all(d['capture']<=1+1e-12) and np.all(d['parent_capture']<=1+1e-12)
                    d.update(positive_centroid=pc,signed_centroid=sc);data[(tag,a['label'])]=d
                    for ap in range(15):moments.append(dict(case=tag,cache=a['label'],aperture=ap,positive_centroid_km_s=float(pc[ap]),signed_centroid_km_s=float(sc[ap]),stored_window_capture=float(d['capture'][ap]),parent_window_capture=float(d['parent_capture'][ap])))
            if n%12==0:print('physical groups',n+1,'of',len(groups),flush=True)
            assert time.monotonic()-started<600,'Declared10minute bound exceeded'
    results=[]
    for c in cases:
        for comparison,(lo,hi) in PAIRS.items():
            a=data[(c['tag'],lo)];b=data[(c['tag'],hi)]
            for ap in range(15):
                delta=float(abs(a['positive_centroid'][ap]-b['positive_centroid'][ap]));old_delta=float(abs(a['signed_centroid'][ap]-b['signed_centroid'][ap]));l1=float(abs(a['signed'][ap]-b['signed'][ap]).sum()/abs(b['signed'][ap]).sum())
                results.append(dict(case=c['tag'],comparison=comparison,aperture=ap,positive_centroid_delta_km_s=delta,positive_centroid_criterion_pass=delta<.5,old_signed_centroid_delta_km_s=old_delta,signed_profile_L1=l1,old_combined_gate_pass=bool(old_delta<.5 and l1<.01),new_diagnostic_combined_pass=bool(delta<.5 and l1<.01),coarse_capture=float(a['capture'][ap]),fine_capture=float(b['capture'][ap])))
    for name,rows in [('all-comparisons.csv',results),('all-moments.csv',moments)]:
        with (out/name).open('x',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(status='FULL_PREDECLARED_DESIGN_DIAGNOSTIC_COMPLETE_OLD_GATES_UNCHANGED',cases=len(cases),profiles=len(moments),comparisons=len(results),positive_centroid_failures=sum(not r['positive_centroid_criterion_pass'] for r in results),new_diagnostic_combined_failures=sum(not r['new_diagnostic_combined_pass'] for r in results),retained_old_combined_failures=sum(not r['old_combined_gate_pass'] for r in results),max_positive_centroid_delta_km_s=max(r['positive_centroid_delta_km_s'] for r in results),max_signed_profile_L1=max(r['signed_profile_L1'] for r in results),stored_capture_range=[min(r['stored_window_capture'] for r in moments),max(r['stored_window_capture'] for r in moments)],parent_capture_range=[min(r['parent_window_capture'] for r in moments),max(r['parent_window_capture'] for r in moments)],max_saved_signed_spectrum_replay_error_mjy_beam=maxerror,seconds=time.monotonic()-started,old_gate_waived=False,observed_source_spectra_read=0)
    save(out/'summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':run_all()
