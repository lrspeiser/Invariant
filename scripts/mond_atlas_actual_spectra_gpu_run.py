"""Frozen actual-source spectrum refinement, no observational arrays."""
from mond_atlas_actual_spectra_build import save,digest
from mond_atlas_actual_spectra import *
import json,csv,time,itertools
from mond_atlas_aperture_injection_gpu import GPUCallback

def run():
    started=time.monotonic();out=PACKAGE/'run002';out.mkdir(exist_ok=False);private=PRIVATE/'run002';private.mkdir(parents=True,exist_ok=True)
    caches=json.loads((PACKAGE/'cache001/summary.json').read_text(encoding='utf-8'));assets={a['label']:a for a in caches['assets']}
    deps=[Path(__file__),ROOT/'scripts/mond_atlas_aperture_injection_gpu.py',ROOT/'work/gravity-first-principles/mond-atlas-aperture-injection-001/gpu001/receipt.json',PACKAGE/'GPU_ADDENDUM.md',ROOT/'scripts/mond_atlas_actual_spectra.py',PACKAGE/'PREFLIGHT.md',PACKAGE/'CORNER_REFINEMENT_ADDENDUM.md',PACKAGE/'cache001/summary.json']+[ROOT/a['cache'] for a in caches['assets']];save(out/'bindings.json',{str(p.relative_to(ROOT)):digest(p) for p in deps});spectra={};cases=[];statuses=[];corners=[];gpu_checks=[]
    try:
        for height,gravity,pressure,branch,spin in itertools.product(['h0p1','h0p4'],['newton','newton_plus_log'],[5.,10.,15.],['boxcar_independent','boxcar_hanning_full','boxcar_hanning_decimated'],[-1,1]):
            label=f'{height}_{gravity}_p{int(pressure)}_{branch}_s{spin}'
            nuisance=[(0.,3.),(0.,10.)]
            if height=='h0p1' and pressure==10:nuisance += list(itertools.product([-30.,30.],[3.,20.]))
            for name,asset in assets.items():
                assert digest(ROOT/asset['cache'])==asset['sha256'];model=ActualSpectrumModel(ROOT/asset['cache'],height,gravity,pressure,spin,branch)
                gpu=GPUCallback(model)
                if name=='p0p03125_z48':
                    err=float(np.max(abs(model.predict([0,10,1])-gpu.predict([0,10,1]))));gpu_checks.append(dict(case=label,max_absolute_mjy_beam=err));assert err<1e-10
                status=dict(case=label,source=name,invalid_groups=model.invalid_count,global_steady_valid=model.global_steady_valid,min_supported_rotation_squared=float(np.nanmin(model.rotation_squared)),max_unknown_envelope_at_multiplier2_mjy_beam=float(model.unknown_envelope([0,10,2]).max()));statuses.append(status)
                for systemic,sigma in nuisance:
                    tag=f'{label}_sys{int(systemic)}_sig{int(sigma)}';prediction=gpu.predict([systemic,sigma,1]);assert prediction.shape==(15,42) and np.isfinite(prediction).all();spectra[f'{name}__{tag}']=prediction
                    if name=='p0p03125_z48':cases.append(dict(tag=tag,height=height,gravity=gravity,pressure_reference=pressure,branch=branch,spin=spin,systemic=systemic,sigma=sigma))
                # All fixedmodel callbacks have finite bounded corner outputs.
                for systemic,sigma in itertools.product([-30.,30.],[3.,20.]):
                    q=gpu.predict([systemic,sigma,1]);assert np.isfinite(q).all()
            print(label,'finished',flush=True)
        save(out/'gpu-checks.json',gpu_checks)
        comparisons=[];velocity=load_instrument()['velocity_km_s'];pairs=[('planar','p0p0625_z24','p0p03125_z24'),('vertical','p0p0625_z24','p0p0625_z48'),('joint','p0p0625_z24','p0p03125_z48'),('fine_planar_vertical','p0p03125_z24','p0p03125_z48'),('fine_vertical_planar','p0p0625_z48','p0p03125_z48')]
        for case in cases:
            for comparison,lo,hi in pairs:
                a=spectra[f"{lo}__{case['tag']}"];b=spectra[f"{hi}__{case['tag']}"];den=np.sum(abs(b),axis=1);ca=np.sum(a*velocity,axis=1)/a.sum(axis=1);cb=np.sum(b*velocity,axis=1)/b.sum(axis=1)
                for ap in range(15):
                    L1=float(np.sum(abs(a[ap]-b[ap]))/den[ap]);delta=float(abs(ca[ap]-cb[ap]));comparisons.append(dict(case=case['tag'],comparison=comparison,aperture=ap,profile_L1=L1,centroid_km_s=delta,reference_L1_mjy_beam=float(den[ap]),passed=bool(L1<.01 and delta<.5)))
        for filename,data in [('cases.csv',cases),('invalid-status.csv',statuses),('refinement.csv',comparisons)]:
            with (out/filename).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        packet=private/'spectra.npz';np.savez_compressed(packet,**spectra)
        metrics={name:dict(comparisons=sum(r['comparison']==name for r in comparisons),failed=sum(r['comparison']==name and not r['passed'] for r in comparisons),max_L1=max(r['profile_L1'] for r in comparisons if r['comparison']==name),max_centroid_km_s=max(r['centroid_km_s'] for r in comparisons if r['comparison']==name)) for name,_,_ in pairs}
        summary=dict(status='ACTUAL_CONDITIONAL_SOURCE_SPECTRA_REFINED',physical_cases=72,nuisance_cases=len(cases),source_quadratures=4,profiles=len(spectra)*15,metrics=metrics,all_refinement_gates_pass=all(r['passed'] for r in comparisons),invalid_global_steady_cases=sum(not r['global_steady_valid'] for r in statuses),max_unknown_envelope_multiplier2_mjy_beam=max(r['max_unknown_envelope_at_multiplier2_mjy_beam'] for r in statuses),private_packet=str(packet.relative_to(ROOT)),private_sha256=digest(packet),observed_source_spectra_read=0,observed_gravity_scores=0,seconds=time.monotonic()-started);save(out/'summary.json',summary);print(json.dumps(summary,indent=2))
    except Exception as exc:save(out/'failure.json',dict(error=repr(exc)));raise
if __name__=='__main__':run()
