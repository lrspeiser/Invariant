from mond_atlas_alternative_spectra_build import P,D,ROOT,save,digest
from mond_atlas_actual_spectra import ActualSpectrumModel,load_instrument
from mond_atlas_aperture_injection_gpu import GPUCallback
from pathlib import Path
import numpy as np,json,csv,itertools,time

def run():
 out=P/'run001';out.mkdir(exist_ok=False);private=D/'run001';private.mkdir(exist_ok=False);start=time.monotonic();assets=json.loads((P/'cache001/summary.json').read_text())['assets'];baseP=ROOT/'work/gravity-first-principles/mond-atlas-actual-spectra-001/run002';baseS=json.loads((baseP/'summary.json').read_text());bp=ROOT/baseS['private_packet'];assert digest(bp)==baseS['private_sha256'];baseline=np.load(bp)
 deps=[Path(__file__),P/'PREFLIGHT.md',P/'cache001/summary.json',ROOT/'scripts/mond_atlas_actual_spectra.py',ROOT/'scripts/mond_atlas_aperture_injection_gpu.py',bp]+[ROOT/a['cache'] for a in assets];save(out/'bindings.json',{str(p.relative_to(ROOT)):digest(p) for p in deps});spectra={};rows=[];statuses=[];changes=[];gpuchecks=[];v=load_instrument()['velocity_km_s'];pairs=[('planar','p0p0625_z24','p0p03125_z24'),('vertical','p0p0625_z24','p0p0625_z48'),('joint','p0p0625_z24','p0p03125_z48'),('fine_planar_vertical','p0p03125_z24','p0p03125_z48'),('fine_vertical_planar','p0p0625_z48','p0p03125_z48')]
 for case,gravity,pressure,spin,branch in itertools.product(['common30','common30_stars_h0.4','missing_zero','missing_annular'],['newton','newton_plus_log'],[5,10,15],[-1,1],['boxcar_independent','boxcar_hanning_full','boxcar_hanning_decimated']):
  label=f'{case}__{gravity}_p{pressure}_{branch}_s{spin}';pred={}
  for asset in [a for a in assets if a['case']==case]:
   assert digest(ROOT/asset['cache'])==asset['sha256'];model=ActualSpectrumModel(ROOT/asset['cache'],model=gravity,pressure_reference=pressure,spin=spin,branch=branch);gpu=GPUCallback(model);statuses.append(dict(case=label,source=asset['label'],invalid_groups=model.invalid_count,global_steady_valid=model.global_steady_valid,max_unknown_envelope_multiplier2=float(model.unknown_envelope([0,10,2]).max())))
   for sigma in [3,10]:
    y=gpu.predict([0,sigma,1]);assert np.isfinite(y).all();pred[asset['label'],sigma]=y;spectra[label+f'_sig{sigma}__'+asset['label']]=y
   if asset['label']=='p0p03125_z48' and pressure==10 and spin==1:
    err=float(abs(model.predict([0,10,1])-pred[asset['label'],10]).max());assert err<1e-10;gpuchecks.append(dict(case=label,max_absolute_error=err))
  for sigma in [3,10]:
   for comparison,lo,hi in pairs:
    a=pred[lo,sigma];b=pred[hi,sigma];l=abs(a-b).sum(1)/abs(b).sum(1);dc=abs((a@v)/a.sum(1)-(b@v)/b.sum(1))
    for i in range(15):rows.append(dict(case=label,sigma=sigma,comparison=comparison,aperture=i,profile_L1=float(l[i]),centroid_km_s=float(dc[i]),passed=bool(l[i]<.01 and dc[i]<.5)))
   height='h0p4' if case=='common30_stars_h0.4' else 'h0p1';key=f'p0p03125_z48__{height}_{gravity}_p{pressure}_{branch}_s{spin}_sys0_sig{sigma}';b=baseline[key];a=pred['p0p03125_z48',sigma]
   for i in range(15):changes.append(dict(case=label,sigma=sigma,aperture=i,relative_L1_vs_baseline=float(abs(a[i]-b[i]).sum()/abs(b[i]).sum()),peak_absolute_change_mjy_beam=float(abs(a[i]-b[i]).max())))
  print(label,'complete',flush=True)
 for filename,data in [('refinement.csv',rows),('unknown-status.csv',statuses),('source-differences.csv',changes)]:
  with (out/filename).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 packet=private/'spectra.npz';np.savez_compressed(packet,**spectra);save(out/'gpu-checks.json',gpuchecks);summary=dict(status='ALTERNATIVE_CONDITIONAL_SOURCE_SPECTRA_COMPLETE',physical_cases=144,profiles=len(spectra)*15,comparisons=len(rows),failed=sum(not r['passed'] for r in rows),max_L1=max(r['profile_L1'] for r in rows),max_centroid_km_s=max(r['centroid_km_s'] for r in rows),max_unknown_envelope_multiplier2=max(r['max_unknown_envelope_multiplier2'] for r in statuses),source_change_by_case={case:dict(median_L1=float(np.median([r['relative_L1_vs_baseline'] for r in changes if r['case'].startswith(case+'__')])),max_L1=max(r['relative_L1_vs_baseline'] for r in changes if r['case'].startswith(case+'__'))) for case in ['common30','common30_stars_h0.4','missing_zero','missing_annular']},private_packet=str(packet.relative_to(ROOT)),private_sha256=digest(packet),observed_source_spectra_read=0,seconds=time.monotonic()-start);save(out/'summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
