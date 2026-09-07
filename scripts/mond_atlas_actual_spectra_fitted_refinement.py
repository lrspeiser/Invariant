from mond_atlas_actual_spectra import *
from mond_atlas_aperture_injection_gpu import GPUCallback
from mond_atlas_actual_spectra_build import save,digest
import csv,json

def run():
 out=PACKAGE/'fitted-injection-refinement';out.mkdir(exist_ok=False)
 save(out/'preflight.json',dict(scope='Evaluate all51 already fitted manufactured injection nuisance vectors on four frozen source caches. No refit, no observed arrays. Identical five pairs, all15apertures, L1<.01 and signed centroid<.5km/s. Primary h0p1/Newton/p10/spin+ only. Keep all failures.',observed_source_spectra_read=0))
 inp=ROOT/'work/gravity-first-principles/mond-atlas-aperture-injection-001/fit002';files=sorted(inp.glob('boxcar*.json'));assert len(files)==51
 assets=json.loads((PACKAGE/'cache001/summary.json').read_text())['assets'];deps=[Path(__file__),inp/'summary.json',ROOT/'scripts/mond_atlas_aperture_injection_gpu.py',ROOT/'scripts/mond_atlas_actual_spectra.py']+files+[ROOT/a['cache'] for a in assets];save(out/'bindings.json',{str(p.relative_to(ROOT)):digest(p) for p in deps})
 models={(branch,a['label']):GPUCallback(ActualSpectrumModel(ROOT/a['cache'],branch=branch)) for branch in ['boxcar_independent','boxcar_hanning_full','boxcar_hanning_decimated'] for a in assets};pairs=[('planar','p0p0625_z24','p0p03125_z24'),('vertical','p0p0625_z24','p0p0625_z48'),('joint','p0p0625_z24','p0p03125_z48'),('fine_planar_vertical','p0p03125_z24','p0p03125_z48'),('fine_vertical_planar','p0p0625_z48','p0p03125_z48')];rows=[];saved={};v=load_instrument()['velocity_km_s']
 for path in files:
  item=json.loads(path.read_text());pars=item['fit']['parameters'];pred={a['label']:models[item['branch'],a['label']].predict(pars) for a in assets}
  for name,x in pred.items():saved[path.stem+'__'+name]=x
  for comparison,lo,hi in pairs:
   a=pred[lo];b=pred[hi];L=np.sum(abs(a-b),axis=1)/np.sum(abs(b),axis=1);d=abs((a@v)/a.sum(1)-(b@v)/b.sum(1))
   for ap in range(15):rows.append(dict(trial=path.stem,comparison=comparison,aperture=ap,profile_L1=float(L[ap]),centroid_km_s=float(d[ap]),passed=bool(L[ap]<.01 and d[ap]<.5)))
 with (out/'refinement.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 private=PRIVATE/'fitted-injection-refinement';private.mkdir(exist_ok=False);packet=private/'spectra.npz';np.savez_compressed(packet,**saved)
 # Independent arithmetic replay uses channelwise scalar sums from persisted arrays.
 z=np.load(packet);err=0
 for row in rows:
  lo,hi=next((lo,hi) for name,lo,hi in pairs if name==row['comparison']);a=z[row['trial']+'__'+lo][row['aperture']];b=z[row['trial']+'__'+hi][row['aperture']];l=sum(abs(float(x-y)) for x,y in zip(a,b))/sum(abs(float(x)) for x in b);ca=sum(float(x*y) for x,y in zip(a,v))/sum(map(float,a));cb=sum(float(x*y) for x,y in zip(b,v))/sum(map(float,b));err=max(err,abs(l-row['profile_L1']),abs(abs(ca-cb)-row['centroid_km_s']));assert (l<.01 and abs(ca-cb)<.5)==row['passed']
 summary=dict(status='FITTED_INJECTION_SOURCE_REFINEMENT_COMPLETE',trials=51,comparisons=len(rows),failed=sum(not r['passed'] for r in rows),max_L1=max(r['profile_L1'] for r in rows),max_centroid_km_s=max(r['centroid_km_s'] for r in rows),independent_scalar_replay_max_error=err,private_packet=str(packet.relative_to(ROOT)),private_sha256=digest(packet),observed_source_spectra_read=0,refits=0);save(out/'summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
