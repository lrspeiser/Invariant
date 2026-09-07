from mond_atlas_alternative_spectra_build import P,D,ROOT,save,digest
from mond_atlas_actual_spectra import load_instrument
from pathlib import Path
import json,csv,numpy as np

def run():
 out=P/'independent-review';out.mkdir(exist_ok=False);r=P/'run001';s=json.loads((r/'summary.json').read_text());packet=ROOT/s['private_packet'];assert digest(packet)==s['private_sha256'];z=np.load(packet);v=load_instrument()['velocity_km_s'];pairs={'planar':('p0p0625_z24','p0p03125_z24'),'vertical':('p0p0625_z24','p0p0625_z48'),'joint':('p0p0625_z24','p0p03125_z48'),'fine_planar_vertical':('p0p03125_z24','p0p03125_z48'),'fine_vertical_planar':('p0p0625_z48','p0p03125_z48')};rows=list(csv.DictReader((r/'refinement.csv').open()));error=0
 for row in rows:
  lo,hi=pairs[row['comparison']];tag=row['case']+'_sig'+row['sigma']+'__';a=z[tag+lo][int(row['aperture'])];b=z[tag+hi][int(row['aperture'])];l=sum(abs(float(x-y)) for x,y in zip(a,b))/sum(abs(float(x)) for x in b);dc=abs(sum(float(x*y) for x,y in zip(a,v))/sum(map(float,a))-sum(float(x*y) for x,y in zip(b,v))/sum(map(float,b)));assert (l<.01 and dc<.5)==(row['passed']=='True');error=max(error,abs(l-float(row['profile_L1'])),abs(dc-float(row['centroid_km_s'])))
 baseP=ROOT/'work/gravity-first-principles/mond-atlas-actual-spectra-001/run002';bs=json.loads((baseP/'summary.json').read_text());bpath=ROOT/bs['private_packet'];assert digest(bpath)==bs['private_sha256'];base=np.load(bpath);changes=list(csv.DictReader((r/'source-differences.csv').open()));ce=0
 for row in changes:
  case,tag=row['case'].split('__');height='h0p4' if case=='common30_stars_h0.4' else 'h0p1';a=z[row['case']+'_sig'+row['sigma']+'__p0p03125_z48'][int(row['aperture'])];b=base['p0p03125_z48__'+height+'_'+tag+'_sys0_sig'+row['sigma']][int(row['aperture'])];ce=max(ce,abs(sum(abs(float(x-y)) for x,y in zip(a,b))/sum(abs(float(x)) for x in b)-float(row['relative_L1_vs_baseline'])))
 save(out/'receipt.json',dict(status='SAVED_SPECTRA_SCALAR_REPLAY_PASS',refinement_rows=len(rows),refinement_max_error=error,source_difference_rows=len(changes),source_difference_max_error=ce,bindings={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),packet,bpath,r/'summary.json',r/'refinement.csv',r/'source-differences.csv']},scope='Separate scalar arithmetic on every stored profile. Does not independently reconstruct projected source weights.',observed_source_spectra_read=0));print('replay',len(rows),error,len(changes),ce)
if __name__=='__main__':run()
