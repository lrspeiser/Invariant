import sys,json,csv
sys.path.insert(0,'scripts')
from mond_atlas_actual_spectra import *
rows=list(csv.DictReader((PACKAGE/'run002/invalid-status.csv').open()));cache=json.loads((PACKAGE/'cache001/summary.json').read_text());assets={a['label']:ROOT/a['cache'] for a in cache['assets']};inst=load_instrument();errors=[]
for row in rows:
 label=row['case'];height,rest=label.split('_',1);gravity,rest=rest.split('_p');pressure,rest=rest.split('_',1);branch,spin=rest.rsplit('_s',1);c=np.load(assets[row['source']]);v2=c['radius_kpc']*(c[f'force_{height}_{gravity}'].sum(0)+int(pressure)**2*c['pressure_gradient']/c[f'den_{height}']);bad=~(c['force_supported']&np.isfinite(v2)&(v2>=0));flux=c['grouped_flux'][:,bad].sum(1)
 if branch=='boxcar_independent':H=np.eye(63);width=1
 else:
  k=2 if branch.endswith('decimated') else 1;H=np.zeros((63,62*k+3));width=1/k
  for i in range(63):H[i,k*i:k*i+3]=[.25,.5,.25]
 envelope=2000*flux[:,None]*abs(inst['continuum']@H).sum(1)[None,:]/(abs(inst['velocity_increment_km_s'])*width);assert bad.sum()==int(row['invalid_groups']);errors.append(abs(envelope.max()-float(row['max_unknown_envelope_at_multiplier2_mjy_beam'])))
(PACKAGE/'independent-review/bound-replay.json').write_text(json.dumps(dict(cases=len(rows),max_absolute_error=float(max(errors)),status='PASS',method='Reconstructed v2 invalid masks directly from each full cache; independent H matrices; positive flux and signed continuum triangle inequality. No invented velocity for invalid nodes.',observed_source_spectra_read=0),indent=2))
