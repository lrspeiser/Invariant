from pathlib import Path
import json,hashlib
import numpy as np
from mond_atlas_native_emission_sampled import aperture_weights,load_instrument
from mond_atlas_native_emission import aperture_weights as integrated_weights
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'work/gravity-first-principles/mond-atlas-hi-column-001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 path=ROOT/'work/private/mond-atlas-hi-column-001/centroid-projection001/f4-emission-0.0625.npz';out=P/'centroid-aperture-audit.json';assert not out.exists()
 instrument=load_instrument();rows=[]
 with np.load(path) as z:xy=z['xy_pixel'];flux=z['flux_jy_km_s'];r=z['radius_kpc'];outside=~z['force_table_supported']
 for i,ap in enumerate(instrument['apertures']):
  a=aperture_weights(xy,instrument,apertures=[ap])[0];b=integrated_weights(xy,instrument,apertures=[ap])[0];total=float(a@flux);alternative=float(b@flux);tail=float(a[outside]@flux[outside]);rows.append(dict(aperture=i,sampled_Jy_km_s_perbeam=total,integrated_Jy_km_s_perbeam=alternative,relative_integrated_minus_sampled=alternative/total-1 if total>0 else None,outside_force_weighted_Jy_km_s_perbeam=tail,outside_nonzero_nodes=int((a[outside]>0).sum()),positive_aperture_flux_tail_fraction=tail/total if total>0 else None))
 receipt=dict(status='ACTUAL_EMITTER_SPATIAL_RESPONSE_AUDIT',source_path=path.relative_to(ROOT).as_posix(),source_sha256=sha(path),source_radius_min=float(r.min()),source_radius_max=float(r.max()),rows=rows,weights_are_spatial_only_no_velocity_or_spectrum=True,observed_response_access=False,dependencies={f'scripts/{name}':sha(ROOT/'scripts'/name) for name in ['mond_atlas_native_emission_sampled.py','mond_atlas_native_emission.py','mond_atlas_hi_column_aperture_audit.py']})
 out.write_text(json.dumps(receipt,indent=2),encoding='utf8');print(json.dumps(dict(max_pixel_convention_fraction=max(abs(x['relative_integrated_minus_sampled']) for x in rows),max_unsupported_weighted_flux=max(x['outside_force_weighted_Jy_km_s_perbeam'] for x in rows),max_unsupported_fraction=max(x['positive_aperture_flux_tail_fraction'] for x in rows))))
if __name__=='__main__':main()
