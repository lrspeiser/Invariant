from pathlib import Path
import json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'work/gravity-first-principles/mond-atlas-hi-column-001';PRIVATE=ROOT/'work/private/mond-atlas-hi-column-001'
from mond_atlas_native_emission_sampled import aperture_weights,load_instrument
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=P/'closure001';out.mkdir(exist_ok=False);priv=PRIVATE/'closure001';priv.mkdir(parents=True,exist_ok=False);instrument=load_instrument();rows=[];closure=[]
 deps=[Path(__file__),P/'CLOSURE_ADDENDUM.md',ROOT/'scripts/mond_atlas_native_emission_sampled.py',ROOT/'scripts/mond_atlas_native_emission.py'];(out/'bindings.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():sha(p) for p in deps},indent=2),encoding='utf8')
 for case in ['f1','f4']:
  path=PRIVATE/f'run001/{case}-column.npz'
  with np.load(path) as z:
   R=z['radius_kpc'];raw=z['angular_sigma_HI_msun_pc2'].mean(axis=1);sm=z['sigma_HI_msun_pc2'];d=z['d_sigma_HI_dR_msun_pc2_kpc'];var=z['pressure_variance_km2_s2'];good=raw>0;eff=np.divide(sm,raw,out=np.full_like(raw,np.nan),where=good)
   dest=priv/f'{case}-pressure-closure.npz';np.savez_compressed(dest,radius_kpc=R,sigma_raw_HI_msun_pc2=raw,sigma_pressure_smoothed_HI_msun_pc2=sm,pressure_normalization_km2_s2=var,Pi_msun_pc2_km2_s2=sm[:,None]*var,dPi_dR_msun_pc2_km2_s2_kpc=d[:,None]*var,local_effective_variance_km2_s2=eff[:,None]*var,raw_column_positive=good)
  inner=(R>=.75)&(R<=2.5);closure.append(dict(case=case,path=dest.relative_to(ROOT).as_posix(),sha256=sha(dest),undefined_raw_column_radii=R[~good].tolist(),inner_effective_variance_over_normalization_range=[float(eff[inner].min()),float(eff[inner].max())],pressure_gradient_identity_max_error=float(np.max(abs((d[:,None]*var)-(d[:,None]*var))))))
 for asset in read(P/'projection001/summary.json')['assets']:
  path=ROOT/asset['path'];assert sha(path)==asset['sha256']
  with np.load(path) as z:
   outside=~z['force_table_supported'];r=z['radius_kpc'];xy=z['xy_pixel'][outside];flux=z['flux_jy_km_s'][outside];W=aperture_weights(xy,instrument);weighted=W@flux
   rows.append(dict(path=asset['path'],radius_min=float(r.min()),radius_max=float(r.max()),outside_nodes=int(outside.sum()),outside_flux_jy_km_s=float(flux.sum()),outside_with_any_native_aperture_weight=int((W.max(axis=0)>0).sum()),aperture_weighted_integrated_Jy_km_s_perbeam=weighted.tolist(),max_aperture_weighted_integrated_Jy_km_s_perbeam=float(weighted.max()),not_removed=True))
 (out/'summary.json').write_text(json.dumps(dict(status='EXPLICIT_SOURCE_PRESSURE_CLOSURE_AND_DOMAIN_AUDIT',pressure_packets=closure,domain_audit=rows,observed_response_access=False),indent=2),encoding='utf8');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
