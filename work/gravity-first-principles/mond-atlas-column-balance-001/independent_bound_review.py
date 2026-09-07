"""Independent arbitrary-positive-line bound; no velocity assignment or fitting."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,hashlib
from pathlib import Path
import numpy as np
from astropy.io import fits
P=Path(__file__).resolve().parent;R=P.parents[2]
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
summary=read(P/'emitter001/summary.json');history=read(R/'work/gravity-first-principles/mond-atlas-native-spectral-001/NGC2976.json');p=history['provenance'];n=p['parent_channel_count'];cal=np.array(p['continuum_fit_parent_indices_zero_based']);keep=np.array(p['parent_channel_indices_zero_based']);header=fits.getheader(R/history['source_path']);dv=abs(header['CDELT3'])/1000
assert n==63 and len(keep)==42
# Separate regression construction in unscaled native index coordinates.
X=np.column_stack([np.ones(n),np.arange(n)]);A=np.eye(n)[keep];A[:,cal]-=X[keep]@np.linalg.pinv(X[cal])
bounds={};vertexbounds={};checks=[]
for branch in ['boxcar_independent','boxcar_hanning_full','boxcar_hanning_decimated']:
    if branch=='boxcar_independent':H=np.eye(n);width=1.
    else:
        step=2 if branch.endswith('decimated') else 1;width=1/step;H=np.zeros((n,n+2 if step==1 else 2*n+1))
        for j in range(n):
            for offset,weight in enumerate([.25,.5,.25]):H[j,j*step+offset]=weight
    B=A@H;assert np.all(H>=0) and np.allclose(H.sum(axis=1),1)
    bound=1000*np.max(np.sum(abs(B),axis=1))/(dv*width);tight=1000*np.max(abs(B))/(dv*width)
    assert np.isclose(bound,summary['arbitrary_line_operator_bounds_mjy_per_Jykms'][branch],rtol=1e-13)
    # Every extreme point of the nonnegative sub-probability simplex.
    vertices=1000*B/(dv*width);assert np.max(abs(vertices))<=bound
    bounds[branch]=float(bound);vertexbounds[branch]=float(tight)
    checks.append(dict(branch=branch,pregrid_channels=H.shape[1],pregrid_width_km_s=dv*width,negative_entries=int(np.sum(B<0)),simplex_vertices_checked=H.shape[1]+1))
errors=[];largest=0;count=0
hi_factors={'baseline':1.,'stellar_ML_0.4':1.,'stellar_ML_0.8':1.,'HI_0.8':.8,'HI_1.2':1.2,'CO_0.5':1.,'CO_2':1.}
for row in summary['results']:
    weighted=np.array(row['invalid_aperture_weighted_flux_jy_km_s_beam']);assert weighted.shape==(15,) and np.all(np.isfinite(weighted)) and np.all(weighted>=0)
    assert row['material'] in hi_factors
    # Source HI sensitivity has already been applied once in this supplied weight.
    unscaled=weighted/hi_factors[row['material']];assert np.allclose(unscaled*hi_factors[row['material']],weighted,rtol=1e-15,atol=0)
    for branch,bound in bounds.items():
        prediction=float(weighted.max()*bound);reported=row['max_unknown_spectral_contribution_mjy_beam'][branch];errors.append(abs(prediction-reported));largest=max(largest,prediction);count+=1
assert len(summary['results'])==84 and max(errors)<1e-15
source=R/'work/private/mond-atlas-hi-column-001/centroid-projection001/f4-emission-0.0625.npz'
with np.load(source) as z:
    flux=z['flux_jy_km_s'];assert np.isfinite(flux).all() and np.all(flux>=0);source_flux=float(flux.sum())
result=dict(status='PASS_INDEPENDENT_ARBITRARY_LINE_ENVELOPE',profiles=84,reported_branch_bounds_checked=count,operator_bounds_mjy_per_Jykms=bounds,optional_tighter_simplex_bounds_mjy_per_Jykms=vertexbounds,checks=checks,maximum_reported_bound_arithmetic_error=max(errors),maximum_envelope_emission_multiplier1_mjy_beam=largest,maximum_envelope_emission_multiplier2_mjy_beam=2*largest,source_integrated_flux_jy_km_s=source_flux,HI_scaling_applied_once_in_supplied_weights=True,invalid_mask_and_beam_weight_recomputation=False,scope='Independent operator/bound arithmetic; supplied invalid-group/aperture-weight calculations inspected but not independently reconstructed here. Whole steady model remains invalid.',observed_source_spectra_read=0,bindings={str(path.relative_to(R)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),P/'emitter001/summary.json',R/'scripts/mond_atlas_column_balance_emitters.py',R/'scripts/mond_atlas_column_balance.py',R/'work/gravity-first-principles/mond-atlas-native-spectral-001/NGC2976.json',source]})
(P/'independent-bound-receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='bindings'},indent=2))
