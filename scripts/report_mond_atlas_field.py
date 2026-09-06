"""Report the first conditional real-source full-field experiment honestly."""
from __future__ import annotations
import argparse,io,unittest
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT,read_json,digest,write_json,write_csv,sparc_inputs
from mond_atlas_image_io import read_primary_image
from build_mond_atlas_ngc2903_source import pixel_geometry


def report(output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    base=ROOT/'work/gravity-first-principles';sdir=base/'mond-atlas-source-001';fdir=base/'mond-atlas-field-001';vdir=base/'mond-atlas-field-002'
    source=read_json(sdir/'source-audit.json');field=read_json(fdir/'field-audit.json');vector=read_json(vdir/'vector-audit.json');config=source['protocol']
    for audit in (source,field,vector):
        for binding in audit['bindings']:
            assert digest(ROOT/binding['path'])==binding['sha256'],binding['path']
    assert digest(ROOT/source['source_packet'])==source['source_packet_sha256']
    for asset in source['source_assets']:assert digest(ROOT/asset['file'])==asset['sha256']
    assert field['source_audit_sha256']==digest(sdir/'source-audit.json')
    assert digest(ROOT/'configs/mond_atlas_ngc2903_field_v1.json')==source['protocol_sha256']==field['protocol_sha256']
    nominal=read_json(fdir/'nominal-result.json');fine=read_json(vdir/'lateral_finer-result.json')
    comparable=read_json(fdir/'lateral_refined-result.json');sym=read_json(vdir/'axisym_lateral_refined-result.json')
    models={case['id']:read_json(fdir/(case['id']+'-result.json')) for case in config['cases']}
    # Published rotation amplitudes are transformed only by distance and sine(i).
    # Missing sky-position information prevents exact re-extraction of new annuli.
    curves,metadata,_,_=sparc_inputs();curve=next(g for g in curves if g['name']=='NGC2903');meta=metadata['NGC2903']
    values=np.array(curve['rows'],float);distance_scale=config['geometry']['distance_mpc']/float(curve['distance_mpc'])
    radius=values[:,0]*distance_scale;published_projected=values[:,1]*np.sin(np.deg2rad(meta['inclination_deg']))
    error_projected=values[:,2]*np.sin(np.deg2rad(meta['inclination_deg']))
    included=(radius>=config['numerics']['force_radius_min_kpc'])&(radius<=config['numerics']['force_radius_max_kpc'])
    rows=[];scores=[]
    for name,result in {**models,'nominal_finer':fine}.items():
        p=result['profile'];r=np.array([a['radius_kpc'] for a in p]);predictions={}
        for gravity in ('newton','mond'):
            pred=np.interp(radius,r,[a[gravity+'_force_speed_kms'] for a in p])*np.sin(np.deg2rad(config['geometry']['inclination_deg']))
            predictions[gravity]=pred
            residual=pred[included]/published_projected[included]-1
            scores.append(dict(case=name,gravity=gravity,compared_radii=int(included.sum()),
                rms_fractional_amplitude_error=float(np.sqrt(np.mean(residual**2))),mean_fractional_amplitude_error=float(np.mean(residual)),
                fitted_parameters=0,independent_likelihood=False))
        for i in range(len(radius)):
            rows.append(dict(case=name,published_radius_kpc=float(values[i,0]),model_distance_radius_kpc=float(radius[i]),
                published_projected_rotation_amplitude_kms=float(published_projected[i]),published_formal_error_projected_kms=float(error_projected[i]),
                inside_predeclared_force_range=bool(included[i]),
                newton_projected_force_speed_kms=float(predictions['newton'][i]) if included[i] else None,
                mond_projected_force_speed_kms=float(predictions['mond'][i]) if included[i] else None))
    write_csv(output/'conditional-motion-comparison.csv',rows);write_csv(output/'conditional-motion-scores.csv',scores)
    structure=[]
    for original,circular in zip(comparable['profile'],sym['profile']):
        structure.append(dict(radius_kpc=original['radius_kpc'],
            axisym_minus_mapped_mond_speed_fraction=circular['mond_force_speed_kms']/original['mond_force_speed_kms']-1,
            mapped_mond_tangential_fraction=original['mond_tangential_fraction'],axisym_mond_tangential_fraction=circular['mond_tangential_fraction']))
    write_csv(output/'structure-comparison.csv',structure)
    # A source-only diagnostic of the publisher's dust separation, same valid pixels.
    assets={a['role']:a for a in source['source_assets'] if 'role' in a};star,h=read_primary_image(ROOT/assets['STELLAR_MASS_MAP']['file'])
    mask,_=read_primary_image(ROOT/assets['STELLAR_ICA_MASK']['file']);dust_path=ROOT/'work/private/ngc2903-matter-002/NGC2903.nonstellar.fits'
    transfer=read_json(ROOT/config['p5_transfer_receipt']);assert digest(dust_path)==transfer['dust_sha256']
    dust,dh=read_primary_image(dust_path)
    assert star.shape==dust.shape and all(h.get(k)==dh.get(k) for k in ('CRPIX1','CRPIX2','CRVAL1','CRVAL2','CD1_1','CD1_2','CD2_1','CD2_2'))
    h=h.copy();h['CRPIX1']+=3;h['CRPIX2']+=1;x,y,area,_=pixel_geometry(h,star.shape,config['geometry']);rr=np.hypot(x,y)
    valid=(mask==0)&np.isfinite(star)&np.isfinite(dust);photometry=[]
    for lo,hi in ((0,2),(2,5),(5,10),(10,15),(15,20),(0,20)):
        use=valid&(rr>=lo)&(rr<hi);stellar=float(np.sum(star[use]*area[use]));nonstellar=float(np.sum(dust[use]*area[use]))
        photometry.append(dict(inner_kpc=lo,outer_kpc=hi,nonstellar_fraction_of_combined_light=nonstellar/(stellar+nonstellar),
            masked_area_fraction=1-float(np.sum(area[use])/np.sum(area[(rr>=lo)&(rr<hi)])),
            stellar_luminosity_on_same_valid_pixels_lsun=stellar*704.04*1e6))
    write_json(output/'stellar-dust-check.json',dict(status='PUBLISHER_ICA_DECOMPOSITION_NOT_NEW_STELLAR_MASS_MEASUREMENT',
        apertures=photometry,dust_path=str(dust_path.relative_to(ROOT)),dust_sha256=digest(dust_path),
        caveat='ICA components have model uncertainty. This is not evidence that a particular catalog stellar mass is wrong; apertures, sky subtraction and distances must first be matched.'))
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_mond_atlas*.py')
    log=io.StringIO();test_result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (output/'validation.log').write_text(log.getvalue(),encoding='utf-8',newline='\n')
    assert test_result.wasSuccessful(),log.getvalue()
    status=dict(goal_complete=False,goal_remains_active=True,previous_turn_classification='progress',
        object_identity_groups=13525,eligible_radial_comparison_galaxies=126,
        galaxies_with_conditional_real_source_full_qumond_fields=1,full_field_runs_this_milestone=11,
        source_sensitivity_cases=6,galaxies_with_validated_3d_mass_posterior=0,
        completed_full_field_galaxy_cube_predictions=0,conditional_published_rotation_points_compared=int(included.sum()),
        radial_numerical_gates=field['declared_convergence_gates_pass'],refined_vector_numerical_gates=vector['refined_vector_convergence_pass'],
        publication_status='LOCAL_ONLY_NOT_COMMITTED_OR_PUSHED',
        next_required=['Match absolute stellar photometry and aperture/sky calibration; source sensitivity ranges are not an observational posterior.',
            'Propagate signed CO/noise, unresolved pixels, native beam, geometry, distance, heights and missing baryon phases through source likelihoods.',
            'Constrain exterior baryonic fields; add AQUAL controls and field-source boundary convergence jointly.',
            'Model bar streaming, warp and pressure support; reconstruct full observed cube with validated signal mask, beam/channel response and covariance.',
            'Validate remaining stellar map transfers and expand to independent galaxies/surveys; preserve object-group holdouts.',
            'Restore ordinary download/CUDA/Git access to acquire missing products and publish milestones.'])
    write_json(output/'execution-status.json',status)
    write_json(output/'verification.json',dict(status='PASS',tests_run=test_result.testsRun,failures=len(test_result.failures),errors=len(test_result.errors),
        source_and_code_bindings_verified=True,source_raw_files_rehashed=6,conditional_field_cases_recomputed=True,
        validated_astrophysical_likelihood=False,goal_complete=False))
    def score(case,gravity):return next(s['rms_fractional_amplitude_error'] for s in scores if s['case']==case and s['gravity']==gravity)
    selected=[r for r in structure if r['radius_kpc'] in (2,5,10,15)]
    table='\n'.join(f"| {r['radius_kpc']:g} | {r['mapped_mond_tangential_fraction']*100:.2f}% | {r['axisym_mond_tangential_fraction']*100:.2f}% | {r['axisym_minus_mapped_mond_speed_fraction']*100:+.3f}% |" for r in selected)
    st='\n'.join(f"| {case['id']} | {100*score(case['id'],'newton'):.2f}% | {100*score(case['id'],'mond'):.2f}% |" for case in config['cases'])
    masses=nominal['source']['component_masses_msun'];refined=vector['comparisons']['half_to_quarter_step']
    text=f'''# MOND atlas — first real-source full-field experiment

**One galaxy now has executed three-dimensional Newtonian and full QUMOND
fields: NGC2903. These are conditional source reconstructions, not a validated
3D mass posterior or a completed cube likelihood. The atlas goal remains open.**

## What this experiment found

1. **The mass arrangement changes the direction of the pull much more than the
   mean rotation prediction.** Keeping the mapped bar and asymmetry produces a
   sideways component about 15.5% of the mean inward force at 2 kpc, 7.5% at
   5 kpc and 5.1% at 10 kpc. Averaging the same mass into circular annuli removes
   most of it. This is a force prediction from a conditional map, not an observed
   streaming detection. It is not evidence for a new gravity law.
2. **Mass conversion dominates this particular sensitivity experiment.** At
   10 kpc, the low/high stellar-and-CO conversion cases give MOND force-equivalent
   speeds of approximately 174–227 km/s; the nominal value is about 202 km/s.
   Doubling the assumed stellar and gas heights lowers the nominal 10 kpc speed
   by about 2.8 km/s. These ranges are illustrative assumptions, not confidence
   limits. The highest conversion is not selected as the preferred model.
3. **The photometric decomposition matters.** In the publisher's ICA maps,
   dust-associated light contributes {photometry[-1]['nonstellar_fraction_of_combined_light']*100:.1f}%
   of combined light on the valid pixels within 20 kpc, rising to
   {photometry[1]['nonstellar_fraction_of_combined_light']*100:.1f}% between 2 and 5 kpc.
   Counting all that light as old stars would change the source. This does not
   establish that catalog masses or stellar ages are wrong. ICA uncertainty,
   aperture and sky calibration must be checked before comparing catalogs.

| Radius (kpc) | Mapped sideways/inward force | Circular-map numerical remainder | Change in mean MOND force-speed after circular averaging |
|---|---:|---:|---:|
{table}

The structure comparison uses the same 0.25 kpc lateral grid and preserves
each component's total mass. Circular averaging also changes within-annulus
detail over 0.5 kpc. The residual sideways force of the circular case measures
discretization and boundary effects; it is not a physical current.

## Actual comparison with published motion

We also evaluated all six predeclared source cases against the 15 published
SPARC rotation points inside the predeclared 2–15 kpc model range. All 34
published points remain in the table; 19 outside that range have no prediction.
No gravity or mass-conversion parameters were fitted to these velocities.

| Conditional source case | Newtonian fractional RMS amplitude error | Full QUMOND fractional RMS amplitude error |
|---|---:|---:|
{st}

The nominal finer-grid comparison is **{100*score('nominal_finer','newton'):.2f}% Newtonian versus
{100*score('nominal_finer','mond'):.2f}% QUMOND**. This descriptive comparison improves
with QUMOND for every listed case, but it does not establish an acceptable
likelihood or select a correct mass model.

The SPARC curve assumes distance 6.6 Mpc and inclination 66 degrees; the independent
photometric source protocol uses 9.058 Mpc and 61.748 degrees. Published radii are
scaled by distance, and both sides are compared as projected rotation amplitudes
V sin(i). These are **not raw line-of-sight velocities**. Published annuli cannot
be exactly reconstructed at the changed geometry from this table. Bars, pressure,
warps, beam response and correlated errors are not modeled by this comparison;
its RMS is not a chi-square significance or a clean held-out prediction.

## Source and numerical evidence

The source builder reads cleaned stellar light, every nonzero ICA mask label,
THINGS HI moment zero and HERACLES signed CO plus its error map. It reads no
target rotation speed or dynamical mass. Native map blurring remains. The CO
publisher's integration window partly uses HI velocity information, so tracer
products are not fully independent of the kinematic observations.

The nominal tapered source contains approximately {masses['stellar_luminosity']/1e9:.2f}
billion solar masses in stars, {masses['atomic_helium']/1e9:.2f} billion in atomic
gas plus helium, and {masses['co21']/1e9:.2f} billion in molecular gas plus helium.
No dark-halo mass is added. Missing area is either left unfilled conditionally or
filled from observed annular means. Neither choice measures the missing matter.
Finite-cell coverage is estimated by pixel-center area assignment; native pixel
and beam sampling must be included in later source convergence.

Signed flux is averaged before nonnegative projection. The negative CO cells
removed by that projection amount to **6.9% of the signed measured CO integral**.
This is a visible noise-bias risk, not a correction known to be valid. CO errors
are retained as fully correlated within-cell bounds; no full source measurement
likelihood or complete missing-phase budget has been established. The source is
linearly tapered between 18 and 20 kpc; exterior mass is not inferred as zero.

Combined baryons are lifted with explicit exponential vertical profiles. The
solver applies the QUMOND nonlinear step to the combined Newtonian vector field
and solves the second Poisson equation, following the
[full-field formulation](https://arxiv.org/abs/0911.5464). It does not sum separately
boosted components. Stellar cleaning and masks follow the
[S4G product definitions](https://irsa.ipac.caltech.edu/data/SPITZER/S4G/docs/P5_README.html).

Eleven full field runs were executed: six source cases and five numerical
controls. The nominal 0.5 kpc grid passed mean radial-force checks but failed the
stricter full-vector comparison. This failure is retained. Refining from 0.25
to 0.125 kpc reduces the vector difference to **{refined['newton_vector_relative_rms']*100:.3f}%
Newtonian and {refined['mond_vector_relative_rms']*100:.3f}% QUMOND**, with the worst
individual ring below 0.9%. Vertical refinement changes the vector by under
0.3%; increasing the box from 24 to 32 kpc half-width changes QUMOND by about
0.20% aggregate, with the worst ring below 0.8%.

These controls establish numerical stability of this conditional interior
calculation. They do not identify the galaxy's real external field. Newtonian
boundaries include monopole, dipole and quadrupole; MOND uses the isolated
spherical monopole boundary. Missing nonspherical exterior terms are only
tested through the reported box change. No AQUAL control is completed here.

## Work remaining and replay

The next decisive test is whether a source-supported bar/streaming model predicts
the actual channel cube better than an ordinary rotating warped disk, using a
validated signal mask, instrument response and noise covariance. Stellar-map
calibration, mass/depth uncertainties and external fields must enter that test.
This one conditional galaxy does not satisfy the target of 10–20 validated
development pilots or the later 100–300 resolved sample. The 13,525 catalog
identity groups are not 13,525 resolved 3D models.

Raw sources remain private. This milestone is **local only**: the linked Git
metadata is outside the writable workspace, so fetch/commit/push are unavailable.
The CUDA environment and new shell downloads also remain unavailable.

- [Conditional motion predictions and excluded radii](conditional-motion-comparison.csv)
- [Conditional motion scores](conditional-motion-scores.csv)
- [Same-grid structure comparison](structure-comparison.csv)
- [Stellar/dust aperture check](stellar-dust-check.json)
- [Source audit and assumptions](../mond-atlas-source-001/source-audit.json)
- [Initial field convergence](../mond-atlas-field-001/field-audit.json)
- [Stricter vector convergence](../mond-atlas-field-002/vector-audit.json)
- [Verification](verification.json), [test log](validation.log), [outstanding work](execution-status.json)

Run with Python/NumPy from the repository root, choosing unused output folders:

```text
python scripts/build_mond_atlas_ngc2903_source.py --output work/gravity-first-principles/source-replay --private work/private/source-replay
python scripts/run_mond_atlas_ngc2903_fields.py --source work/gravity-first-principles/source-replay --output work/gravity-first-principles/field-replay --private work/private/field-replay --convergence
python scripts/check_mond_atlas_field_pattern.py --source work/gravity-first-principles/source-replay --previous work/gravity-first-principles/field-replay --output work/gravity-first-principles/vector-replay --private work/private/vector-replay
python -m unittest discover -s tests -p "test_mond_atlas*.py" -v
```
'''
    (output/'README.md').write_text(text,encoding='utf-8',newline='\n')
    paths=[Path(__file__),sdir/'source-audit.json',fdir/'field-audit.json',vdir/'vector-audit.json',ROOT/'configs/sparc_rotation_curves_full_v1.json',
           base/'map-response-metadata-001/SPARC_Lelli2016c.mrt']
    paths+=list(fdir.glob('*-result.json'))+list(vdir.glob('*-result.json'))
    write_json(output/'input-bindings.json',{str(p.relative_to(ROOT)):digest(p) for p in paths})
    print(dict(report=str(output/'README.md'),tests=test_result.testsRun,goal_complete=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report(args.output.resolve())
