"""Report corrected source/field consistency and repeated background failures."""
from __future__ import annotations
import argparse,csv,io,re,shutil,unittest
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest,canonical_name
from check_mond_atlas_field_pattern import forces

BASE=ROOT/'work/gravity-first-principles'
NAMES=('mond-atlas-field-005','mond-atlas-source-basis-001','mond-atlas-field-006',
       'mond-atlas-offplane-001','mond-atlas-noise-robustness-001')


def csv_rows(path):
    with path.open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def report(output):
    if output.exists():raise FileExistsError('immutable report')
    folders=[BASE/n for n in NAMES];summaries=[read_json(p/'summary.json') for p in folders]
    replay,source,field,offplane,noise=summaries
    for summary in summaries:
        for group in ('bindings','input_bindings','source_bindings','code_hashes'):
            for path,expected in summary.get(group,{}).items():
                if digest(ROOT/path)!=expected:raise ValueError('bound input changed: '+path)
        for product in summary.get('products',[]):
            if digest(ROOT/product['path'])!=product['sha256']:raise ValueError('source product changed')
    old_manifest_path=BASE/'mond-atlas-execution-007/publication-manifest.json'
    old_manifest=read_json(old_manifest_path)
    for item in old_manifest['files']:
        if digest(ROOT/item['path'])!=item['sha256']:raise ValueError('earlier milestone changed: '+item['path'])
    output.mkdir(parents=True)
    old_handoff=ROOT/'docs/MOND_OBSERVATION_ATLAS_GOAL.md'
    (output/'prior-goal-handoff.md').write_bytes(old_handoff.read_bytes())
    integrity=[]
    for folder in (folders[0],folders[2]):
        for path in sorted(folder.glob('*-result.json')):
            result=read_json(path);f=forces(folder,result['id'])
            mass_error=abs(sum(result['source']['component_mass_msun'].values())/result['source']['finite_grid_total_mass_msun']-1)
            profile_error=0.
            for p in result['profile']:
                r=p['radius_kpc'];use=f[:,0]==r
                if use.sum()!=72:raise ValueError('force angular sampling changed')
                for theory,start in [('newton',2),('mond',4)]:
                    profile_error=max(profile_error,abs(np.sqrt(r*f[use,start].mean())-p[theory+'_force_speed_kms']))
            residual=max(result['numerical'][k]['relative_pde_residual'] for k in ('newton','mond'))
            if max(mass_error,profile_error,residual)>1e-10:raise ValueError('field integrity failed')
            for asset in result['files']:
                if (ROOT/asset['path']).stat().st_size!=asset['bytes']:raise ValueError('field asset size changed')
            integrity.append(dict(id=result['id'],mass_relative_error=mass_error,
                maximum_profile_replay_error_kms=profile_error,maximum_pde_relative_residual=residual))
    write_csv(output/'field-integrity.csv',integrity)
    sensitivity=[]
    thin=forces(folders[2],'common_thin_lateral');mixed=forces(folders[2],'common_mixed_lateral')
    if not np.array_equal(thin[:,:2],mixed[:,:2]):raise ValueError('different sensitivity sample points')
    for r in np.unique(thin[:,0]):
        use=thin[:,0]==r
        for theory,start in [('newton',2),('mond',4)]:
            a=thin[use,start:start+2];b=mixed[use,start:start+2]
            va=np.sqrt(r*a[:,0].mean());vb=np.sqrt(r*b[:,0].mean())
            sensitivity.append(dict(theory=theory,radius_kpc=r,thin_force_equivalent_speed_kms=va,
                mixed_force_equivalent_speed_kms=vb,speed_fractional_change=vb/va-1,
                thin_tangential_rms_over_inward_mean=np.sqrt(np.mean(a[:,1]**2))/a[:,0].mean(),
                mixed_tangential_rms_over_inward_mean=np.sqrt(np.mean(b[:,1]**2))/b[:,0].mean(),
                tangential_difference_over_thin_inward_mean=np.sqrt(np.mean((b[:,1]-a[:,1])**2))/a[:,0].mean(),
                vector_difference_over_thin_rms=np.sqrt(np.sum((b-a)**2)/np.sum(a**2))))
    write_csv(output/'midplane-source-sensitivity.csv',sensitivity)
    atlas_assets=csv_rows(BASE/'mond-atlas-catalog-004/assets.csv')
    identities={r['atlas_identity_id']:r['object_group_id'] for r in csv_rows(BASE/'mond-atlas-identity-001/identity-to-object.csv')}
    astrometry=read_json(BASE/'mond-atlas-astrometry-001/summary.json')
    image_pass=set(astrometry['footprint_strict_pass']);noise_pass=set(noise['split_stable_pass'])
    transfer=read_json(BASE/'ngc2903-matter-002/p1-p5-registration.json')
    eligibility=[];role_overlay=[]
    for item in atlas_assets:
        effective=item['role'];reason='unchanged'
        if effective=='STELLAR_MASS_MAP':
            if item['bunit']!='MJy/sr':raise ValueError('unexpected cleaned stellar units')
            effective='STELLAR_ICA_3P6_FLUX';reason='Publisher P5 stellar image is cleaned flux, requiring a separate light-to-mass conversion.'
        role_overlay.append(dict(path=item['path'],atlas_identity_id=item['atlas_id'],object_group_id=identities[item['atlas_id']],
            original_role=item['role'],effective_measurement_role=effective,units=item['bunit'],reason=reason,
            original_asset_sha256=item['sha256']))
    write_csv(output/'asset-role-overlay.csv',role_overlay)
    for name in sorted(image_pass|set(noise['split_stable_pass'])|set(noise['split_sensitive_or_failed'])):
        identity='NAME:'+canonical_name(name)
        assets=[a for a in atlas_assets if a['atlas_id']==identity];roles={a['role'] for a in assets}
        if not assets:raise ValueError('pilot identity missing: '+name)
        cleaned='STELLAR_MASS_MAP' in roles
        eligibility.append(dict(galaxy=name,object_group_id=identities[identity],
            raw_or_P1_astrometry_pass=name in image_pass,background_all_declared_splits_pass=name in noise_pass,
            joint_image_and_background_prerequisites_pass=name in image_pass and name in noise_pass,
            cleaned_stellar_flux_available=cleaned,stellar_input='P5 cleaned flux' if cleaned else 'raw IRAC flux',
            cleaned_stellar_relative_transfer_validated=name=='NGC2903' and transfer['pass_gate'],
            co21_moment_and_error_available={'CO21_MOM0','CO21_EMOM0'}.issubset(roles),
            co_spatial_total_baryon_coverage_validated=False,total_mass_3d_posterior_validated=False,
            admitted_full_field_cube_likelihood=False,development_exposed=True))
    write_csv(output/'pilot-readiness.csv',eligibility)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_mond_atlas*.py')
    log=io.StringIO();tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (output/'validation.log').write_text(log.getvalue(),encoding='utf-8',newline='\n')
    if not tests.wasSuccessful():raise RuntimeError(log.getvalue())
    current_counts=dict(object_identity_groups=13525,certified_distinct_galaxies=False,radial_baseline_galaxies=175,
        radial_descriptive_cut_galaxies=126,resolved_seed_galaxies=12,raw_seed_assets=137,
        source_image_fits_total=22,conditional_field_runs_total=29,conditional_field_galaxies=1,
        new_field_runs=10,background_partition_evaluations=noise['partition_evaluations'],
        background_split_stable_galaxies=len(noise_pass),
        joint_raw_image_and_background_prerequisites=sum(r['joint_image_and_background_prerequisites_pass'] for r in eligibility),
        admitted_full_field_cube_predictions=0,unit_tests_passed=tests.testsRun)
    status=dict(status='LOCAL_SOURCE_FIELD_AND_NOISE_VALIDATION_MILESTONE',goal_complete=False,goal_remains_active=True,
        source_admission_disposition='SOURCE_BLOCKED',publication_status='LOCAL_ONLY_GITHUB_WRITE_APPROVAL_UNAVAILABLE',counts=current_counts,
        previous_mixed_source_refinement_pass=replay['mixed_refinement_gates_pass'],
        common_basis_midplane_numerical_gates_pass=field['numerical_gates_pass'],
        common_basis_offplane_full_vector_gates_pass=offplane['offplane_full_vector_gates_pass'],
        background_split_failures=noise['split_sensitive_or_failed'],
        required_remaining=[
            'Resolve the source covariance, native pixel/beam forward projection, absolute flux, geometry provenance, mass conversions and missing components before admitting motion scores.',
            'Repair and independently test background covariance where split changes expose failures; do not select a favorable partition.',
            'Treat 3D sources as constrained ensembles; quantify external fields and refine numerical failures without changing gates.',
            'Complete distinct AQUAL controls and a validated warped/streaming/pressure-supported cube likelihood.',
            'Validate additional pilot source maps, then execute the 10-20 pilots and expand eligible resolved objects with galaxy/group/survey holdouts.',
            'Publish the verified manifest when Git or connector writes are permitted, checking remote ancestry and preserving concurrent changes. Both current routes are blocked.'])
    write_json(output/'execution-status.json',status)
    write_json(output/'publication-access.json',dict(
        repository='lrspeiser/Invariant',verified_remote_main='afc721a13782acec4ebc94ad8f6d97ed71be7152',
        remote_comparison_status='identical',remote_base_tree='322287a59fbb75aaf447c642ffba6e9ad86ffd01',
        local_fetch=dict(command='git fetch origin main',exit_code=1,
            error='Cannot open linked worktree FETCH_HEAD outside the writable root: Permission denied.'),
        connector_reads_succeeded=True,
        connector_write=dict(action='create_blob',path='configs/mond_atlas_astrometry_v1.json',
            expected_git_blob_sha='88229dc943aca3004a00dfdb139497060d3c0598',is_error=True,
            message='MCP tool call requires approval, but approval policy is never'),
        returned_new_blob_sha=None,created_commit_sha=None,ref_update_attempted=False,published=False,
        scope='Records the actual tool outcomes observed during this milestone; no alternate write was attempted after the connector block.'))
    write_json(output/'verification.json',dict(status='ARTIFACT_INTEGRITY_AND_UNIT_TESTS_PASS_SCIENTIFIC_ADMISSION_INCOMPLETE',
        tests_run=tests.testsRun,failures=len(tests.failures),errors=len(tests.errors),field_results_checked=len(integrity),
        potential_hashes_verified_by_offplane_runner=len(offplane['verified_potential_assets']),
        field_assets_sizes_rechecked=True,large_potential_hashes_not_repeated_by_report=True,
        prior_manifest_files_unchanged_before_handoff_update=len(old_manifest['files']),
        updated_existing_file='docs/MOND_OBSERVATION_ATLAS_GOAL.md',new_galaxy_motion_scores=0,
        midplane_gates_pass=field['numerical_gates_pass'],offplane_gates_pass=offplane['offplane_full_vector_gates_pass'],goal_complete=False))
    fits=csv_rows(folders[1]/'source-basis-comparison.csv');noise_rows=csv_rows(folders[4]/'galaxies.csv')
    fit_table='\n'.join(f"| {r['id']} | {100*float(r['old_cell_basis_image_rms']):.2f}% | {100*float(r['old_values_with_actual_field_basis_image_rms']):.2f}% | {100*float(r['refitted_common_basis_image_rms']):.2f}% |" for r in fits)
    field_table='\n'.join(f"| {r['case']} / {r['perturbation']} | {100*r['newton_vector_relative_rms']:.3f}% | {100*r['mond_vector_relative_rms']:.3f}% | {'pass' if r['gates_pass'] else 'FAIL'} |" for r in field['checks'])
    off_table='\n'.join(f"| {r['case']} / {r['perturbation']} / {r['theory']} | {100*r['vector_relative_rms']:.3f}% | {100*r['maximum_group_relative_rms']:.3f}% | {100*r['vertical_component_relative_rms']:.3f}% | {'pass' if r['full_vector_gates_pass'] else 'FAIL'} |" for r in offplane['checks'])
    noise_table='\n'.join(f"| {r['galaxy']} | {r['unique_passes']}/{r['unique_partitions']} | {float(r['channel_lag1_min']):.3f} to {float(r['channel_lag1_max']):.3f} | {r['failed_gates'] or 'none'} |" for r in noise_rows)
    selected=[r for r in sensitivity if r['radius_kpc'] in (2.,5.,10.)]
    sensitivity_table='\n'.join(f"| {r['theory']} | {r['radius_kpc']:g} | {100*r['speed_fractional_change']:+.2f}% | {100*r['thin_tangential_rms_over_inward_mean']:.2f}% | {100*r['mixed_tangential_rms_over_inward_mean']:.2f}% |" for r in selected)
    above=csv_rows(folders[3]/'conditional-source-sensitivity.csv')
    vertical_table='\n'.join(f"| {r['theory']} | {float(r['height_kpc']):g} | {100*float(r['downward_mean_fractional_change']):+.2f}% |" for r in above if float(r['radius_kpc'])==5.)
    text=f'''# MOND atlas: consistent source maps, 3D force checks and noise robustness

The image and gravity calculations now use the same mathematical description
of the source. The revised stellar light fits have **1.21% and 4.80% image RMS
mismatch** for two different assumed depth arrangements. These are conditional
reconstructions, not measured depths or a noise-calibrated posterior.

A useful conditional pattern is that the two source models have quite similar
mean rotational force but much less similar vertical force. At 5 kpc radius,
their QUMOND force-equivalent speeds differ by about **1.7%**, while their mean
downward force 0.25 kpc above the disk differs by about **32%**. Newtonian gravity
shows the same qualitative sensitivity. This is a model comparison, not an
observed force difference or evidence that one gravity law is correct.

The broader background check also exposes a practical limit: **9/12 galaxies
pass every declared split; NGC2841, NGC2903 and NGC3198 do not.** A favorable
single partition had understated that uncertainty. No new galaxy motion
comparison was performed in this phase, and the atlas remains unfinished.

## The source correction

The earlier image inverse used constant brightness inside each map pixel,
whereas the gravity code interpolated between pixel centers. Those are different
light distributions. For the thin stellar case, its reported 0.41% image error
became 5.04% when projecting the distribution actually used by the gravity code.

The new inverse integrates the same bilinear basis used by the field loader.
Independent line-of-sight and analytic finite-pixel tests check that operator.
The source fit, support, coverage weights and regularization are unchanged.

| Source | Earlier constant-pixel fit | Earlier values under gravity basis | Refit common basis |
|---|---:|---:|---:|
{fit_table}

The stellar alternatives are a single 0.1 kpc exponential layer and a mixture
with 25% of its light in a 0.1 kpc layer and 75% in a 0.4 kpc layer. Gas layers
remain 0.2 kpc. The 5% image diagnostic is only a gross mismatch flag: its weights
describe coverage, not the actual source-noise covariance. The alternatives
also differ in recovered planar structure, so their gravity difference cannot
be attributed to thickness alone.

The catalog now has an explicit [asset-role overlay](asset-role-overlay.csv):
five files historically named `STELLAR_MASS_MAP` contain cleaned **flux in
MJy/sr**, requiring a separate mass-to-light conversion. This follows the
[publisher's P5 description](https://irsa.ipac.caltech.edu/data/SPITZER/S4G/docs/P5_README.html).
Original receipt names and hashes remain intact.

## Numerical checks for both revised sources

A disk-backed implementation reproduces the previous dense calculation to
roundoff. It applies global transforms in smaller groups; it does not split
the galaxy into independently gravitating chunks. The previous mixed-source
failure was resolved by another lateral refinement: 1.06% Newtonian and 0.95%
QUMOND vector RMS difference, below the unchanged 3% aggregate and 5% ring gates.
That success applied to the old coefficients, so both new source fits were
tested separately.

For each new model, the base grid is 0.125 kpc in all directions in a box with
24 kpc half-width. Separate perturbations halve horizontal spacing, halve
vertical spacing, or enlarge the box half-width to 32 kpc. Each is compared
with the same base, over 2–15 kpc. Their joint-refinement cross terms are not bounded.

| Model / check | Newtonian vector RMS | QUMOND vector RMS | 3% aggregate and 5% ring gates |
|---|---:|---:|---|
{field_table}

The potentials solve Newtonian gravity and [QUMOND](https://arxiv.org/abs/0911.5464)
from combined baryons, with the same fixed conversion factors and isolated
boundary assumptions. Box convergence does not establish physical isolation.
This is not an AQUAL calculation or an observational motion likelihood.

## The check now includes forces above the disk

The additional audit samples all three force components at 0.25, 0.5 and 1 kpc
above the plane, 14 radii and 72 azimuths. Mirrored points test the vertical
reflection symmetry of the declared source. Quadratic potentials with cross
terms provide an independent exact force-sampling benchmark.

| Model / check / law | Full-vector RMS | Worst radius-height group | Vertical component RMS | Full-vector gates |
|---|---:|---:|---:|---|
{off_table}

Full-vector gates retain 3% aggregate and 5% per radius-height group. The
vertical-only column has its own denominator and remains a separate diagnostic:
passing a total-vector gate does not mean every component has that accuracy.
Only the stated sample positions and perturbations are certified by these checks.

## What the alternative source arrangements change

These are conditional field comparisons, not observations of a galaxy changing
shape. The following uses the horizontally refined solutions; all numerical
flags above still apply. A force-equivalent speed summarizes mean inward force
as sqrt(radius × mean inward acceleration); it is not an actual circular orbit
in a barred potential.

| Law | Radius (kpc) | Mixed-versus-thin speed change | Thin sideways RMS / mean inward force | Mixed sideways RMS / mean inward force |
|---|---:|---:|---:|---:|
{sensitivity_table}

The model totals are 50.43 and 50.18 billion solar masses, only a 0.50% change.
Their distributions differ in depth and in the reconstructed planar structure.
At radius 5 kpc, the mixed source predicts less downward force than the thin
source, with the difference decreasing farther above the disk:

| Law | Height above disk (kpc) | Mixed-versus-thin mean downward force change |
|---|---:|---:|
{vertical_table}

This supplies a concrete direction for future observations: independently
constrained vertical structure or vertical motions may separate models whose
mean rotation predictions differ much less. Converting gas thickness into
gravity already requires a pressure/equilibrium model; it cannot be treated as
a direct, gravity-independent force measurement. The two source alternatives
also fit the image with different errors, so these percentages are not an
observational confidence interval on the galaxy's gravity.

The complete [midplane table](midplane-source-sensitivity.csv) and
[above-plane table](../mond-atlas-offplane-001/conditional-source-sensitivity.csv)
retain signed changes and component values. Distinct depth alternatives can
therefore be tested through directional and vertical information, but these two
source fits are not a calibrated ensemble and do not establish a measured effect.

## Background noise depends on the calibration region

Eight frozen geometry-only partitions were evaluated in both directions,
producing 16 unique splits per galaxy and **192 completed checks**. Spatial
guards, the background annulus, covariance form and diagnostic limits were
unchanged. Galaxy-region values were zeroed before evaluation. We did not select
a winning split. These splits reuse the same background realization; their
failure fractions are neither independent probabilities nor p-values.

| Galaxy | Passing splits | Residual adjacent-channel product range | Failed diagnostic(s) |
|---|---:|---:|---|
{noise_table}

The residual adjacent-channel diagnostic is the mean product of neighboring
whitened channels, with an absolute limit of 0.15; it is not a normalized Pearson
coefficient. All global whitened mean-square values pass the existing 0.5–2.0
range. Some NGC3198 quadrant checks also fail. The cause could include covariance
form, nonstationary noise, residual emission or finite calibration samples.
This audit alone does not distinguish them. In particular, NGC2903's two failed
splits prevent treating its earlier single-split pass as robust noise validation.

## Actual atlas coverage and remaining work

- **13,525 object groups** in the identity overlay, with unresolved associations;
  these are not certified distinct galaxies or complete 3D models.
- **175 radial baselines**, with 126 meeting the fixed descriptive cuts. Those
  earlier algebraic radial calculations are not full-field disk solutions.
- **12 resolved seed galaxies / 137 original assets**. Eight pass both the raw
  image astrometry and every current background split; this is only a pair of
  prerequisites. See [pilot readiness](pilot-readiness.csv).
- **22 source-image fits and 29 conditional full-field runs**, including replay
  and convergence runs, still covering only **one field galaxy**.
- **{tests.testsRun} passing unit tests; zero admitted full-field galaxy cube likelihoods.**

The next scientific requirements are source covariance and native-pixel/beam
projection, absolute photometry, recovery of the missing raw geometry tables,
independent mass/depth/environment constraints, missing baryonic phases, AQUAL
controls, and a validated motion cube model including pressure and noncircular
motions. The remaining pilots and resolved-sample expansion have not been run.

All new source and numerical packages prospectively declare `SOURCE_BLOCKED`.
That describes admission for observational scoring, not a failure to execute
these diagnostics. The earlier exploratory rotation comparison remains
nonadmitted under the [repository policy](../../../docs/OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md).

## Reproducibility and publication

The [verification record](verification.json), [test log](validation.log),
[field integrity](field-integrity.csv), [status](execution-status.json), and
[input bindings](input-bindings.json) distinguish completed work from remaining
requirements. The [publication manifest](publication-manifest.json) lists only
intended code, configuration and compact results; all raw observations and
large numerical fields stay outside Git.

Ordinary `git fetch origin main` failed because the linked worktree's Git
metadata is outside the writable workspace. The connected GitHub integration
can read the repository, and its remote comparison verified `main` still at
`afc721a1`. Its first blob-creation call was then rejected because the tool
requires approval and this session's approval policy is `never`. No remote
blob SHA, commit or ref update was returned; nothing was published. The
manifest and exact-byte transfer helper are retained for ordinary publication
when permitted. Downloads and the previous CUDA environment remain unavailable.
Computations here use the working bundled CPU runtime.

To replay, choose unused output paths and preserve the named source bindings:

```text
python scripts/run_mond_atlas_common_basis_fields.py --output NEW_FIELD_DIR --private NEW_PRIVATE_DIR
python scripts/run_mond_atlas_noise_robustness.py --output NEW_NOISE_DIR
python scripts/run_mond_atlas_offplane.py --output NEW_OFFPLANE_DIR
python -m unittest discover -s tests -p "test_mond_atlas*.py" -v
```

The off-plane protocol binds the original field directory. To inspect a newly
replayed field set, create a new protocol copy with its explicit path rather
than editing a frozen result.
'''
    (output/'README.md').write_text(text,encoding='utf-8',newline='\n')
    handoff=f'''# Active goal: MOND observation atlas

User authorization: “Make the above our goal and complete the work,” 2026-09-06.
The active Codex goal remains unfinished. Build the catalog and execute ordinary-
matter Newtonian/full-field MOND predictions from independently constrained 3D
ensembles, then compare through validated instruments, masks and noise. The
experiment contains no dark-halo term. A spectral velocity axis is not depth.

## Current result

Read `work/gravity-first-principles/{output.name}/README.md` and its
`execution-status.json`, `verification.json`, and `publication-manifest.json`.
This supersedes execution-007's current status while preserving every earlier
scientific artifact and failure. The previous handoff is archived with the report.

- Identity overlay: 13,525 groups, not certified distinct galaxies; 90 proximity
  pairs and 58 missing coordinates remain unresolved.
- Radial baseline: 175 galaxies, 126 passing the declared descriptive cuts.
- Source basis corrected: image inversion and gravity interpolation now share
  bilinear source nodes. Revised stellar fits have 1.21% and 4.80% image RMS;
  they are not measured depths or a calibrated source posterior.
- Previous mixed-source convergence failure resolved: 1.06% Newtonian and 0.95%
  QUMOND at the next lateral refinement. The former failure remains retained.
- Both corrected coefficient sets now have separate lateral, vertical and box
  checks: midplane full-vector gates pass = {field['numerical_gates_pass']}.
- Above-plane three-component checks at 0.25, 0.5 and 1 kpc: full-vector gates
  pass = {offplane['offplane_full_vector_gates_pass']}. Read component errors and
  individual failures rather than interpreting one Boolean as universal accuracy.
- Conditional pattern: at radius 5 kpc the two models change QUMOND's mean
  force-equivalent speed by 1.7%, but its downward pull at height 0.25 kpc by
  32%. Model total mass differs by only 0.50%. This is joint deprojection
  sensitivity, not an observed anomaly or an observational confidence interval.
- Noise: 192 partition checks. Nine galaxies pass every declared split;
  NGC2841, NGC2903 and NGC3198 fail some. Eight galaxies pass both raw/P1
  astrometry and all background partitions. These are not valid galaxy likelihoods.
- Five files historically named STELLAR_MASS_MAP are explicitly classified as
  cleaned stellar flux in MJy/sr by the new role overlay, not preconverted mass.
- Totals: 22 source-image fits, 29 conditional field runs for one galaxy,
  {tests.testsRun} passing unit tests, zero admitted full-field galaxy cube predictions.

## Admission and remaining work

Read `docs/OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md` before any new
source/solver or motion scoring. New packages prospectively declare SOURCE_BLOCKED.
The older motion comparison remains exploratory and nonadmitted; no retrospective
preregistration is claimed. Keep all previously used seed galaxies development-exposed.

1. Model source noise, native beams/pixel footprints, absolute calibration,
   source conversions and missing mass phases; retain nondetections and missingness.
2. Recover both missing original S4G geometry tables; the derived record is bound
   but raw-source revalidation is incomplete. Validate P1-to-P5 transfer beyond
   NGC2903 and account for dust in the seven raw IRAC stellar inputs.
3. Constrain geometry/depth/exterior-field ensembles independently of target
   motions; resolve remaining numerical failures without weakening gates.
4. Improve and validate covariance across spatial splits and within the galaxy;
   establish selection-mask validity. Do not choose favorable background splits.
5. Add distinct AQUAL controls and an instrument-aware motion model for pressure,
   warps, streaming and other permitted motions; execute true cube likelihoods.
6. Complete 10–20 development pilots and expand toward 100–300 eligible resolved
   systems, then evaluate galaxy/group/survey transfer. Population catalog rows
   do not substitute for executed resolved predictions.
7. Publish the combined verified milestone when a write route is permitted,
   with a fresh ancestry check and a non-forced ref update preserving remote work.

## Access and preservation

The working Python 3.12/NumPy CPU runtime is bundled under the user's
`.cache/codex-runtimes`. The previous CUDA virtualenv cannot start. Direct shell
downloads are denied. Linked Git metadata is outside the writable root; the
latest ordinary fetch failed. The connected GitHub integration can access the
repository; its preparation check found main unchanged at
afc721a13782acec4ebc94ad8f6d97ed71be7152. Its first blob write was blocked because
approval is required but the session policy is never. Nothing was published.
Do not alter the restricted local Git metadata or bypass the connector rejection.
Raw observations and large field arrays remain outside Git. Preserve unrelated
untracked work. The current report manifest contains the intended publication set.

Key immutable runs: catalog-004, identity-001, radial-002, astrometry-001,
noise-002 and noise-robustness-001, source-basis-001, field-005/006, offplane-001
(all prefixed `mond-atlas-` under `work/gravity-first-principles`). Earlier
execution reports preserve the admission correction, source representation
failures and numerical counterexamples. The goal remains active.
'''
    old_handoff.write_text(handoff,encoding='utf-8',newline='\n')
    bind_paths=[Path(__file__),old_manifest_path]+[f/'summary.json' for f in folders]+[
        BASE/'mond-atlas-catalog-004/assets.csv',BASE/'mond-atlas-identity-001/identity-to-object.csv',
        BASE/'mond-atlas-astrometry-001/summary.json',BASE/'ngc2903-matter-002/p1-p5-registration.json']
    write_json(output/'input-bindings.json',{str(p.relative_to(ROOT)):digest(p) for p in bind_paths})
    intended={ROOT/r['path'] for r in old_manifest['files']};intended.add(old_manifest_path)
    extra_names=['mond_atlas_blocked_fields','run_mond_atlas_blocked_refinement','mond_atlas_nodal_projection',
        'run_mond_atlas_source_basis','run_mond_atlas_common_basis_fields','run_mond_atlas_noise_robustness',
        'mond_atlas_force_sampling','run_mond_atlas_offplane','report_mond_atlas_consistency','prepare_mond_atlas_publication']
    intended.update(ROOT/'scripts'/(n+'.py') for n in extra_names)
    intended.update(ROOT/'configs'/(n+'.json') for n in ['mond_atlas_blocked_fields_v1','mond_atlas_source_basis_v1',
        'mond_atlas_common_basis_fields_v1','mond_atlas_noise_robustness_v1','mond_atlas_offplane_v1'])
    intended.update(ROOT/'tests'/('test_mond_atlas_'+n+'.py') for n in ['blocked','nodal','noise_robustness','force_sampling'])
    for folder in folders+[output]:intended.update(p for p in folder.rglob('*') if p.is_file())
    items=[]
    for path in sorted(intended,key=lambda p:p.as_posix()):
        relative=path.relative_to(ROOT).as_posix()
        if relative.startswith('work/private/') or path.suffix.lower() in ('.fits','.npy','.npz'):
            raise ValueError('raw or large field in publication list')
        items.append(dict(path=relative,bytes=path.stat().st_size,sha256=digest(path)))
    write_json(output/'publication-manifest.json',dict(status='LOCAL_VALIDATED_MILESTONE_WRITES_BLOCKED_SCIENTIFIC_ADMISSION_INCOMPLETE',
        base_commit=old_manifest['base_commit'],goal_complete=False,ordinary_fetch_and_remote_review_required_before_publication=True,
        raw_observation_files_included=0,source_admission_complete=False,files=items,file_count=len(items),total_bytes=sum(i['bytes'] for i in items)))
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if not link.startswith('https:') and not (output/link).is_file():raise ValueError('broken report link: '+link)
    print(dict(report=str(output/'README.md'),counts=current_counts,midplane_gates=field['numerical_gates_pass'],
        offplane_gates=offplane['offplane_full_vector_gates_pass'],manifest_files=len(items),goal_complete=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report(args.output.resolve())
