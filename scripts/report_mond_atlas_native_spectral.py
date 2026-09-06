"""Record native processing provenance and verified selected-plane closure."""
from __future__ import annotations
import argparse,copy,csv,io,re,unittest
from pathlib import Path
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest

BASE=ROOT/'work/gravity-first-principles'
RUNS=['mond-atlas-native-spectral-001','mond-atlas-preprocessing-replay-001','mond-atlas-preprocessing-replay-002']


def csv_rows(path):
    with path.open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def report(output):
    if output.exists():raise FileExistsError('immutable report')
    previous=BASE/'mond-atlas-execution-009';manifest_path=previous/'publication-manifest.json';manifest=read_json(manifest_path)
    for item in manifest['files']:
        path=ROOT/item['path']
        if path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:raise ValueError('old milestone changed: '+item['path'])
    native=read_json(BASE/RUNS[0]/'summary.json');replay=read_json(BASE/RUNS[2]/'summary.json')
    checked={}
    for summary in [native,replay,read_json(BASE/RUNS[1]/'prospective-bindings.json')]:
        for group in ['bindings','source_bindings']:
            for relative,expected in summary.get(group,{}).items():
                if relative not in checked:checked[relative]=digest(ROOT/relative)
                if checked[relative]!=expected:raise ValueError('new stage binding changed: '+relative)
    if not replay['all_replay_gates_pass'] or not replay['all_declared_candidates_replayed']:raise ValueError('incomplete or failed replay')
    output.mkdir(parents=True)
    handoff=ROOT/'docs/MOND_OBSERVATION_ATLAS_GOAL.md';(output/'prior-goal-handoff.md').write_bytes(handoff.read_bytes())
    native_rows=csv_rows(BASE/RUNS[0]/'galaxies.csv');replay_rows=csv_rows(BASE/RUNS[2]/'galaxies.csv')
    by_native={r['galaxy']:r for r in native_rows};by_replay={r['galaxy']:r for r in replay_rows}
    readiness=[]
    for old in csv_rows(previous/'pilot-readiness.csv'):
        name=old['galaxy'];n=by_native[name];r=by_replay[name]
        readiness.append(dict(**old,native_spectral_history_directly_mapped=n['direct_channel_mapping']=='True',
            retained_historical_continuum_candidate_channels=n['retained_historical_continuum_channels'] or None,
            selected_plane_preprocessing_replay_pass=r['replay_pass']=='True' if r['replay_pass'] else None,
            exact_visibility_covariance_recovered=False,historical_candidates_certified_line_free=False))
    write_csv(output/'pilot-readiness.csv',readiness)
    table='\n'.join(f"| {n['galaxy']} | {n['stored_channels']} | {n['retained_historical_continuum_channels'] if n['direct_channel_mapping']=='True' else 'unresolved'} | {float(by_replay[n['galaxy']]['native_median_inner_outer_scale_ratio']):.3f} | {float(by_replay[n['galaxy']]['coarse_median_inner_outer_scale_ratio']):.3f} |" if by_replay[n['galaxy']].get('native_median_inner_outer_scale_ratio') else f"| {n['galaxy']} | {n['stored_channels']} | {n['retained_historical_continuum_channels'] if n['direct_channel_mapping']=='True' else 'unresolved'} | — | — |" for n in native_rows)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_mond_atlas*.py')
    log=io.StringIO();tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (output/'validation.log').write_text(log.getvalue(),encoding='utf-8',newline='\n')
    if not tests.wasSuccessful():raise RuntimeError(log.getvalue())
    text=f'''# MOND atlas: recover the native spectral processing

**The original continuum-subtraction choices can be traced directly for nine
galaxies. Eight retain 29 candidate continuum channels in the released cubes.**
The selected native planes now reproduce our earlier smoothed cube planes
exactly, including their old background offsets. This closes a concrete part
of the instrument-processing chain; it does not validate a galaxy gravity fit.

The preceding report showed why a fixed sky annulus was not reliable as pure
background. The new check uses the observing pipeline's historical channel
choices instead of choosing apparently quiet channels from our current residuals.
These are historical continuum-fit candidates, not certified emission-free data.

## What was recovered

The FITS HISTORY records contain the channel weights used by UVLIN, its polynomial
order, and the IMAGR channel range and averaging settings. The
[AIPS imaging documentation](https://www.aips.nrao.edu/CookHTML/CookBookse50.html)
defines those range and increment controls. Direct mapping is accepted here only
for one unambiguous UVLIN/IMAGR chain, matching dataset identifiers, no channel
averaging, the expected output channel count, and complete single-IF weights.

DDO154's output interval contains none of its recorded continuum-fit channels.
That does not prove that every output channel contains HI. NGC2841 and NGC3521
were assembled from separate spectral cubes. NGC7331 combines histories with
different continuum fits. Those mappings remain unresolved rather than being
assigned the last header's fit across the whole cube.

The [THINGS measurement paper](https://arxiv.org/abs/0810.2125) distinguishes its
standard cubes for noise-dependent selection from rescaled, blanked products for
flux measurements. It describes continuum removal using selected channels and
emission detection after spatial smoothing. These processing distinctions matter
when building both baryonic source maps and motion likelihoods.

## A spatial comparison on the retained candidate channels

The native robust scale is Gaussian-normalized median absolute deviation (MAD),
measured inside 300 arcsec and at 550–680 arcsec from the FITS reference pixel in
the image projection plane. Native pixels are sampled every fourth position.
Each row below gives the median across that galaxy's available candidate channels.
The coarse comparison uses the same channel identities after the original
additional smoothing and 8×8 block average.

| Galaxy | Stored channels | Retained historical candidates | Native center/outer MAD | Smoothed center/outer MAD |
|---|---:|---:|---:|---:|
{table}

Native medians span 0.969–1.030. Several smoothed medians are around 1.10–1.16.
This is a descriptive spatial-scale pattern, not a significance calculation.
Smoothing reduces the number of independent measurements and can expose weak
extended emission or correlated instrumental structure. The small candidate
channel counts, same-observation dependence and correlated spatial samples do
not support assigning the difference to any one cause yet.

No numerical pass threshold was selected for these scale ratios. The next test
must simulate finite-sample stationary noise through the same processing and
measure recovery of injected signals before treating these differences as
evidence of nonstationarity or emission. Agreement in these candidates does not
validate every other channel or the full spatial/channel covariance.

## The preprocessing check is exact on its stated scope

Twenty-nine candidate planes were read from the hashed native cubes, filtered
with the recorded extra Gaussian using zero extension and float32 intermediate
planes, block-averaged, and given the original per-channel offset subtraction.
All 29 reproduce the cached planes with maximum absolute difference **zero**.
Independent direct-convolution, impulse-flux, centroid and block-flux tests check
the new implementation. This numerical equality is specific to these planes;
it is not a claim that the old physical beam approximation was complete.

The first replay stopped after eight successful planes because its declared
spectral contract only accepted FELO-HEL headers. The repair adds the native
radio-velocity and frequency formats already present in the inputs, using only
their monotonic index ordering. The [spectral FITS reference](https://arxiv.org/abs/astro-ph/0507293)
distinguishes those coordinate types. The original incomplete run and its
[failure receipt](../mond-atlas-preprocessing-replay-001/execution-failure.json)
are retained. No smoothing threshold, selected channel or replay tolerance changed.

The old extra circular smoothing does not make a native elliptical beam exactly
circular. Intensity units remain Jy per native beam. A restoring beam is also
not a complete description of interferometer noise. Those limitations remain.

## Continuum removal introduces covariance beyond neighboring channels

[AIPS continuum subtraction](https://www.aips.nrao.edu/CookHTML/CookBookse49.html)
fits the real and imaginary visibility spectra and subtracts the fitted baseline.
We implemented a conditional linear-algebra control: for a supplied spectral
covariance C and weighted polynomial-removal operator A, the residual covariance
is A C Aᵀ. This includes covariance from the uncertain subtracted baseline and
its cross terms with any channels that helped estimate it.

For example, subtracting the mean of m independent calibration channels from
different independent output channels gives covariance I + 11ᵀ/m. Subtracting
that mean from the calibration channels themselves instead gives I − 11ᵀ/m.
Those exact limits show why calibration channels and other channels do not have
identical noise after baseline removal. A short-lag-only covariance can miss
this structure even when the input noise had no long-range correlations.

The nine directly mapped histories were each propagated under independent and
three-tap smoothed unit-variance input-noise controls: **18 conditional cases**.
Polynomial annihilation, a separate weighted least-squares calculation, the exact
constant-fit limits and 60,000 simulated correlated draws verify the algebra.
The actual visibility weights, flags and nonlinear imaging were not replayed;
these cases are not measured covariance models or new galaxy motion scores.

## Atlas status and reproducibility

Current scale remains 13,525 identity groups (not certified distinct galaxies),
175 radial baseline galaxies, 126 passing its descriptive cuts, 12 resolved seed
galaxies, 22 source-image fits and 29 conditional field runs for one galaxy.
**Zero full-field galaxy cube likelihoods are admitted.** The full goal remains
active. The previous fixed-exclusion background result remains two galaxies
passing all splits; the new diagnostic does not promote other objects to that gate.

The atlas suite passes **{tests.testsRun} tests**. This report rehashes all {len(manifest['files'])} files
of the preceding publication manifest before updating the handoff. All new
packages declared SOURCE_BLOCKED before implementation and did no new galaxy
motion fitting. Raw cubes and large fields remain outside Git. The header-text
extracts, source hashes, exact settings, per-channel diagnostics and conditional
covariance controls are retained in the linked stage directories.

```text
python scripts/run_mond_atlas_native_spectral.py --output <new-native-directory>
python scripts/run_mond_atlas_preprocessing_replay_v2.py --output <new-replay-directory>
python -m unittest discover -s tests -p "test_mond_atlas*.py" -v
```

Use new output paths. Configuration inputs deliberately bind the original stages;
replaying one stage does not silently redirect downstream inputs. Inspect
[pilot readiness](pilot-readiness.csv), [execution status](execution-status.json),
[verification](verification.json) and the [publication manifest](publication-manifest.json).

Publication remains local. The previous GitHub blob write required approval,
which the session policy does not permit. Local linked Git metadata is also
outside the writable root. No alternative write was attempted or claimed.
'''
    (output/'README.md').write_text(text,encoding='utf-8',newline='\n')
    status=copy.deepcopy(read_json(previous/'execution-status.json'))
    status.update(status='LOCAL_NATIVE_SPECTRAL_PROVENANCE_AND_PREPROCESSING_MILESTONE',
        previous_milestone=str(previous.relative_to(ROOT)),goal_turn_progress='Recovered historical continuum channels, executed native spatial diagnostics and exactly replayed 29 cached planes; useful local scientific work continues.',
        native_history_mapping_unresolved=native['unresolved_history_mapping_galaxies'],
        native_history_candidates_certified_line_free=False,visibility_domain_instrument_replay_complete=False)
    status['counts'].update(native_history_direct_mappings=9,retained_historical_continuum_candidate_channels=29,
        native_candidate_spatial_diagnostic_galaxies=8,conditional_continuum_covariance_cases=18,
        exact_selected_plane_preprocessing_replays=29,unit_tests_passed=tests.testsRun)
    status['required_remaining'][1]='Simulate finite-sample stationary noise through the verified preprocessing, test injected-signal recovery and source-selection dependence, resolve composite spectral histories, and validate native/channel/spatial covariance transfer before admitting cube likelihoods.'
    write_json(output/'execution-status.json',status)
    verification=dict(status='PROVENANCE_ALGEBRA_AND_SELECTED_PLANE_REPLAY_PASS_ADMISSION_INCOMPLETE',tests_run=tests.testsRun,
        failures=len(tests.failures),errors=len(tests.errors),prior_manifest_files_verified=len(manifest['files']),
        new_stage_bound_inputs_rehashed=len(checked),native_cube_hashes_reverified=12,
        direct_history_mappings=9,unresolved_history_mappings=3,native_candidate_channels=29,
        candidate_plane_replay_maximum_absolute_error=max(float(r['maximum_absolute_replay_error']) for r in replay_rows if r.get('maximum_absolute_replay_error')),
        exact_replayed_candidate_planes=29,all_declared_replay_gates_pass=True,prior_replay_failure_retained=True,
        new_motion_fits=0,goal_complete=False)
    write_json(output/'verification.json',verification)
    write_json(output/'publication-access.json',dict(status='PREVIOUS_WRITE_BLOCK_RETAINED',
        inherited_evidence_path=str((previous/'publication-access.json').relative_to(ROOT)),
        inherited_evidence_sha256=digest(previous/'publication-access.json'),new_write_attempt=False,published=False))
    old_handoff=(output/'prior-goal-handoff.md').read_text(encoding='utf-8')
    old_handoff=old_handoff.replace('mond-atlas-execution-009/','mond-atlas-execution-010/').replace('This supersedes execution-008 readiness','This supersedes execution-009 readiness').replace('67 passing atlas unit tests',f'{tests.testsRun} passing atlas unit tests')
    addition='''
Latest executed instrument work (execution-010): native FITS HISTORY maps the
UVLIN continuum-fit choices to stored channels for 9/12 galaxies. Eight retain
29 candidates; DDO154 retains none of its historical fit channels. NGC2841 and
NGC3521 have MCUBE composition and NGC7331 combines different fit histories; do
not propagate a single last-header covariance across those cubes. Historical
fit channels are not certified line-free or an independent observing epoch.

Native per-galaxy median center/outer MAD ratios are .969–1.030 on those 29
candidate channels. After the old smoothing/block average, several are 1.10–1.16.
No significance or cause is established: finite independent-beam counts can
matter. Next use stationary-noise and injected-signal controls through the exact
processing before calling this nonstationarity or HI leakage.

All 29 selected cached planes replay EXACTLY from native images with the recorded
Gaussian/filter/mean subtraction. The incomplete first replay is retained: it
stopped on a legacy radio-velocity header after 8 successful planes; v2 adds the
radio/frequency index-order contracts without changing data choices or gates.
Files: mond-atlas-native-spectral-001, mond-atlas-preprocessing-replay-001/002.
Eighteen conditional baseline-subtraction covariance controls use A C A^T, with
independent algebra and Monte Carlo tests. These do not replay visibility-level
weights, flags or nonlinear CLEAN and are not admitted observed likelihoods.

'''
    position=old_handoff.index('Latest finding:')
    old_handoff=old_handoff[:position]+addition+old_handoff[position:]
    handoff.write_text(old_handoff,encoding='utf-8',newline='\n')
    write_json(output/'input-bindings.json',{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),manifest_path,BASE/RUNS[0]/'summary.json',BASE/RUNS[2]/'summary.json']})
    intended={ROOT/r['path'] for r in manifest['files']};intended.add(manifest_path)
    intended.update(ROOT/'scripts'/(n+'.py') for n in ['mond_atlas_native_spectral','run_mond_atlas_native_spectral',
        'mond_atlas_preprocessing','mond_atlas_channel_order','run_mond_atlas_preprocessing_replay',
        'run_mond_atlas_preprocessing_replay_v2','report_mond_atlas_native_spectral'])
    intended.update(ROOT/'configs'/(n+'.json') for n in ['mond_atlas_native_spectral_v1','mond_atlas_preprocessing_replay_v1','mond_atlas_preprocessing_replay_v2'])
    intended.update(ROOT/'tests'/('test_mond_atlas_'+n+'.py') for n in ['native_spectral','preprocessing','channel_order'])
    for folder in [BASE/name for name in RUNS]+[output]:intended.update(p for p in folder.rglob('*') if p.is_file())
    items=[]
    for path in sorted(intended,key=lambda p:p.as_posix()):
        relative=path.relative_to(ROOT).as_posix()
        if relative.startswith('work/private/') or path.suffix.lower() in ['.fits','.npy','.npz']:raise ValueError('raw data in publication manifest')
        items.append(dict(path=relative,bytes=path.stat().st_size,sha256=digest(path)))
    write_json(output/'publication-manifest.json',dict(status='LOCAL_VALIDATED_MILESTONE_WRITES_BLOCKED_SCIENTIFIC_ADMISSION_INCOMPLETE',
        base_commit=manifest['base_commit'],goal_complete=False,ordinary_fetch_and_remote_review_required_before_publication=True,
        raw_observation_files_included=0,source_admission_complete=False,files=items,file_count=len(items),total_bytes=sum(i['bytes'] for i in items)))
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if not link.startswith('https:') and not (output/link).is_file():raise ValueError('broken report link: '+link)
    print(dict(report=str(output/'README.md'),unit_tests=tests.testsRun,manifest_files=len(items),manifest_bytes=sum(i['bytes'] for i in items),goal_complete=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report(args.output.resolve())
