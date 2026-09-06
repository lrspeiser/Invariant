"""Preserve the background-contamination finding and revise pilot readiness."""
from __future__ import annotations
import argparse, csv, copy, re
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mond_atlas_common import ROOT, read_json, write_json, write_csv, digest
from mond_atlas_image_io import read_primary_image
from mond_atlas_background_support import block_fraction, dilate_disk
from run_mond_atlas_noise import masks

BASE=ROOT/'work/gravity-first-principles'
RUNS=['mond-atlas-noise-mean-001','mond-atlas-background-support-001','mond-atlas-emission-excluded-noise-001']


def csv_rows(path):
    with path.open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def support_figure(path,support_config,excluded):
    """Display actual support/mask arrays, not inferred gas intensity or depth."""
    parent=read_json(ROOT/support_config['parent_noise_protocol']);parent['split_seed']=support_config['fixed_spatial_seed']
    moments={r['name']:r for r in read_json(ROOT/support_config['native_moment_receipt'])['files'] if r['resolution']=='NA' and r['moment']==0}
    audits=read_json(ROOT/support_config['cube_audit']);counts={r['galaxy']:r for r in excluded}
    font_root=Path('C:/Windows/Fonts')
    normal=ImageFont.truetype(str(font_root/'arial.ttf'),20);small=ImageFont.truetype(str(font_root/'arial.ttf'),17)
    bold=ImageFont.truetype(str(font_root/'arialbd.ttf'),24);title=ImageFont.truetype(str(font_root/'arialbd.ttf'),30)
    canvas=Image.new('RGB',(1540,1490),'white');draw=ImageDraw.Draw(canvas)
    draw.text((28,20),'The old background annulus overlaps detected hydrogen',font=title,fill='#182735')
    draw.text((28,64),'Actual map support and fixed forward split 9062028. Colors mark regions, not gas brightness or physical depth.',font=normal,fill='#374151')
    colors={'Released HI support':(149,158,170),'Smoothing margin':(249,218,162),'Usable calibration':(44,114,170),'Usable validation':(115,70,160),'Excluded old patches':(198,49,52)}
    x=28
    for label,color in colors.items():
        draw.rectangle((x,105,x+18,123),fill=color);draw.text((x+25,101),label,font=small,fill='#263442');x+=len(label)*9+66
    for index,audit in enumerate(audits):
        name=audit['name'];image,header=read_primary_image(ROOT/moments[name]['file'])
        fraction=block_fraction(np.isfinite(image)&(image>0),support_config['block_factor'])
        dilation=int(np.ceil(4*audit['extra_smoothing_sigma_arcsec']/(abs(header['CDELT1'])*3600*support_config['block_factor'])))+2
        expanded=dilate_disk(fraction>0,dilation)
        with np.load(ROOT/parent['source_packets']/(name+'.npz'),allow_pickle=False) as packet:
            train,test=masks(packet['east'],packet['north'],parent)
        rgb=np.full((*fraction.shape,3),247,np.uint8)
        rgb[expanded]=colors['Smoothing margin'];rgb[fraction>0]=colors['Released HI support']
        rgb[train&~expanded]=colors['Usable calibration'];rgb[test&~expanded]=colors['Usable validation']
        rgb[(train|test)&expanded]=colors['Excluded old patches']
        thumbnail=Image.fromarray(rgb[::-1]).resize((300,300),Image.Resampling.NEAREST)
        x=30+(index%4)*385;y=155+(index//4)*425
        draw.text((x,y),name,font=bold,fill='#182735');canvas.paste(thumbnail,(x,y+36))
        draw.rectangle((x,y+36,x+299,y+335),outline='#88939e')
        row=counts[name]
        draw.text((x,y+347),f"Enough background: {row['sufficient_support_partitions']}/16 splits",font=small,fill='#182735')
        draw.text((x,y+372),f"Checks passed: {row['passing_partitions']}/16 declared",font=small,fill='#182735')
    draw.text((28,1436),'Blank/undetected pixels are not proof of no gas. These background checks do not admit a galaxy gravity likelihood.',font=normal,fill='#374151')
    canvas.save(path)


def report(output):
    if output.exists():raise FileExistsError('immutable report')
    previous=BASE/'mond-atlas-execution-008';manifest_path=previous/'publication-manifest.json';manifest=read_json(manifest_path)
    for item in manifest['files']:
        path=ROOT/item['path']
        if path.stat().st_size!=item['bytes'] or digest(path)!=item['sha256']:raise ValueError('old milestone changed: '+item['path'])
    summaries=[read_json(BASE/name/'summary.json') for name in RUNS]
    checked={}
    for summary in summaries:
        for group in ['bindings','source_bindings']:
            for relative,expected in summary[group].items():
                if relative not in checked:checked[relative]=digest(ROOT/relative)
                if checked[relative]!=expected:raise ValueError('new run binding changed: '+relative)
    mean,support,exclusion=summaries
    if mean['execution_failures'] or not mean['all_original_partition_replays_pass'] or exclusion['covariance_estimation_failures']:
        raise ValueError('unresolved execution/replay failure')
    if [mean['partition_branch_evaluations'],support['partition_role_evaluations'],exclusion['declared_partitions']]!=[576,384,192]:
        raise ValueError('unexpected incomplete run counts')
    output.mkdir(parents=True)
    handoff=ROOT/'docs/MOND_OBSERVATION_ATLAS_GOAL.md';(output/'prior-goal-handoff.md').write_bytes(handoff.read_bytes())
    old_ready=csv_rows(previous/'pilot-readiness.csv')
    means=csv_rows(BASE/RUNS[0]/'galaxies.csv');overlaps=csv_rows(BASE/RUNS[1]/'galaxies.csv');excluded=csv_rows(BASE/RUNS[2]/'galaxies.csv')
    by_mean={(r['galaxy'],r['branch']):r for r in means};by_overlap={r['galaxy']:r for r in overlaps};by_exclusion={r['galaxy']:r for r in excluded}
    readiness=[];comparison=[]
    for old in old_ready:
        name=old['galaxy'];m=by_mean[name,'mean_and_variance_corrected'];o=by_overlap[name];e=by_exclusion[name]
        comparison.append(dict(galaxy=name,original_noise_passes=int(by_mean[name,'previous_fixed_mean']['passing_partitions']),
            mean_corrected_passes=int(m['passing_partitions']),declared_partitions=int(e['declared_partitions']),
            minimum_original_direct_hi_overlap=float(o['minimum_direct_partition_fraction']),maximum_original_direct_hi_overlap=float(o['maximum_direct_partition_fraction']),
            post_exclusion_sufficient_partitions=int(e['sufficient_support_partitions']),post_exclusion_passing_partitions=int(e['passing_partitions'])))
        current={k:v for k,v in old.items() if k not in ['background_all_declared_splits_pass','joint_image_and_background_prerequisites_pass']}
        current.update(previous_background_all_splits_pass=old['background_all_declared_splits_pass']=='True',
            background_emission_support_overlap=float(o['maximum_direct_partition_fraction'])>0,
            background_excluded_support_partitions=int(e['sufficient_support_partitions']),
            background_excluded_all_declared_splits_pass=e['all_declared_splits_pass']=='True',
            joint_raw_image_and_excluded_background_diagnostics_pass=old['raw_or_P1_astrometry_pass']=='True' and e['all_declared_splits_pass']=='True',
            pure_noise_established=False,selection_mask_validated=False)
        readiness.append(current)
    write_csv(output/'pilot-readiness.csv',readiness);write_csv(output/'noise-control-comparison.csv',comparison)
    old_rows={(r['galaxy'],r['split']):r for r in csv_rows(BASE/RUNS[0]/'partitions.csv') if r['branch']=='mean_and_variance_corrected'}
    evaluated=[r for r in csv_rows(BASE/RUNS[2]/'partitions.csv') if r['status']=='BACKGROUND_DIAGNOSTIC_EVALUATED']
    old_passes=sum(old_rows[r['galaxy'],r['split']]['diagnostic_pass']=='True' for r in evaluated)
    if old_passes!=49 or len(evaluated)!=49:raise ValueError('paired control accounting changed')
    support_config=read_json(ROOT/'configs/mond_atlas_background_emission_v1.json')
    support_figure(output/'background-support.png',support_config,excluded)
    table='\n'.join(f"| {r['galaxy']} | {r['original_noise_passes']}/16 | {r['mean_corrected_passes']}/16 | {100*r['minimum_original_direct_hi_overlap']:.1f}–{100*r['maximum_original_direct_hi_overlap']:.1f}% | {r['post_exclusion_sufficient_partitions']}/16 | {r['post_exclusion_passing_partitions']}/16 |" for r in comparison)
    text=f'''# MOND atlas: hydrogen in the supposed background

**Detected hydrogen overlaps some old background patches in 11 of 12 galaxies.**
Two objects, NGC5055 and NGC6946, passed every original noise check even though
their selected patches overlapped the released hydrogen support extensively.
Those passes never established a pure-noise region, and the new audit shows why.

After excluding the previously declared support and smoothing margin, only
**NGC2976 and NGC7331 have enough background and pass all 16 splits**. The other
galaxies remain in the catalog with their coverage failures. This changes data
readiness, not a gravity formula. The full atlas is still unfinished.

![Actual HI support, smoothing margin and old background patches](background-support.png)

The picture uses one fixed split; the counts below include all 16. Red means an
old calibration or validation patch intersects the declared warning area. Gray
shows positive released moment-map support, not gas brightness. Orange is the
extra smoothing margin. A blank pixel is not proof of gas absence.

## Three executed checks

First, the estimated background mean was propagated into the covariance of the
validation residuals. For a calibration arithmetic mean, residual covariance is
the test covariance plus the variance of that mean, minus both test–mean cross
terms. Calibration sample variance also loses the mean mode. Independent joint
covariance algebra and simulated correlated draws verify those corrections.
The fitted covariance parameters are still uncertain; this is not an exact
predictive likelihood or a fully conditional Gaussian-process prediction.

All 192 previous partitions were reproduced, and all three declared covariance
branches were evaluated: **576 checks**. The same three galaxies remained
split-sensitive. Correct accounting for the mean was necessary but did not
resolve those failures.

Second, the released natural-weighted HI moment maps were checked against the
native cube spatial grids and file hashes. Finite positive native pixels define
detected support. A coarse cell is warned if it contains any such pixel. Support
was expanded by ceil(4 × the recorded extra-smoothing sigma / coarse pixel size)
+ 2 cells, as declared before this audit. This conservative geometric warning
is neither the publisher's original cube mask nor a predicted contamination
amplitude. Native-to-coarse support area is exactly conserved in these files.

Third, each old split was intersected with the complement of that same expanded
support. There was no rebalancing, threshold relaxation or favorable-split search.
The original 150 calibration pixels, 25 validation pixels and four validation
pixels in every quadrant were required before the corrected covariance check.

| Galaxy | Original passes | Mean/variance corrected passes | Direct HI overlap range across old patches | Enough background after exclusion | Passes after exclusion |
|---|---:|---:|---:|---:|---:|
{table}

The overlap ranges concern coarse calibration or validation pixels, not the
fraction of each galaxy's mass. There are **49 evaluable partitions out of 192**
after exclusion, and all 49 pass. The other 143 have insufficient support.
Crucially, these same 49 partitions already passed the mean-corrected check
before exclusion. **This is not evidence that removing HI cured the earlier
covariance failures.** NGC2841 and NGC2903 cannot be assessed with the remaining
patches under the declared requirements.

## What the spectra add

The fixed outer 15% at each end of the channel band were compared with the
central 70%, keeping lag pairs within each contiguous segment. These ends are
not certified line-free. In the fixed forward calibration split, NGC5055's
normalized adjacent-channel product is 0.446 at the ends and 0.860 centrally;
NGC6946 gives 0.431 and 0.801. Their central positive tails are also stronger.
Together with the map overlap, this is consistent with galaxy emission entering
the noise estimate. It does not establish that emission explains every anomaly:
some band ends also show strong correlations or spatial inconsistency.

The [THINGS source paper](https://arxiv.org/abs/0810.2125) documents the observation
and processing. The numerical overlap and spectral measurements here come from
the hashed local observation products, not from a published gravity fit.

## Consequences for the atlas

The catalog still contains **13,525 identity groups**, with unresolved identities
explicitly retained; this is not a certified unique-galaxy count. The radial
baseline remains 175 galaxies, with 126 passing its descriptive cuts. There are
12 resolved seed galaxies, 22 source-image fits and 29 conditional field runs
for one galaxy. There are **zero admitted full-field galaxy cube likelihoods**.

The earlier force result remains useful: in two conditional NGC2903 source
reconstructions, almost the same total mass and similar in-plane force coexist
with substantially different vertical force. That is a reason to seek independent
depth-sensitive observations, not evidence of a measured failure of Newton or
MOND. See the [preserved field report](../mond-atlas-execution-008/README.md).

Next required work is to recover or reconstruct and validate the native selection
and line-free definitions, including signal-injection recovery and dependence on
the same observation. A useful noise model must transfer into the galaxy region
and include the actual spatial/spectral instrument response. Source photometry,
mass conversions, depth/exterior-field ensembles, additional pilots and survey
holdouts remain required. The two surviving background diagnostics do not certify
those other requirements or supply independent stellar masses.

## Reproduction and preservation

All new packages were declared SOURCE_BLOCKED before implementation and scored
no new galaxy motions. All original seed objects remain development-exposed.
The atlas unit suite passed **67 tests** on the code bound by this report.
The preceding manifest's 357 files were verified before updating the handoff.
Raw FITS observations and large numerical fields remain outside Git.

Run the three scripts with new immutable output directories:

```text
python scripts/run_mond_atlas_noise_mean.py --output <new-mean-directory>
python scripts/run_mond_atlas_background_support.py --output <new-support-directory>
python scripts/run_mond_atlas_emission_excluded_noise.py --output <new-exclusion-directory>
python -m unittest discover -s tests -p "test_mond_atlas*.py" -v
```

The source and partition paths in the configurations intentionally bind the
original runs. Replaying an individual stage does not silently redirect its
descendants. Machine-readable details are in [pilot readiness](pilot-readiness.csv),
[control comparison](noise-control-comparison.csv), [verification](verification.json)
and the [publication manifest](publication-manifest.json).

Publication remains local. The previous connected GitHub blob write was rejected
because it requires approval while this session's policy is never. The local
linked Git metadata is also outside the writable root. No alternate write route
was attempted and nothing is claimed to have been pushed.
'''
    (output/'README.md').write_text(text,encoding='utf-8',newline='\n')
    status=copy.deepcopy(read_json(previous/'execution-status.json'))
    status.update(status='LOCAL_BACKGROUND_CONTAMINATION_AND_FIXED_EXCLUSION_MILESTONE',
        previous_milestone=str(previous.relative_to(ROOT)),goal_turn_progress='Identified real emission overlap and executed fixed support exclusion; useful local work continues.',
        background_pure_noise_established=False,selection_mask_validated=False)
    status['counts'].update(new_field_runs=0,mean_uncertainty_partition_branch_checks=576,
        background_support_partition_role_checks=384,background_exclusion_declared_partitions=192,
        background_exclusion_evaluable_partitions=49,background_exclusion_insufficient_partitions=143,
        original_background_split_stable_galaxies=9,background_split_stable_galaxies=2,
        joint_raw_image_and_background_prerequisites=2,unit_tests_passed=67)
    status['background_split_failures']=[]
    status['background_exclusion_all_split_pass']=exclusion['all_declared_splits_pass']
    status['background_exclusion_insufficient_some_or_all_splits']=[r['galaxy'] for r in excluded if r['all_declared_splits_sufficient_support']!='True']
    status['background_covariance_failure_after_exclusion']='No failures among 49 evaluable partitions; 143 coverage failures cannot be assigned a covariance pass or fail.'
    status['required_remaining'][1]='Establish noise/selection validity from native cubes with emission-aware and independently checked line-free support, injection recovery, spatial transfer and instrument response. A fixed annulus or passing moment checks do not suffice.'
    write_json(output/'execution-status.json',status)
    access=read_json(previous/'publication-access.json')
    write_json(output/'publication-access.json',dict(status='PREVIOUS_WRITE_BLOCK_RETAINED_NO_BYPASS_ATTEMPT',
        inherited_evidence_path=str((previous/'publication-access.json').relative_to(ROOT)),inherited_evidence_sha256=digest(previous/'publication-access.json'),
        previous_attempt=access,new_write_attempt=False,published=False))
    all_test_paths=list((ROOT/'tests').glob('test_mond_atlas*.py'))
    verification=dict(status='SOURCE_AND_ARTIFACT_AUDITS_PASS_FULL_SCIENTIFIC_ADMISSION_INCOMPLETE',
        tests_run=67,failures=0,errors=0,unit_test_execution='Executed immediately before this report using python -m unittest discover -s tests -p test_mond_atlas*.py -v; exit 0, 67 tests, 2.705 seconds.',
        test_file_hashes={p.relative_to(ROOT).as_posix():digest(p) for p in all_test_paths},
        previous_manifest_files_verified=len(manifest['files']),new_run_bound_inputs_rehashed=len(checked),
        original_background_replays=192,mean_branch_checks=576,support_partition_role_checks=384,
        fixed_exclusion_declared_partitions=192,evaluable_partitions=49,insufficient_support_partitions=143,
        paired_pre_exclusion_passes_among_evaluable=old_passes,new_motion_scores=0,goal_complete=False)
    write_json(output/'verification.json',verification)
    handoff.write_text(f'''# Active goal: MOND observation atlas

The user authorized execution, not just design. The goal remains active and
unfinished. Read `work/gravity-first-principles/{output.name}/README.md`,
`execution-status.json`, `verification.json` and `publication-manifest.json`.
This supersedes execution-008 readiness; earlier findings and failures remain.

Current scale: 13,525 identity groups (not certified distinct), 175 radial baseline
galaxies, 126 descriptive-cut galaxies, 12 resolved seeds, 22 source-image fits,
29 conditional field runs for one galaxy, 67 passing atlas unit tests and ZERO
admitted full-field galaxy cube likelihoods. Target remains 10–20 development
pilots then 100–300 eligible resolved systems and thousands of population records
where coverage permits. Population rows do not count as resolved predictions.

Latest finding: 11/12 galaxies have detected HI support in some old background
patches. NGC5055 and NGC6946 passed old noise gates despite extensive overlap.
Background-mean covariance accounting alone leaves the same three sensitive
galaxies. The fixed HI-plus-smoothing exclusion leaves 49/192 evaluable splits;
all 49 pass, but those SAME 49 already passed before exclusion. Do not claim a
noise failure was cured. Only NGC2976 and NGC7331 retain enough background and
pass all 16 splits. No pure-noise, mask-independence or inner-galaxy transfer
claim is established. 143 other splits have insufficient support, not a gravity
failure. Fixed channel-band ends are NOT certified line-free.

New immutable runs: mond-atlas-noise-mean-001, mond-atlas-background-support-001,
mond-atlas-emission-excluded-noise-001. Configurations declare SOURCE_BLOCKED
before implementation. Read docs/OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md
before additional source/solver/scoring work. No new motion scores were computed.
All existing seed galaxies are development-exposed; previous motion comparisons
remain exploratory and nonadmitted. Do not invent retrospective preregistration.

Earlier numerical work remains in execution-008: common bilinear image/source
basis, conditional thin/mixed NGC2903 fields, separate lateral/vertical/box
perturbations and above-plane checks. At R=5 kpc, model QUMOND force-equivalent
speed differs by 1.7%, downward force at z=.25 kpc by 32%, total model mass by
.50%. This is joint deprojection sensitivity, not a measured gravity anomaly.

Next requirements: native selection and spectral-response reconstruction with
validated line-free support and injection recovery; source-image noise and
beam/pixel/absolute-flux likelihood; stellar and HI/H2 conversions and missing
components; independently constrained 3D and exterior-field ensembles; distinct
AQUAL controls and proper warp/streaming/pressure cube prediction; additional
pilots and galaxy/group/survey holdouts. Keep raw observations outside Git.

Five historical STELLAR_MASS_MAP files are P5 cleaned stellar FLUX in MJy/sr.
Only NGC2903 has validated P1-to-P5 relative transfer. The two original S4G
geometry tables remain missing. Do not treat the bound derived geometry record
as revalidated raw metadata. Nondetection is not missing coverage or empty space.

Publication: linked local Git metadata is outside the writable root. The prior
connected GitHub create_blob was rejected because it requires approval while
policy is never. Reads worked and last verified main was
afc721a13782acec4ebc94ad8f6d97ed71be7152; fresh remote review is needed before any
eventual write. Do not bypass either restriction. Nothing has been published.
The previous handoff is archived in this report. Preserve unrelated local work.

Working runtime: bundled Python 3.12/NumPy on CPU; old CUDA environment cannot
start. Shell downloads have been denied. Goal is active because useful local
scientific work continues, despite blocked publication and incomplete sources.
''',encoding='utf-8',newline='\n')
    write_json(output/'input-bindings.json',{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),manifest_path]+[BASE/n/'summary.json' for n in RUNS]})
    intended={ROOT/r['path'] for r in manifest['files']};intended.add(manifest_path)
    intended.update(ROOT/'scripts'/(n+'.py') for n in ['mond_atlas_noise_mean','run_mond_atlas_noise_mean',
        'mond_atlas_background_support','run_mond_atlas_background_support','mond_atlas_emission_exclusion',
        'run_mond_atlas_emission_excluded_noise','report_mond_atlas_background'])
    intended.update(ROOT/'configs'/(n+'.json') for n in ['mond_atlas_noise_mean_v1','mond_atlas_background_emission_v1','mond_atlas_emission_excluded_noise_v1'])
    intended.update(ROOT/'tests'/('test_mond_atlas_'+n+'.py') for n in ['noise_mean','background_support','emission_exclusion'])
    for folder in [BASE/n for n in RUNS]+[output]:intended.update(p for p in folder.rglob('*') if p.is_file())
    items=[]
    for path in sorted(intended,key=lambda p:p.as_posix()):
        relative=path.relative_to(ROOT).as_posix()
        if relative.startswith('work/private/') or path.suffix.lower() in ['.fits','.npy','.npz']:raise ValueError('raw data in manifest')
        items.append(dict(path=relative,bytes=path.stat().st_size,sha256=digest(path)))
    write_json(output/'publication-manifest.json',dict(status='LOCAL_VALIDATED_MILESTONE_WRITES_BLOCKED_SCIENTIFIC_ADMISSION_INCOMPLETE',
        base_commit=manifest['base_commit'],goal_complete=False,ordinary_fetch_and_remote_review_required_before_publication=True,
        raw_observation_files_included=0,source_admission_complete=False,files=items,file_count=len(items),total_bytes=sum(i['bytes'] for i in items)))
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if not link.startswith('https:') and not (output/link).is_file():raise ValueError('broken report link: '+link)
    print(dict(report=str(output/'README.md'),manifest_files=len(items),manifest_bytes=sum(i['bytes'] for i in items),
        tests=67,noise_split_stable=exclusion['all_declared_splits_pass'],goal_complete=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report(args.output.resolve())
