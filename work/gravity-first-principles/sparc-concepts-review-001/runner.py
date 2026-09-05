"""Review previously exposed SPARC receipts without new fitting or target access."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
base=Path(__file__).parent; root=base/'Invariant'; outputs=base.parent/'outputs'
dest=root/'work/gravity-first-principles/sparc-concepts-review-001'
dest.mkdir(exist_ok=False)
names={
    'Acceleration-only adjustment':'universal-galaxy-law-construction-v1',
    'Local brightness-gradient adjustment':'universal-galaxy-law-construction-v2-photometric',
    'Signed nonlocal profile correction':'universal-galaxy-law-construction-v3-nonlocal-profile',
    'Interior matter / exterior vacancy focusing':'conditional-formula-generator-v4'}
summaries=[]; all_rows=[]; inputs=[]
for label,name in names.items():
    path=root/f'runs/gravity/g4/{name}.json'
    data=json.loads(path.read_text())
    inputs.append(dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    rows=[]
    for galaxy in data['galaxies']:
        before=float(galaxy['rar_score']['chi_square']); after=float(galaxy['candidate_score']['chi_square'])
        rows.append(dict(model=label,galaxy=galaxy['galaxy'],rar_chi_square=before,
            candidate_chi_square=after,absolute_gain=before-after,fractional_gain=(before-after)/before))
    total_before=sum(r['rar_chi_square'] for r in rows); total_after=sum(r['candidate_chi_square'] for r in rows)
    assert abs(total_after-float(data['scores']['candidate_chi_square']))<.001
    largest=max(rows,key=lambda r:r['absolute_gain'])
    summaries.append(dict(model=label,galaxies=len(rows),improved=sum(r['absolute_gain']>0 for r in rows),
        aggregate_gain=(total_before-total_after)/total_before,
        median_galaxy_gain=statistics.median(r['fractional_gain'] for r in rows),
        largest_gain_galaxy=largest['galaxy'],largest_share_of_net_gain=largest['absolute_gain']/(total_before-total_after),
        aggregate_gain_without_largest=(total_before-total_after-largest['absolute_gain'])/(total_before-largest['rar_chi_square']),
        receipt_decision=data['decision']))
    all_rows.extend(rows)
report_paths=[
    'docs/GRAVITY_G3_META_LAW_V2_RESULT.md','docs/GRAVITY_G4_FIRST_PRINCIPLES_MECHANISM_SEARCH_RESULT.md',
    'docs/GRAVITY_G4_AUXILIARY_ACTION_DERIVATION_RESULT.md','docs/GRAVITY_G4_CLUSTER_LENSING_EXPLORATION_RESULT.md',
    'docs/GRAVITY_ITEM3_SMOOTH_DENSITY_RESULT.md','docs/GRAVITY_ITEM56_DISK_GALAXY_GATE_RESULT.md',
    'docs/GRAVITY_ITEM61_CROSS_SCALE_GATE_RESULT.md',
    'runs/gravity/open-gravity-sparc-139-environmental-generalization-v1/receipt.json',
    'work/gravity-first-principles/sigmagravity-import-002/source-snapshots/frontiers/docs/FORMULA_SCORECARD.md']
for name in report_paths:
    inputs.append(dict(path=name,sha256=hashlib.sha256((root/name).read_bytes()).hexdigest()))
result=dict(scope='Review of exposed historical receipts and reports; no new formula fits or reserved observations opened.',
    summaries=summaries,inputs=inputs,novel_physical_laws_admitted=0)
(dest/'review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(dest/'runner.py').write_bytes(Path(__file__).read_bytes())
for path in [dest/'per_galaxy_comparison.csv',outputs/'SPARC-existing-formula-comparison.csv']:
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
table='\n'.join(f"| {s['model']} | {100*s['aggregate_gain']:.2f}% better | {s['improved']}/{s['galaxies']} | {100*s['median_galaxy_gain']:+.2f}% |" for s in summaries)
focus=summaries[-1]
report=f'''# SPARC: existing evidence and initial concepts for review

We should build from the existing work, rather than start a new small pilot. The reviewed Invariant development record includes 139 SPARC galaxies and 2,720 radial measurements, their gas/disk/bulge mass-model components, and a matching published surface-brightness supplement. Historical runs record 35 confirmation galaxies as unopened at those run dates; that is not a new repository-wide certification that every later branch left them untouched. The imported Sigma record also contains a different 131-galaxy, 968-outer-point assessment. Those populations and metrics must not be pooled or called independent evidence merely because they have different names.

This review reads exposed scores and source summaries only. It does not rerun fits, open confirmation responses, certify every historical file, or claim exhaustive review of the repository. A machine-readable input/hash inventory and per-galaxy score table accompany it.

## What the existing galaxy results actually say

The table compares recorded squared, uncertainty-weighted velocity errors with the same RAR benchmark. A percentage is an improvement in that score, not a percentage of galaxies explained or a probability that the theory is correct. The last column is the median of the per-galaxy fractional score changes; positive means better.

| Historical formula | Aggregate score | Galaxies improved | Median galaxy improvement |
|---|---:|---:|---:|
{table}

The most eye-catching aggregate result, interior/exterior focusing, improves only {focus['improved']} galaxies. {focus['largest_gain_galaxy']} supplies {100*focus['largest_share_of_net_gain']:.1f}% of its net improvement. Removing that galaxy changes the aggregate improvement to {100*focus['aggregate_gain_without_largest']:.2f}%. This is an influence diagnostic, not a reason to discard the galaxy. Its source mass model, inclination, distance, and uncertainty model deserve inspection alongside other successes and failures. The median galaxy is slightly worse. Therefore the record does not establish a broadly successful extra-attraction law.

The signed nonlocal profile model used both matter inside a radius and structure outside it. The focusing model used the combination of substantial interior starlight and relatively sparse exterior starlight. Both are relevant to the user's overlapping-range intuition. Neither demonstrates that empty space causes gravity, or that a disk obeys the spherical-shell experiment. Disk geometry already affects Newtonian gravity and must be computed correctly.

The G3 learned baryonic-structure model also reported a 5.54% improvement, but it was adjusted after earlier fold results and was not a compact first-principles law. Across these campaigns, model selection repeatedly used an already explored development population. Whole-galaxy folds help, but do not erase that history.

## Three initial concepts

### 1. Signed response to the surrounding matter profile

Plain language: the surroundings may redistribute the pull, sometimes increasing it and sometimes reducing it. An always-positive boost cannot fix galaxies where the baseline already predicts too much speed.

Existing evidence: the signed nonlocal correction improved the aggregate score by 8.18%; the always-positive focusing winner concentrated its gains in a minority of galaxies. The new action prototype independently shows that its required reaction term can weaken gravity and spread a redistribution's effect beyond its support. These are motivations to compare, not proof that the prototype is the mechanism behind the SPARC residuals.

Illustrative development equation:

    g_trial(r) = g_RAR(r) + a0 W(g_bar/a0) F(profile_inside, profile_outside)

F is allowed to be positive or negative; W suppresses the correction at high acceleration. This is a radial diagnostic template, not a complete field equation. The old signed kernels are the starting comparators, not new inventions. We should change the shape only after locating reproducible radial residual patterns, and preserve finite positive total predictions.

First question: do models systematically miss the inner-to-outer transition, or do their errors mostly reflect whole-galaxy mass/distance calibration? These require different repairs.

### 2. More than one response range

Plain language: a nearby region and a broader region could contribute differently. The pull need not be determined by just the matter at the star's exact location.

Existing evidence: the useful historical profiles combined interior and exterior kernels with different reaches. A fold-stable cross-scale parent was only 6.06% worse than RAR, while the broad mechanism selector was 48.34% worse. This makes a small, interpretable family more informative than another enormous unconstrained search.

Illustrative template:

    F = A F_near + B F_far

We would compare one range with two and ask whether the second range improves many galaxies, rather than only reducing the worst few residuals. A,B and the range rule are shared across galaxies. The old kernels use logarithmic radius ratios; the new Helmholtz prototype uses a physical length. These are distinct models. We must not silently treat their scale parameters as equivalent. Any range tied to galaxy size must be computed from the measured source and ultimately derived, not fitted separately per galaxy. The fixed linear overlapping-range example still has the wrong universal mass scaling unless a collective scale emerges.

### 3. A physical control on the extra response's strength

Plain language: the same mechanism might be weak in organized disks and stronger in diffuse multi-center systems, but it needs to recognize a measurable physical difference rather than an object label.

Existing evidence: the historical cross-scale kernel's cluster diagnostic preferred beta=2, compared with beta=0.5 for its galaxy parent. A separate cluster-selected beta=1.5 formula badly overpredicted the SPARC population (equal-galaxy loss 625.410 versus RAR 33.556). These are different historical campaigns, not interchangeable fitted constants. The cluster-lensing acceleration tables were model-dependent, and later cluster source-geometry problems further limit claims. They are leads about failed transfer, not proof of the required physical factor.

Illustrative template:

    strength = shared_maximum * bounded_function(measured_source_state)

Simple density and potential/tidal repairs have already failed important tests. In the 139-galaxy environmental replay, no tested repair produced the required generalization signal; three density-dependent candidates were source-blocked rather than tested. Thus we should not merely repeat the same logistic functions with different coefficients. Coherence is a secondary candidate only if it is independently measured and has consistent source reaction; using the rotation speeds we are predicting as its input would undermine the test.

## How the data should be allowed to change the formulas

Use the existing 139 galaxies as openly labeled development data. Create a residual atlas showing inner, transition, and outer errors; gas-rich versus star-dominated structure; and sensitivity to distance, inclination, stellar mass-to-light ratio, thickness, and missing gas. Changes to a universal formula should be motivated by a repeated source-linked pattern that survives those uncertainty checks.

Track pooled error, equal-galaxy error, median improvement, the fraction of galaxies helped, and worst population regressions together. The 18.76% example shows why a single total is insufficient. Keep successful, unsuccessful, and unresolved objects visible. Do not remove UGC02953 or any other influential object simply to make a preferred conclusion look cleaner.

Record each revision's physical rationale and expose which data motivated it. After choosing a small family, evaluate whole galaxies excluded from that fitting step; call this development generalization because historical exposure remains. Independent confirmation requires a separate exposure audit and a frozen formula. Do not claim that these gas rotation curves directly establish outer stellar dynamics, lensing, or Solar System success.

## Recommendation for the review

Start with concepts 1 and 2 together: a signed, source-dependent response with one versus two ranges. Keep the existing best focusing and signed-profile laws as unchanged comparators, alongside RAR. Concept 3 is the cross-scale follow-up, not permission to assign separate galaxy and cluster constants. The objective is to learn which aspects of the matter profile explain repeatable errors, then derive a consistent field law that produces that behavior.

The more flexible NFW comparison remains a useful performance benchmark, but its object-specific parameters differ from a universal law's parameter budget. Failing the old numerical ceiling is an unmet project gate, not a theorem that every modified-gravity mechanism is impossible.

No new formula has been fitted or accepted in this review. All three concepts remain proposals for discussion.
'''
(dest/'review.md').write_text(report,encoding='utf-8')
(outputs/'SPARC-initial-concepts-review.md').write_text(report,encoding='utf-8')
print(json.dumps(summaries,indent=2))
