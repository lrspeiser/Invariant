"""Assemble the executed atlas milestone, readiness gates and offline galaxy viewer."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mond_atlas_common import ROOT, PROTOCOL, canonical_name, digest, read_json, write_csv, write_json


def assemble(catalog, radial, numerics, output):
    catalog,radial,numerics,output = [p.resolve() for p in (catalog,radial,numerics,output)]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    c,r,n = read_json(catalog/"summary.json"),read_json(radial/"summary.json"),read_json(numerics/"validation.json")
    legacy_path=ROOT/"work/gravity-first-principles/constrained-matter-findings-001/summary.json"
    legacy=read_json(legacy_path)
    with (catalog/"galaxies.csv").open(encoding="utf-8") as stream:
        catalog_rows=list(csv.DictReader(stream))
    pilot=[x for x in catalog_rows if x["has_local_hi_cube"]=="True"]
    readiness=[]
    for x in pilot:
        name=canonical_name(x["display_name"])
        astrometry=name in legacy["stellar_astrometric_pass"]
        folds=name in legacy["combined_possible_fold_objects"]
        readiness.append(dict(galaxy=x["display_name"],local_hi_cube=True,
            local_stellar_image=x["has_local_stellar_image"]=="True",local_co_image=x["has_local_co_image"]=="True",
            prior_strict_stellar_astrometry_pass=astrometry,
            possible_multiple_sightline_intersections=folds,
            independent_total_mass_coverage="not_validated",
            depth_and_thickness_constraints="prior_ensemble_required",
            externally_constrained_gravity_boundary="not_available_in_this_milestone",
            joint_channel_spatial_covariance="not_observationally_validated",
            validated_motion_independent_mask=False,
            full_field_cube_prediction_admitted=False,
            next_action="expand matched matter coverage and validate covariance" if astrometry
                        else "repair and independently validate stellar-to-HI registration"))
    write_csv(output/"pilot-readiness.csv",readiness)
    registry=read_json(ROOT/"work/gravity-first-principles/mond-atlas-design-001/source-registry.json")
    queue=[]
    present={"sparc":"175 local radial mass models", "things":"12 local spectral cubes plus moment maps",
             "s4g":"stellar images for part of the 12-galaxy seed; full catalog not ingested",
             "manga_dr17":"10071 previously selected local catalog records; not full cube ingestion",
             "hi_manga_dr3":"integrated detections/limits imported through local MaNGA catalog",
             "wallaby_pilot_1":"mixed pilot release records in the legacy DR2 extract; phase counts not separately audited",
             "wallaby_pilot_2":"303 legacy predictor/rejection records, including repeat releases"}
    for entry in registry["sources"]:
        queue.append(dict(survey=entry["id"],official_sources=entry["sources"],
            status="LOCAL_PARTIAL" if entry["id"] in present else "NOT_INGESTED_BY_THIS_MILESTONE",
            local_scope=present.get(entry["id"],"no new source records ingested"),
            availability=entry["availability"],required_action="obtain and verify per-object products, identity/coverage, response independence and selection",
            count_claim="catalogue count is not a completed 3D gravity prediction count"))
    write_json(output/"acquisition-queue.json",queue)
    blocker=dict(goal_complete=False, goal_remains_active=True,
        completed=["local coverage catalog with asset hashes and identity review queue",
                   "175-galaxy radial baseline, 126-galaxy descriptive comparison and exploratory residual tests",
                   "finite-domain Newtonian/QUMOND numerical engine and analytic checks",
                   "finite-depth line projection, beam/channel operations and separable covariance calculations"],
        remaining=["resolve 95 positional identity candidates and 58 missing positions",
                   "acquire additional public per-galaxy resolved assets and independently constrain total baryons",
                   "repair/validate 11 seed stellar registrations and matched component coverage",
                   "build observationally allowed 3D mass ensembles with measured exterior-field constraints",
                   "validate real finite-volume source transforms, gas pressure/streaming models, beam response, spatial/channel covariance and selection masks",
                   "execute and validate full-field cube predictions for 10-20 development pilots, then expand eligible sample",
                   "AQUAL formulation controls and independent galaxy/group/survey transfer evaluation",
                   "commit and push completed validated milestones to main"],
        access_limits=["Existing CUDA virtualenv launcher: Access is denied.",
                      "Shell network request: WinError 10013 (socket access forbidden). No direct downloads performed.",
                      "Git fetch cannot write FETCH_HEAD because linked worktree Git metadata lies outside writable roots."],
        permitted_runtime="Bundled Python 3.12.14 with NumPy 2.3.5; CPU execution",
        publication_status="LOCAL_ONLY_NOT_COMMITTED_OR_PUSHED",
        no_permission_bypass_attempted=True)
    write_json(output/"execution-status.json",blocker)
    pattern_lines=[]
    for p in r["patterns"]:
        pattern_lines.append(f"| {p['feature'].replace('_',' ')} | {p['mse_gain_percent']:+.2f}% | {p['bonferroni_p']:.2f} |")
    report=f"""# MOND observation atlas — executed milestone

The atlas goal is active and **not complete**. We built and ran a local catalog,
a fixed radial comparison, and numerical foundations for the 3D cube pipeline.
There are **zero newly validated full-field galaxy cube predictions** in this milestone.

## What is available

| Product | Actually processed |
|---|---:|
| Catalog identity groups | {c['catalog_identity_groups']:,} |
| Possible duplicate pairs awaiting review | {c['crossmatch_review_pairs']} |
| Identity groups without a coordinate | {c['identities_without_coordinates']} |
| Local MaNGA population records | 10,071 |
| PROBES-I catalog rows | 3,163 |
| SPARC radial models | 175 galaxies / 3,391 radii |
| Legacy WALLABY records | 303, including failures and repeat releases |
| Verified image/cube assets | {c['identity_verified_assets']} / {c['unique_asset_files']} files |
| Verified raw asset bytes | {c['unique_asset_bytes']:,} |
| Galaxies with local HI cubes | 12 |
| Completed total-3D-mass / full-cube gravity validations | 0 |

These source counts overlap. The 13,530 identity groups are **not a certified count
of distinct physical galaxies**. Exact recognized names merge; positional proximity
only creates a review entry. MaNGA is a prior custom selection, not the survey's
official high-quality sample. PROBES supplies extended rotation curves and matched
photometry for a subset; this ingestion verifies its catalog, not every profile.
[SPARC](https://astroweb.case.edu/SPARC/), [PROBES-I](https://arxiv.org/abs/2209.09912).

The MaNGA records preserve 3,033 published integrated-HI detections, 3,280 upper
limits and 3,758 unknown/unavailable entries. A missing measurement is never
replaced with zero gas. Halo and environment-acceleration proxy columns were
excluded from the ingestion allowlist. Stellar-population masses remain model-dependent.

## The first measured pattern

We held the gravity formula fixed: a0 = 1.2e-10 m/s², disk M/L = 0.5 and bulge
M/L = 0.7, with the simple MOND interpolation function and no dark-halo term.
The published signed gas-force convention is preserved. Every input archive row
was checked against the stored SPARC decimal strings. No galaxy gravity parameters
were fitted.

Quality 1–2, inclination 30–80 degrees, at least five valid radii, and finite
positive radius/speed/error/inward total force leave **{r['selected_galaxies']} galaxies
and {r['selected_radii']:,} radii**. All excluded rows and galaxies remain in the output.
We did not remove low-speed points for having large relative errors.

**MOND's radial approximation has smaller fractional speed error in
{r['galaxies_mond_lower_fractional_error_than_newton']} of those 126 galaxies.**
The median galaxy's RMS fractional speed error falls from
{r['median_galaxy_newton_rms_fractional_error_percent']:.1f}% for baryon-only Newtonian gravity
to {r['median_galaxy_mond_rms_fractional_error_percent']:.1f}% for this MOND approximation.
The more outlier-sensitive galaxy-weighted RMS is
{r['galaxy_weighted_newton_rms_fractional_error_percent']:.1f}% versus
{r['galaxy_weighted_mond_rms_fractional_error_percent']:.1f}%.
Slow inner measurements can dominate fractional errors; neither statistic is a
calibrated likelihood or a claim that all galaxies fit well.

In plain terms, the usual low-acceleration correction helps many galaxies. We
then asked whether a galaxy's gas fraction, brightness concentration, size or
broad type helps predict the error left over, after accounting for the range
of baryonic acceleration sampled.

| Added descriptor | Change in held-galaxy residual MSE; positive is better | Four-test adjusted permutation p |
|---|---:|---:|
{chr(10).join(pattern_lines)}

Gas fraction, size and broad galaxy type did not help this test. Surface
brightness gave a small 3.7% improvement, but its galaxy-bootstrap interval includes
no improvement and its adjusted permutation p is 0.06. That is a lead to examine,
not evidence that density causes a MOND failure. These are exploratory tests on
previously used galaxies. Whole-galaxy cross-validation prevents pixel-level
leakage; it does not create a pristine holdout, remove shared group effects, or
establish transfer to a different survey. The gas fraction is an HI-plus-uniform-
stellar-M/L proxy, not a measured total baryon fraction.

The 27-corner sensitivity envelope varies stellar M/L by ±20%, distance by its
reported error, and inclination by its reported error. It is **not a probability
interval**. Inclination changes velocity projection only in this radial stage;
the full source geometry must be rebuilt in the later 3D stage.

## Numerical work that is ready

The new NumPy engine solves Newtonian and QUMOND potentials on a 3D grid using
explicit boundary potentials and a face-centered nonlinear flux. It passed eight
analytic/symmetry/convergence gates. At the finest Plummer-sphere resolution,
force RMS error is {100*n['spherical_resolution'][-1]['newton_relative_force_rms']:.3f}%
for Newtonian gravity and {100*n['spherical_resolution'][-1]['mond_relative_force_rms']:.3f}%
for QUMOND, on the stated radial test region. This implements the two-Poisson-
equation formulation in [Milgrom (2010)](https://arxiv.org/abs/0911.5464).

The cube building blocks integrate all physical depth layers, apply finite
spectral channels and a spatial beam, and score supplied channel/spatial
covariances. Tests verify that two layers on one sightline remain two velocity
components and that the separable covariance score agrees with a dense matrix
calculation. A spectral cube's velocity axis is never interpreted as depth.

The 17-test offline suite passed. These numerical tests do **not** validate an
astronomical source model, real selection mask, separability of real noise, or
exterior-field boundary. The new implementation has no AQUAL solver yet.

## Why the full atlas is still unfinished

Only NGC2903 passed the prior strict stellar-registration validation. Eleven seed
registrations need repair or stronger independent checks. Even NGC2903 had only
27 of 242 tested positions with the required joint stellar/HI/CO coverage.
The earlier analysis identified possible multiple sightline intersections in
NGC2841, NGC2903, NGC3521 and NGC7331. The new depth integrator can represent them,
but has not yet been connected to validated observational 3D ensembles.

Every seed's readiness row keeps these gaps explicit. Total matter coverage,
depth priors, external baryonic fields, gas support/streaming, a validated mask,
and real channel-plus-spatial covariance remain required before gravity scoring.
The radial approximation cannot substitute for this work.

This session can write research files and run the bundled CPU Python. The
existing CUDA environment cannot launch; direct shell downloads are denied;
Git cannot write the linked worktree metadata outside the writable workspace.
**Nothing from this milestone has been committed or pushed to main.** Raw
observations remain in their existing private directories.

## Files and replay

- [Browse all 175 radial comparisons](atlas.html)
- [Pilot readiness](pilot-readiness.csv)
- [Remaining work and access status](execution-status.json)
- [Acquisition queue](acquisition-queue.json)
- [Catalog summary](../{catalog.name}/summary.json)
- [Catalog identities](../{catalog.name}/galaxies.csv)
- [Identity review queue](../{catalog.name}/identity-review.csv)
- [Radial predictions](../{radial.name}/radial-predictions.csv)
- [Galaxy residuals](../{radial.name}/galaxy-residuals.csv)
- [Pattern results](../{radial.name}/patterns.json)
- [Full-field numerical checks](../{numerics.name}/validation.json)

From the repository root, use Python with NumPy. Each run refuses to overwrite
an existing directory. The recorded execution used Python 3.12.14 / NumPy 2.3.5.

```text
python tests/test_mond_atlas_offline.py
python scripts/build_mond_atlas_catalog.py --output work/gravity-first-principles/mond-atlas-catalog-replay
python scripts/run_mond_atlas_radial.py --output work/gravity-first-principles/mond-atlas-radial-replay
python scripts/mond_atlas_fields.py --output work/gravity-first-principles/mond-atlas-numerics-replay/validation.json
```
"""
    (output/"README.md").write_text(report,encoding="utf-8",newline="\n")
    with (radial/"radial-predictions.csv").open(encoding="utf-8") as stream:
        points=list(csv.DictReader(stream))
    grouped={}
    for p in points:
        grouped.setdefault(p["galaxy"],[]).append({k:float(p[k]) if p[k] else None for k in
            ["radius_kpc","observed_speed_km_s","published_error_km_s","newton_speed_km_s",
             "algebraic_mond_speed_km_s","sensitivity_min_km_s","sensitivity_max_km_s"]})
    with (radial/"galaxy-residuals.csv").open(encoding="utf-8") as stream:
        info={r["galaxy"]:r for r in csv.DictReader(stream)}
    data=json.dumps(dict(curves=grouped,info=info),allow_nan=False).replace("<","\\u003c")
    template=ROOT/"scripts/mond_atlas_viewer.html"
    (output/"atlas.html").write_text(template.read_text(encoding="utf-8").replace("__ATLAS_DATA__",data),
                                     encoding="utf-8",newline="\n")
    input_paths=[PROTOCOL,catalog/"summary.json",radial/"summary.json",numerics/"validation.json",legacy_path]
    write_json(output/"input-bindings.json",{p.relative_to(ROOT).as_posix():digest(p) for p in input_paths})
    return dict(report=str(output/"README.md"),viewer=str(output/"atlas.html"),goal_complete=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog",type=Path,required=True)
    p.add_argument("--radial",type=Path,required=True)
    p.add_argument("--numerics",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(assemble(args.catalog,args.radial,args.numerics,args.output),indent=2))
