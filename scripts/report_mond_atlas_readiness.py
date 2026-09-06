"""Publish the source-footprint, identity and guarded covariance milestone locally."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
from pathlib import Path
import unittest

from mond_atlas_common import ROOT,digest,read_json,write_csv,write_json


def make_report(output):
    output=output.resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    base=ROOT/"work/gravity-first-principles"
    astro=base/"mond-atlas-astrometry-001"
    noise=base/"mond-atlas-noise-002"
    identity=base/"mond-atlas-identity-001"
    a,n,i=[read_json(p/"summary.json") for p in [astro,noise,identity]]
    assert not a["errors"] and not n["execution_failures"]
    rows=[];source_assets=[]
    for path in sorted(astro.glob("*.json")):
        if path.name=="summary.json":continue
        star=read_json(path);cube=read_json(noise/(star["galaxy"]+".json"))
        for key in ["image","catalog"]:
            asset=ROOT/star["input_"+key]
            assert digest(asset)==star[key+"_sha256"]
            source_assets.append(dict(galaxy=star["galaxy"],role="stellar_astrometry_"+key,
                path=asset.relative_to(ROOT).as_posix(),sha256=star[key+"_sha256"],bytes=asset.stat().st_size))
        assert digest(ROOT/cube["packet_path"])==cube["packet_sha256"]
        rows.append(dict(galaxy=star["galaxy"],gaia_footprint_strict_pass=star["footprint_strict_pass"],
            supported_validation_stars=star["finite_validation"],
            median_gaia_offset_arcsec=star["footprint_validation_median_arcsec"],
            p90_gaia_offset_arcsec=star["footprint_validation_p90_arcsec"],
            joint_background_diagnostic_pass=cube["diagnostic_pass"],
            joint_background_mean_square=cube["joint_validation_mean_square"],
            joint_background_channel_lag1=cube["joint_validation_channel_lag1"],
            background_train_test_separation_arcsec=cube["minimum_calibration_validation_separation_arcsec"],
            both_preliminary_checks_pass=star["footprint_strict_pass"] and cube["diagnostic_pass"],
            p5_transfer="previously_validated_for_NGC2903_only" if star["galaxy"]=="NGC2903" else "not_validated_or_not_a_P5_product",
            full_3d_baryon_model_validated=False,independent_gravity_cube_prediction=False))
    write_csv(output/"pilot-readiness.csv",rows)
    write_csv(output/"astrometry-source-assets.csv",source_assets)
    input_paths=[astro/"summary.json",noise/"summary.json",identity/"summary.json"]
    for summary in [a,n]:
        for path,expected in summary["input_hashes"].items():assert digest(ROOT/path)==expected,path
        for file,expected in summary["code_hashes"].items():assert digest(ROOT/"scripts"/file)==expected,file
    assert digest(ROOT/"scripts/resolve_mond_atlas_identities.py")==i["code_sha256"]
    for path,expected in i["source_hashes"].items():assert digest(ROOT/path)==expected
    suite=unittest.TestSuite()
    for name in ["test_mond_atlas_offline","test_mond_atlas_astrometry","test_mond_atlas_noise"]:
        spec=importlib.util.spec_from_file_location(name,ROOT/"tests"/(name+".py"))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (output/"validation.log").write_text(log.getvalue(),encoding="utf-8",newline="\n")
    assert result.wasSuccessful(),log.getvalue()
    verification=dict(status="PASS",tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        raw_astrometry_files_rehashed=len(source_assets),cached_cube_packets_rehashed=len(rows),
        original_astrometry_distance_replay_max_arcsec=a["maximum_original_distance_replay_arcsec"],
        source_and_code_bindings_verified=True,goal_complete=False)
    write_json(output/"verification.json",verification)
    both=[r["galaxy"] for r in rows if r["both_preliminary_checks_pass"]]
    status=dict(goal_complete=False,goal_remains_active=True,
        previous_turn_classification="progress",
        object_identity_groups=i["object_groups"],remaining_identity_pairs=i["unresolved_proximity_pairs"],
        stellar_flux_astrometry_pass=a["footprint_strict_pass"],background_noise_diagnostic_pass=n["diagnostic_pass"],
        both_preliminary_checks_pass=both,completed_full_field_galaxy_cube_predictions=0,
        corrected_prior_claim="The earlier assertion that 11 stellar registrations need repair was too strong: the previous gate included Gaia positions in blank parts of the mosaics. 11 now pass the unchanged gate within finite exposure.",
        outstanding=["NGC4214 needs more independently supported astrometric reference stars.",
            "NGC3198 exceeds the declared residual channel-correlation gate in the balanced background split.",
            "Cross-split changes in background diagnostics require uncertainty/robustness checks; passing one split is not a calibrated likelihood.",
            "Transfer from raw/P1 infrared images to cleaned P5 stellar maps is still validated only for NGC2903.",
            "Total baryonic mass conversion, common coverage, missing-pixel treatment and spatial-depth ensembles remain to be validated.",
            "Instrument reconstruction, galaxy selection masks, source-only motion predictions, full-field boundaries and AQUAL checks remain.",
            "90 positional identity pairs and 58 missing coordinates remain.",
            "Additional public acquisition, larger resolved sample and independent survey/group transfer remain.",
            "Publish completed milestones to main when linked-worktree write access is restored."],
        publication_status="LOCAL_ONLY_NOT_COMMITTED_OR_PUSHED")
    write_json(output/"execution-status.json",status)
    table="\n".join(f"| {r['galaxy']} | {r['supported_validation_stars']} | {r['median_gaia_offset_arcsec']:.3f} | {'pass' if r['gaia_footprint_strict_pass'] else 'too few stars'} | {r['joint_background_mean_square']:.3f} | {r['joint_background_channel_lag1']:+.3f} | {'pass' if r['joint_background_diagnostic_pass'] else 'fails channel correlation'} |" for r in rows)
    report=f"""# MOND atlas — source and noise readiness update

**The atlas is still in progress. This milestone clears two preliminary checks
for ten of the twelve pilot galaxies; it does not yet deliver a full 3D gravity prediction.**

## A substantial correction to the previous assessment

The earlier registration test treated every Gaia star inside the rectangular
image boundary as a usable reference. The mosaics contain blank corners and
other unexposed areas. **283 of the 816 tested Gaia positions lacked finite image
support.** The test assigned these positions large mismatches, making good
images look misregistered.

We kept the previous calibration/validation assignments, the previously selected
plain-TAN coordinates, and the original accuracy thresholds. We required a
finite 7×7 image patch at each Gaia position; measured zero flux remains valid.
No translation, rotation, or gravity parameter was fitted.

**Eleven images now pass the same strict thresholds; only NGC4214 lacks enough
supported stars.** This corrects the earlier claim that eleven registrations
needed repair. The old star-distance computations were reproduced to a maximum
difference of {a['maximum_original_distance_replay_arcsec']:.2e} arcsecond, cross-checking
the new restricted FITS/TAN implementation against the prior Astropy/SciPy outputs.
Independent vector-projection, FITS scaling/blanking, and roundtrip tests also pass.

This is a development repair of a faulty test. It is not new independent sky
data, a complete exposure-weight calibration, or validation of the cleaned
stellar maps. P1/IRAC flux coordinates must still transfer correctly to P5.
NGC2903 has an earlier validated P1/P5 reconstruction; that transfer has not been
established for the other four P5 galaxies. P5 masks still exclude every nonzero
label, including negative labels, as the [publisher specifies](https://irsa.ipac.caltech.edu/data/SPITZER/S4G/docs/P5_README.html).

## We also fitted spatial noise correlations to real cubes

Channel-only noise models miss the fact that neighboring pixels share noise
after beam smoothing. We fitted a channel covariance and a spatial correlation
model using background regions 550–680 arcseconds from each galaxy center.
Calibration and validation use separate guarded blocks; validation is subsampled
to avoid a nearly singular covariance from adjacent, heavily smoothed pixels.
The minimum calibration-to-validation separation is recorded for each galaxy.

The first checkerboard attempt starved some sky quadrants of validation samples.
Its outputs are retained in noise run 001. Run 002 assigns blocks using coordinates
alone so every quadrant is represented, with the same noise thresholds. All
means, variances and correlation parameters are then refitted from calibration
pixels only. A test that multiplies held-region noise by four leaves all fitted
covariances unchanged and correctly fails the validation gate.

In the balanced run, **eleven galaxies pass the preliminary background moment
checks**. NGC3198 retains channel correlation 0.176, above the declared 0.15
threshold. NGC2841 failed this channel test in the earlier unbalanced split but
passes the balanced split: the failure identity is split-dependent, which must
be investigated rather than declared an intrinsic property of either galaxy.
Passing these broad gates does not establish exact Gaussian/separable noise,
the mask within the emitting galaxy, or a calibrated chi-square significance.

| Galaxy | Supported validation stars | Median Gaia offset (arcsec) | Stellar flux coordinates | Held joint-noise mean square | Channel lag-1 | Noise diagnostic |
|---|---:|---:|---|---:|---:|---|
{table}

“Mean square” would be near one under a correctly normalized noise model. It is
a diagnostic here, not a gravity fit. The overlap clearing both preliminary
checks is **{len(both)} galaxies**. No stellar mass, rotation speed, gas descriptor or
gravity formula was adjusted to obtain those passes.

## Five real duplicates resolved

Five pairs of MaNGA identities share the same positive NSA catalog ID, the same
IAU source name, and sub-arcsecond coordinates within the same input release.
We now group each pair as one object, with exact source-row evidence. This brings
the working grouping from 13,530 to **13,525 object groups**, with **90 proximity
pairs still unresolved**. This remains an uncertified count of distinct physical
galaxies. Original observations are preserved, and future holdouts must use the
group identity so repeated observations cannot leak across train/test splits.

## Next required work

The critical remaining work is total-matter mapping and the actual full-field
motion prediction: cleaned-map transfer, conversion uncertainties, masked regions,
shared stellar/atomic/molecular coverage, allowed spatial-depth structures,
exterior fields, gas support/streaming and instrument response. The local 3D and
cube engines are foundations for that work. The catalog and radial findings from
the [previous milestone](../mond-atlas-execution-002/README.md) remain available;
its registration-readiness statement is superseded by this audit.

The existing CUDA environment, shell downloads and linked-worktree Git writes
remain unavailable under this session's permissions. This work ran on bundled
CPU Python/NumPy. **This milestone has not been committed or pushed to main.**

## Evidence and replay

- [Updated pilot readiness](pilot-readiness.csv)
- [Astrometry source hashes](astrometry-source-assets.csv)
- [All 816 star positions and footprint decisions](../mond-atlas-astrometry-001/stars.csv)
- [Astrometry summary](../mond-atlas-astrometry-001/summary.json)
- [Balanced real-cube noise summary](../mond-atlas-noise-002/summary.json)
- [Object grouping overlay](../mond-atlas-identity-001/object-groups.csv)
- [Exact duplicate evidence](../mond-atlas-identity-001/merge-evidence.json)
- [Verification](verification.json) and [28-test log](validation.log)
- [Full outstanding work](execution-status.json)

From the repository root with Python and NumPy; choose new output directories:

```text
python scripts/run_mond_atlas_astrometry.py --output work/gravity-first-principles/mond-atlas-astrometry-replay
python scripts/resolve_mond_atlas_identities.py --catalog work/gravity-first-principles/mond-atlas-catalog-004 --output work/gravity-first-principles/mond-atlas-identity-replay
python scripts/run_mond_atlas_noise.py --output work/gravity-first-principles/mond-atlas-noise-replay --private work/private/mond-atlas-noise-replay
python tests/test_mond_atlas_astrometry.py
python tests/test_mond_atlas_noise.py
```
"""
    (output/"README.md").write_text(report,encoding="utf-8",newline="\n")
    write_json(output/"input-bindings.json",{p.relative_to(ROOT).as_posix():digest(p) for p in input_paths})
    return dict(report=str(output/"README.md"),tests=result.testsRun,both_preliminary_checks_pass=len(both),goal_complete=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();print(json.dumps(make_report(args.output),indent=2))
