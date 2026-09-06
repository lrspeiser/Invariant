"""Publish the source reprojection correction and retained numerical failure."""
from __future__ import annotations
import argparse,csv,io,re,unittest
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT,read_json,write_json,write_csv,digest


def report(output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);base=ROOT/'work/gravity-first-principles'
    folders=[base/name for name in ('mond-atlas-projection-001','mond-atlas-projection-002','mond-atlas-field-003','mond-atlas-field-004')]
    summaries=[read_json(folder/'summary.json') for folder in folders]
    for summary in summaries:
        for key in ('code_hashes','source_bindings','bindings'):
            for path,expected in summary.get(key,{}).items():assert digest(ROOT/path)==expected,path
        for product in summary.get('products',[]):assert digest(ROOT/product['path'])==product['sha256']
    with (folders[0]/'source-closure.csv').open() as stream:single=list(csv.DictReader(stream))
    with (folders[1]/'mixed-height-source-closure.csv').open() as stream:mixed=list(csv.DictReader(stream))
    table='\n'.join(f"| {float(r['height_kpc']):g} | {100*float(r['unchanged_lift_relative_image_rms']):.2f}% | {100*float(r['refitted_source_relative_image_rms']):.2f}% |" for r in single if r['component']=='stellar_luminosity')
    mixed_table='\n'.join(f"| {100*float(r['thin_light_fraction']):g}% | {100*(1-float(r['thin_light_fraction'])):g}% | {100*float(r['source_image_relative_rms']):.2f}% |" for r in mixed)
    checks=[]
    for name,values in list(summaries[2]['thin_model_numerical_checks'].items())+list(summaries[3]['checks'].items()):
        checks.append(dict(case=name,**values,aggregate_gate_pass=max(values['newton_vector_relative_rms'],values['mond_vector_relative_rms'])<.03,
            every_ring_gate_pass=max(values['newton_maximum_ring_relative_rms'],values['mond_maximum_ring_relative_rms'])<.05))
    write_csv(output/'numerical-checks.csv',checks)
    check_table='\n'.join(f"| {r['case']} | {100*r['newton_vector_relative_rms']:.3f}% | {100*r['mond_vector_relative_rms']:.3f}% | {'pass' if r['aggregate_gate_pass'] and r['every_ring_gate_pass'] else 'fails aggregate gate'} |" for r in checks)
    force_rows=[]
    for folder in folders[2:]:
        for path in folder.glob('*-result.json'):
            r=read_json(path)
            assert max(r['residuals'].values())<1e-10
            assert abs(sum(r['source']['component_mass_msun'].values())/r['source']['finite_grid_total_mass_msun']-1)<1e-10
            with (folder/(r['id']+'-forces.csv')).open() as stream:data=list(csv.DictReader(stream))
            assert len(data)==72*len(r['profile'])
            for p in r['profile']:
                for gravity in ('newton','mond'):
                    force=np.array([float(x[gravity+'_inward']) for x in data if float(x['radius_kpc'])==p['radius_kpc']])
                    assert abs(np.sqrt(p['radius_kpc']*force.mean())-p[gravity+'_force_speed_kms'])<1e-10
            force_rows.append(dict(id=r['id'],source_mass_msun=r['source']['finite_grid_total_mass_msun'],
                newton_pde_residual=r['residuals']['newton_relative_pde_residual'],mond_pde_residual=r['residuals']['mond_relative_pde_residual']))
    write_csv(output/'field-integrity.csv',force_rows)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_mond_atlas*.py')
    log=io.StringIO();tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (output/'validation.log').write_text(log.getvalue(),encoding='utf-8',newline='\n');assert tests.wasSuccessful(),log.getvalue()
    status=dict(goal_complete=False,goal_remains_active=True,previous_goal_turn_classification='progress',
        object_identity_groups=13525,galaxies_with_conditional_full_fields=1,source_image_cases_executed_this_turn=18,
        new_full_field_runs_this_turn=8,total_conditional_full_field_runs=19,numerical_unit_tests_passed=tests.testsRun,
        repaired_thin_source_numerical_gates_pass=summaries[2]['thin_model_numerical_gates_pass'],
        repaired_mixed_source_numerical_gates_pass=summaries[3]['mixed_model_numerical_gates_pass'],
        latest_admitted_motion_comparisons=0,completed_full_field_galaxy_cube_predictions=0,
        source_admission_disposition='SOURCE_BLOCKED',publication_status='LOCAL_ONLY',
        corrected_prior_source_assumption='The old conditional model obtained a face-on map geometrically and then added finite thickness. Reprojection changes the image; the nominal 0.4 kpc stellar lift has 22.87 percent RMS mismatch in the restricted source diagnostic.',
        next_required=['Finish mixed-source lateral convergence without relaxing the existing 3 percent aggregate gate; consider a memory-bounded solver for the next resolution.',
            'Use the projection operator in future source construction; do not treat the old thin deprojection followed by arbitrary thickening as observationally consistent.',
            'Calibrate source covariance, native beam/pixel projection, absolute photometry, distance and missing components before posterior or motion admission.',
            'Recover and reverify the original S4G geometry tables; current derived values are bound, but the raw referenced files are absent.',
            'Complete independent geometry/depth/external-field ensembles, AQUAL controls and a full gas-motion cube likelihood.',
            'Validate other pilot mass maps, acquire the broader resolved dataset and preserve new galaxy/survey/group holdouts.',
            'Publish completed milestones when linked Git writes are permitted.'])
    write_json(output/'execution-status.json',status)
    write_json(output/'verification.json',dict(status='UNIT_TESTS_AND_ARTIFACT_INTEGRITY_PASS_MIXED_FIELD_CONVERGENCE_FAILS',
        tests_run=tests.testsRun,failures=len(tests.failures),errors=len(tests.errors),field_runs_verified=len(force_rows),
        all_source_optimizers_converged=all(summaries[i]['all_optimizers_converged'] for i in (0,1)),
        projection_figure_visually_reviewed=True,admitted_scientific_motion_comparisons=0,goal_complete=False))
    text=f'''# MOND atlas — source projection changes what we can trust

**The earlier 3D construction failed an important image check.** Geometrically
stretching the observed stellar map and then adding a 0.4 kpc vertical thickness
produces a projected light pattern with **22.87% RMS mismatch** in this source
diagnostic. The field solver can accurately solve that density, but that does not
make the density a faithful model of the observed galaxy.

![Measured stellar image, the overly smoothed earlier thick model, its refitted version, and image mismatch across assumed heights](../mond-atlas-projection-002/source-projection.png)

The image is a source diagnostic in geometrically stretched coordinates, not a
measured 3D view. All three panels use the same logarithmic scale, proportional
to the projected stellar luminosity times cos(inclination). Gray cells are not
usable source measurements. No rotation velocities enter this experiment.

## Why thickness changes the picture

An inclined thick disk projects several shifted layers onto the same image.
Repeating the geometrically stretched photograph through every layer smears the
photograph a second time. A valid construction has to find a distribution whose
projection reproduces the measured light.

For the tested flat, separable exponential family,

`rho(X,Y,z) = Sigma(X,Y) exp(-abs(z)/h) / (2h)`

the projected image is a convolution along the stretched minor direction with
an exponential kernel of scale `h tan(i)`. We integrated that kernel over source
and image cells and checked it against independent numerical line-of-sight
integration. The inverse calculation fits a nonnegative planar distribution
using only source-image values and coverage weights. It retains signed CO
measurements and does not reinterpret blank sky pixels as measured zero mass.

## Different depth arrangements can reproduce similar images

| Single stellar exponential height (kpc) | Earlier lift: image RMS mismatch | Refit planar light: image RMS mismatch |
|---|---:|---:|
{table}

The 0.4 kpc case still misses the stellar image by 8.38% after refitting. Smaller
heights can reproduce more of the structure. That does **not** measure the true
height: a thin model can absorb projected structure into its planar distribution.
An independent synthetic test explicitly constructs two distinct depths that
reproduce the same positive projected image.

We also allowed a shared planar light distribution to have both 0.1 and 0.4 kpc
vertical populations, with fractions fixed before this source-only follow-up:

| Light in 0.1 kpc layer | Light in 0.4 kpc layer | Refit image RMS mismatch |
|---|---|---:|
{mixed_table}

Thus, a mostly thick model with a thin contribution can resemble the measured
image better than the single thick layer. These fractions are illustrative light
fractions, not independently measured stellar masses, ages or confidence bounds.
The total recovered stellar luminosity changes by less than 1% between the pure
0.1 kpc case and the 25%-thin mixture. Their spatial-depth distributions differ.

**The 5% threshold is only a flag for a substantial construction mismatch.** It
is not a noise-calibrated acceptance rule, a posterior interval or proof that a
particular height is observationally allowed. The fits use the available source
image, not an independent withheld image. Source errors, covariance and physical
population priors still need to be included.

Atomic gas and CO were tested in the same way. At the nominal 0.2 kpc gas height,
the reconstructed HI source misses its projected image by about 0.09%, and CO
by 3.56%. CO already has a 3.15% floor for the zero-height nonnegative fit because
the measured source includes negative noise values. These percentages are not
comparable statistical significances: the tracer noise and selection differ.

## We recomputed gravity, and retained a numerical failure

Two source alternatives—pure 0.1 kpc stellar light and the 25%-thin mixture—were
lifted with cell-integrated vertical weights and combined with the refitted gas
sources. The constant stellar mass-to-light assumption makes the light fractions
conditional mass fractions. Both Newtonian and full QUMOND fields were solved.
This compares jointly changed planar and depth distributions, not height alone.

The first calculation suggests that mean force-equivalent speeds are less
sensitive than the sideways component. **The mixed model has not yet passed
the required lateral convergence test**, so its precise directional-force
differences are not a validated finding. The original failure is preserved.

| Numerical follow-up | Newtonian vector RMS difference | QUMOND vector RMS difference | Original gates |
|---|---:|---:|---|
{check_table}

The unchanged requirements are 3% aggregate vector RMS and 5% in every radius
ring between 2 and 15 kpc. The thin model passes; the mixed model's lateral test
has 3.53% Newtonian and 3.14% QUMOND aggregate differences. Vertical and box
checks pass. The next step is finer spatial convergence, not relaxing the gates.
The small linear Poisson residuals do not override this failed discretization test.

## What this changes in the atlas

- The old conditional force results remain reproducible, but their nominal
  stellar source cannot be treated as an image-consistent 3D reconstruction.
- Future mass ensembles must project back through the observation model before
  their gravity predictions can be admitted. Preserving total mass is insufficient.
- A single projected image leaves substantial depth ambiguity. The atlas needs
  ensembles and additional independent constraints rather than one asserted 3D map.
- No new kinematic response comparisons were made in this phase. The earlier
  exploratory rotation comparison remains nonadmitted, as already disclosed.

The stellar maps and conversion assumptions come from the
[S4G ICA study](https://arxiv.org/abs/1410.0009) and
[light-to-mass calibration](https://arxiv.org/abs/1402.5210). The tracer products
come from [THINGS](https://arxiv.org/abs/0810.2125) and
[HERACLES](https://arxiv.org/abs/0905.4742). Reading those papers does not supply
missing observational covariance, geometry or mass phases.

There is also a concrete provenance gap: both raw S4G geometry tables referenced
by the stored derived configuration are absent from this workspace and from the
checked original checkout. The derived record and hashes remain available, but
the original tables must be recovered before fresh raw-record verification.

This milestone executed **18 source-image fits, 8 additional full field runs and
43 passing unit tests**. One field convergence gate remains failed. Only one
galaxy has conditional field calculations; **zero galaxies yet have an admitted
full-field cube likelihood**. The 10–20 pilot and larger resolved-sample goals
are not complete. Git publication, new shell downloads and the old CUDA runtime
remain unavailable; all new artifacts are local.

## Evidence and replay

- [Single-height source diagnostics](../mond-atlas-projection-001/source-closure.csv)
- [Image errors by radial annulus](../mond-atlas-projection-001/source-closure-annuli.csv)
- [Mixed-height diagnostics](../mond-atlas-projection-002/mixed-height-source-closure.csv)
- [Projection source assumptions and hashes](../mond-atlas-projection-001/summary.json)
- [Reconstructed-source field results](../mond-atlas-field-003/summary.json)
- [Mixed-source numerical failure](../mond-atlas-field-004/summary.json)
- [Numerical checks](numerical-checks.csv), [field integrity](field-integrity.csv)
- [Verification](verification.json), [43-test log](validation.log), [remaining work](execution-status.json)

Choose unused output directories when replaying from the repository root:

```text
python scripts/run_mond_atlas_source_projection.py --output work/gravity-first-principles/projection-replay --private work/private/projection-replay
python scripts/run_mond_atlas_mixed_source.py --output work/gravity-first-principles/mixed-replay --private work/private/mixed-replay
python scripts/run_mond_atlas_reprojected_fields.py --output work/gravity-first-principles/projected-fields-replay --private work/private/projected-fields-replay
python scripts/continue_mond_atlas_reprojected_checks.py --output work/gravity-first-principles/mixed-checks-replay --private work/private/mixed-checks-replay
python -m unittest discover -s tests -p "test_mond_atlas*.py" -v
```

The default configurations bind the original immutable source packets. To chain
a new acquisition/reconstruction instead, make new protocol copies with those
explicit source paths and retain their new hashes. Do not edit a frozen run.
'''
    (output/'README.md').write_text(text,encoding='utf-8',newline='\n')
    paths=[Path(__file__)]+[folder/'summary.json' for folder in folders]+[folders[1]/'source-projection.png',ROOT/'scripts/plot_mond_atlas_projection.py']
    write_json(output/'input-bindings.json',{str(p.relative_to(ROOT)):digest(p) for p in paths})
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if not link.startswith('https:'):assert (output/link).is_file(),link
    print(dict(report=str(output/'README.md'),tests=tests.testsRun,new_source_fits=18,new_field_runs=len(force_rows),mixed_convergence_pass=False,goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();report(args.output.resolve())
