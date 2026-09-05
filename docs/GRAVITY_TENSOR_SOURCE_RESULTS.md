# C3 potential interpolation: first actual-source pilot

The numerical representation now reproduces direct source evaluations at all
128 fixed off-grid probes within the registered force, Hessian and third-tensor
targets in a small central region. This is progress toward the full action
calculation; no new gravity law or astronomical prediction is validated.

## What was built and tested

One tensor Hermite potential supplies every force, Hessian and third derivative.
It is C3 across cell boundaries, uses physical R and z derivatives through order
three in each coordinate, and evaluates regular cylindrical axis quotients
without subtracting nearly equal Hessian components. Six new tests cover
symbolic fields, an independent high-precision boundary solve, nonpolynomial
convergence, symmetry/domain validation, all 16 Hankel mixed derivatives, and
all 16 omitted-tail derivatives against independent Gaussian integrals.
The focused CI lint and **237 tests pass**. The exact execution inputs were
snapshotted and verified (118 control inputs;
43 pilot inputs).

## Actual-source pilot

The unchanged primary NGC 3198 source was evaluated over R=0..4 kpc and
z=0..0.8 kpc, with the retained K=400 Hankel quadrature, 2400-interval vertical
source and 50-digit low-wavenumber tail correction. Probe locations were fixed
one-quarter into each coarse cell before execution. They are absent from both
interpolation grids. The direct reference evaluates the retained integral
formulas, not the interpolator. The high mixed partials have synthetic controls;
their actual-source quadrature refinement remains pending.

| Grid | Largest scaled force error | Hessian error | Third-tensor error |
|---|---:|---:|---:|
| 17 x 9 | 2.65936964e-06 | 2.46707316e-06 | 0.000160233049 |
| 33 x 17 | 7.33683275e-09 | 3.48452821e-07 | 1.86697534e-06 |
| Fixed target | 1e-4 | 0.002 | 0.01 |

Third-tensor discrepancy improves by 85.82 times.
Force and Hessian errors use reference norms; third-tensor error uses the
larger of its reference norm and H/(spherical radius + minimum source height).
These are sampled interpolation differences, not uniform error bounds or
independent observational measurements. Both thicknesses, outer disk/taper,
60--80 kpc join and the remaining spatial domain still need interpolation tests.

## Preserved numerical failures and precision limit

The first synthetic execution exposed an incorrect reflection sign at z=0,
which zeroed a nonzero even derivative. That bug was repaired.
The second execution passed the 117 ordinary polynomial field comparisons but
failed the much stricter near-axis TRpp/R comparison. Independent 70-digit
arithmetic finds a roughly 1.65e-11 error from rounding the supplied potential
samples, versus about 4.17e-14 from evaluating their interpolant. Correctly
rounding the large-offset samples does not remove their information loss.

The revised controls retain the original analytic tolerance after removing
the additive constant before sampling and separately compare the original
rounded samples with an independently solved high-precision interpolant.
This does **not** turn the original large-offset analytic comparison into a pass.
A gauge shift after samples have been rounded cannot recover lost bits.
Actual-source admission must therefore check interpolation and source sampling
precision together, especially as cells shrink and higher derivatives amplify
roundoff. The pilot subtracts one global gauge after source evaluation;
its measured errors include the remaining input precision loss.

## Next scientific gate

Expand the same potential representation across both source thicknesses,
radial source interfaces and the exterior join; vary the source quadratures
and interpolation independently. Then evaluate the full length-action flux
and its separate Poisson solve before rescoring the galaxy or widening lengths.
Cluster dynamics/lensing, precision Solar System behavior, stability, light
coupling and independent confirmation remain required. Overlap and elastic
mechanisms retain their separate conservation and mass-scaling questions.

## Evidence

- Pilot: `be895d0d3098b7b98c5172ba16c83128f8dbb450946ddfca4a44b64bcbb4cf6c`
- Focused controls: `af6ee423cb8cdfeda851515c4c69520de304f1cd8f47744818c0cd3389a2c0a0`
- Axis conditioning diagnostic: `620d859ce26d63c0f678338814e5551eb4546db7446b25d1679ab94475e1944c`

All are local append-only results under `work/gravity-first-principles`.
Early test-failure details are retained as transcript-derived records; the
later diagnostic and pilot have execution snapshots. No remote push was made.
