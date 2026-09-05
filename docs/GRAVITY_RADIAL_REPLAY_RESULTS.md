# Refined source passes the registered numerical replay

All six numerical cases pass for both disk thicknesses on the refined 241 x 23
mesh. Each case retains all 2,842 probe entries. Independent differentiation
of the canonical joined potential passes at 360 probe entries per thickness,
including the added radial boundaries. Lint and 246 focused tests pass.

This admits this representation to a development field solve under sampled
numerical checks. It is not a uniform continuum error bound, an astronomical
fit, or validation of a gravity law. Earlier failed meshes and interventions
remain negative numerical results; none is reclassified.

## Replayed comparisons

Only new radial samples were generated. All 81,328 old mixed-derivative values
in each of the twelve tables remain bit-identical after JSON serialization.
The cutoff-200 case reuses its previously generated refinement samples.
The canonical source is compared against its old mesh; the other cases are
compared against the refined canonical source. The integration variations
change radial or wavenumber quadrature, cutoff, vertical resolution, or tail
extent individually. Source and field thresholds are unchanged.

| Case | Thickness | Scaled third-tensor change | Scaled density-gradient error | All field/source checks |
|---|---|---:|---:|---|
| canonical | primary | 4.62648257e-05 | 0.000466401686 | pass |
| canonical | height_half | 0.00170443594 | 0.00216191327 | pass |
| radial_coarse | primary | 2.12724947e-05 | 0.000466400707 | pass |
| radial_coarse | height_half | 6.98015262e-05 | 0.00216166591 | pass |
| wavenumber_coarse | primary | 3.68668554e-05 | 0.000466366837 | pass |
| wavenumber_coarse | height_half | 6.27658908e-05 | 0.00215696583 | pass |
| cutoff_200 | primary | 2.16514113e-05 | 0.000466401686 | pass |
| cutoff_200 | height_half | 0.000211353247 | 0.00229235033 | pass |
| vertical_coarse | primary | 3.33375371e-05 | 0.000466394603 | pass |
| vertical_coarse | height_half | 2.87495163e-05 | 0.00216324775 | pass |
| tail_extent | primary | 9.33551638e-07 | 0.000466369574 | pass |
| tail_extent | height_half | 1.39716343e-06 | 0.00216191327 | pass |

The density-gradient limit is 0.01; the worst retained value is 0.00229235.
Force, Hessian, and third-tensor change limits are respectively 0.0001,
0.002, and 0.01; the density limit is 0.002. These are normalized diagnostics,
not fractional observational errors or uncertainties on a fitted parameter.

Independent derivative checks retain both step sizes and central diagnostics.
At radial interfaces, both one-sided stencils must pass the 0.0001 target at
step 0.0005 kpc. Central stencils remain appropriate elsewhere. A C3
interpolant need not have a continuous fourth derivative, so an interface
central-stencil failure is not silently recast as a successful central check.
The audit covers meridional differentiation and the implemented third tensor;
separate analytic controls check azimuthal geometry. This is not an independent
three-dimensional scalar differentiation of every Hessian component.

## Next experiment

Use these canonical providers in the separate full-action Poisson solve.
Rotate the Cartesian/cylindrical anomalous flux into the solver's spherical
components and add the separately evaluated Newtonian field after solving.
Flux itself must not be mistaken for the physical gravitational acceleration.
Freeze the numerical grid, boundary, angular-resolution and radial-resolution
comparisons before examining results. Retain the existing globally specified
54 parameter cards and both physical thicknesses; do not select cards because
they appear easier to resolve. Resolve numerical differences before reporting
any new astronomical ranking or claimed small length-scale signal.

The broader goal still requires direct outer-star data, cluster dynamics and
lensing, precision Solar System predictions, conservation, stability, a
complete matter/light theory and independent validation. No new observational
score or physical exclusion is added at this checkpoint.

## Evidence

- `radial-tensor-replay-001`: `9136c90030a114b89b350f816beffa41ce24ad7958d6eeafaad107483e5c74c2` (70 verified input snapshots)
- `tensor-controls-007`: `a44a6e64fd764421edf9b64c413b3df4cba1cc0d75922cc724d690e51195aafd` (130 verified input snapshots)
