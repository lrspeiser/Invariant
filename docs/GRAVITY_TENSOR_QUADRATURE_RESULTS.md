# Source-integration campaign: one retained interpolation failure

All five registered integration variations are complete for both source
thicknesses. **Nine of ten case/thickness combinations pass all targets.**
The reduced-cutoff, half-thickness case fails the physical density-gradient
identity after interpolation. The canonical source is not thereby falsified,
but numerical promotion remains withheld while this sensitivity is resolved.
**246 focused tests and lint pass.** No new gravity law or observational fit
is validated.

## What was held fixed and varied

The physical source, fine tensor mesh, 60--80 kpc exterior join and global
gravity parameters were fixed. Each case recomputes all 16 mixed partials,
constructs the same interpolator and compares its joined fields at 1,562
probe entries per thickness: 1,364 retained entries plus 198 derivative/source
entries. Overlaps between those sets are retained; these are not 1,562
independent observations. The five perturbations are radial quadrature
128 to 64 nodes per interval, wavenumber quadrature 64 to 32, cutoff 400 to
200 inverse kpc, vertical resolution 2400 to 1200 intervals, and vertical
extent 24 to 32 scale heights at unchanged interval spacing.

| Variation | Primary third-tensor change | Half-thickness third-tensor change | Half-thickness density-gradient error | All-target passes |
|---|---:|---:|---:|---:|
| radial_coarse | 1.1711804e-05 | 0.00079541994 | 0.0040655825 | 2/2 |
| wavenumber_coarse | 2.7126215e-05 | 4.9824559e-05 | 0.0069281802 | 2/2 |
| cutoff_200 | 3.4373768e-05 | 0.0034436674 | 0.013115396 | 1/2 |
| vertical_coarse | 1.3729094e-05 | 1.3527668e-05 | 0.006838584 | 2/2 |
| tail_extent | 9.3355164e-07 | 1.3971634e-06 | 0.0068167523 | 2/2 |

Across every case, the largest scaled changes are 2.97741094e-08
in force, 4.17864538e-06 in Hessian and 0.00344366741
in third tensor. These remain below 1e-4, 0.002 and 0.01, respectively.
All density checks also pass. One density-gradient check fails its unchanged
0.01 target: **0.0131154 at R=65, z=0.025 kpc**, half thickness and cutoff 200.

The comparison scales and physical-source identity denominators are the
same as in the preceding joined-source audit. Raw absolute changes in the
16 mixed inputs are retained with their derivative-order-dependent units;
they have no standalone admission threshold. These finite parameter
variations do not establish a uniform continuum error bound or prove that
every possible quadrature error is small.

## Direct evaluation identifies the next numerical question

At the exposed failing point, independent direct evaluation of the retained
integral formulas gives density-gradient errors of **0.000654369 at cutoff
200** and **0.000654496 at cutoff 400**. Both are below the 0.01 target and
are much smaller than the cutoff-200 interpolated error. Thus this point's
failed identity arises in the representation of the sampled mixed jets;
it is not a failure of the direct cutoff-200 source identity at that point.
The diagnostic does not establish this conclusion uniformly across space,
nor does it yet distinguish interpolation truncation, rapidly varying mixed
inputs and input-rounding effects.

A second, deliberately bounded diagnostic tested the global-gauge arithmetic
hypothesis. Retaining the same raw source samples and using only the existing
cell-local anchor gave error **0.0131112**, compared with **0.0131154** using
the global shift. It still fails. The observed corner-potential round-trip
changes are at most about 9.1e-13 in potential units. Removing the global
shift alone is therefore **not a repair for this failure**. This prototype
was not promoted into the source provider; its negative result is retained.

## What follows

Separate the contributions of the mixed-derivative inputs to the failing
interpolant, then independently refine spatial sampling and integration
cutoff with an appropriately resolved radial quadrature. Retain both
thicknesses and all probes, and add off-grid checks so that turning one
failed probe into a grid node cannot masquerade as a general repair.
Do not loosen the target or discard the failed coarse-cutoff case.

The finer canonical cutoff may remain usable, but a coarser-case failure
is neither automatic rejection nor sufficient evidence of its accuracy.
After resolving the numerical qualification question, evaluate and refine
the separate full-action Poisson solve before astronomical rescoring.
Cluster matter/light, precision Solar System fits, stability, a complete
light sector, direct outer-star data and independent confirmation remain
requirements of the active discovery goal. This checkpoint adds zero new
observational scores or physical exclusions.

## Evidence

- `tensor-quadrature-radial-001`: `33472a0c9c6066f28d0a30751f7e1a5e48a66a89c64e13605e53ae63c338ac66` (52 verified snapshots)
- `tensor-quadrature-wavenumber-001`: `4fbeca53cb8e3586974f4693a62c94fac678de4864c2141e00ec0baaabb2be27` (52 verified snapshots)
- `tensor-quadrature-cutoff-001`: `4eaa80f4a1562f4ad14e8ddf3760dffcd53f674694da0a1de9d1d33c24f3de4c` (52 verified snapshots)
- `tensor-quadrature-vertical-001`: `0736af3f88c79fa3b4112af14cc93ae72f50a358bf52be62da36e3b027608a05` (52 verified snapshots)
- `tensor-quadrature-extent-001`: `3bf7363cb732940170aed6cb0d4e3f082dcdb671f7aba771082dc82c2f053282` (52 verified snapshots)
- `tensor-cutoff-diagnostic-001`: `658aafa123a7d3c48a7d42f8bb586c2037f7aa5142c51e50295f9702f4d3b4bf` (44 verified snapshots)
- `tensor-gauge-diagnostic-001`: `1d24d753632e6c725976383bd95d8c89ad0005c074eabaa797cc3f21825a6e41` (42 verified snapshots)
- `tensor-controls-004`: `e0ebaf251de74a1c55e71906734d1a1fe79412ac1195c10cb703fd222c5070a1` (125 verified snapshots)
- `tensor-controls-005`: `71a463eb00977f43f7ae8a745e64a09dfce13bffee724ef73e319054d5b83f99` (127 verified snapshots)
