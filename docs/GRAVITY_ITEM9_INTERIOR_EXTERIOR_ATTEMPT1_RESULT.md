# Gravity roadmap Item 9: interior/exterior balance attempt 1

## Decision

`REJECT_ITEM9_INTERIOR_EXTERIOR_EXPLORATION`

This is a nonpromoted positive lead. Eleven of twelve frozen gates pass on a large, fresh
PROBES I sample, but the qualifying selector regresses on the prespecified SHIVir survey
subset. The 323 confirmation rotation profiles therefore remain sealed. The result does not
authorize Item 10 or establish a gravity law.

## Independent real-data problem

PROBES I contains 3,163 resolved galaxy rotation curves with homogeneous multiband
photometry. The response-blind metadata audit retained 1,292 galaxies with clean g/r/z/W1
photometry, finite distances, and no SPARC contribution. A salted primary-survey by distance
split assigned 969 identities to exploration and 323 to confirmation before the profile
archive was accessed.

The test used r-band surface-brightness and cumulative-light profiles with a fixed
mass-to-light ratio. It converted line-of-sight rotation velocities using a frozen
photometric inclination rule. This is a stellar-light test: it does not include resolved
atomic, molecular, or ionized gas and uses a spherical enclosed-mass approximation rather
than an exact thin-disk Poisson solution.

Primary sources:

- PROBES I article: <https://doi.org/10.3847/1538-4365/ac83ad>
- fixed dataset revision: <https://doi.org/10.5281/zenodo.10456319>

## Frozen creative search

Commit `bcf9df003f8c311ef07f591d46e3140fa3d4b889` froze the sample,
features, operators, amplitude grid, formula labels, folds, quality rules, controls, and
gates before the profile archive was downloaded. Commit `088798d3` bound the implementation
to that freeze.

The grammar contains 72 nonlocal operators over stellar surface-brightness occupancy,
stellar acceleration occupancy, enclosed mass fraction, outer inverse-radius potential,
and outer lever arm. Four log-radius scales, two interior/exterior combinations, five
target-blind galaxy conditions, five logistic intercepts, and seven slopes produce 12,600
formula cells in 11,160 declared equivalence classes.

The labels were frozen rather than inferred from fit quality:

- 8,400 cells are `COMBINATION` descendants of the earlier SPARC focusing construction;
- 1,400 are `KNOWN_FAMILY_COMBINATION` cumulative-mass balance cells;
- 2,800 outer-potential or outer-lever cells are `UNRESOLVED` potentially new syntheses;
- no cell claims historical novelty.

One cell exactly preserves the earlier SPARC focusing operator and amplitude map. Its
measurement representation changes from full SPARC baryonic acceleration to PROBES stellar
light, so this is a structural independent replay, not an identical pipeline replay.

## Quality and leakage boundary

The frozen extractor opened all 969 exploration rotation profiles and no confirmation
rotation entry. Eight hundred twenty-three galaxies with 53,341 points pass quality. Of the
146 failures, 131 fail the inclination range, 14 lack enough accepted rotation points, and
one also lacks the required radial span. Failed identities were not replaced.

The ZIP container necessarily contains the sealed files, but the entry-level receipt shows
that all 323 confirmation `*_rc.prof` entries were present and zero were opened. The model-fit
and structural-parameter tables, dynamical/dark mass, lensing mass, galaxy identity as a
numeric feature, per-galaxy gravitational constants, paid model calls, and post-response
formula generation all remained at zero.

Two post-freeze implementation repairs are disclosed:

- `41c006a6` assigns a finite placeholder only to already-ineligible `R=0` rows so their
  frozen quality mask can discard them without division by zero;
- `90c6bec7` converts canonically serialized threshold strings back to floats before an
  exact operator lookup.
- `31bebca1` makes the full replay verifier ignore only measured wall-clock duration while
  continuing to compare every scientific field and independently validate the stored
  content hash.

Neither repair changes a formula, admitted point, sample identity, gate, or candidate count.

## Results

| Evaluation | Equal-galaxy held-out MSE | Held-out R2 |
|---|---:|---:|
| Stellar Newtonian approximation | 0.103279 | -0.446 |
| Fixed stellar-only RAR | 0.029626 | 0.585 |
| Flexible local control / strongest baseline | 0.025462 | 0.644 |
| Frozen nonlocal qualifying selector | 0.022076 | 0.691 |

- The nonlocal selector improves MSE over the strongest local baseline by `13.30%`.
- The paired whole-galaxy sign-flip result is `p=0.002` over 499 frozen permutations.
- Every outer fold independently selects the same structural operator: acceleration
  occupancy at threshold `g_bar/g_dagger=1`, log-radius scale `1`, and
  `I_in-I_out`. The five training folds select different frozen amplitude-condition cells,
  so this is strong operator convergence but not yet one uniquely identified formula.
- The gain is positive in all four distance bins and both stellar-mass halves.
- Courteau97, Mathewson96, SCII, and ShellFlow improve. SHIVir's 28 quality galaxies regress
  by `0.009396` MSE and fail the frozen all-survey gate. The three Mathewson92 galaxies also
  regress descriptively but are below the frozen 20-galaxy stratum gate.
- The exact earlier SPARC focusing cell improves over fixed stellar-only RAR by `14.33%`
  and has MSE `0.025380`; this is only about `0.32%` better than the flexible local control.
- Eleven of twelve gates pass. The decision remains `REJECT`, and confirmation stays locked.

## Interpretation and next test

The result is materially better than a fit found only on a known solution set: a frozen
nonlocal operator family and the earlier exact focusing cell both predict an independent,
non-SPARC population better than the stellar RAR. The selector also beats a strong local
mass-profile model. This is evidence that interior/exterior structure may carry predictive
information; it is not evidence that gravity bends toward baryons, that the relation is
causal, or that dark matter or general relativity is unnecessary.

The SHIVir counterexample and the changing amplitude maps prevent promotion. The next Item 9
attempt must replay the frozen operator on an untouched low-mass/LSB compilation, preserve
survey-level gates, and investigate whether the failure tracks measurement pipeline,
missing gas, inclination, or genuine physical nonuniversality. It must not tune on these 823
opened galaxies or open the 323 PROBES I confirmations.

## Compute and replay evidence

The RTX 5090 evaluated 672,096,600 candidate-point cells and 10,369,800
candidate-galaxy cells in `4.387 s`, about 153.2 million candidate-point evaluations per
second. A NumPy cross-check of 128 candidates agrees with the GPU to a maximum galaxy-MSE
difference of `1.86e-14`.

- config SHA-256: `d999372e22d3b7da6fae6fc775b5a41f0ea50940a9b9c4625b03f1931ebab9a3`
- metadata source SHA-256: `593f36e60d14978ada9d0535fcff07afcdc465a33e2d92ef0b70921c4660cc3e`
- sample manifest SHA-256: `ac7f33920fcd040c985489d553fc0f1e24465cf0e272443b357892a9e50dfec5`
- candidate manifest SHA-256: `7ae63d5673c440f693ed5565c2b6540c9bc14fa0fa0f9ebb65996c2033053d3f`
- profile source SHA-256: `7732fa1d253a7ca30574f39550906048a320a2c7622cb51c4e8722193ccf2db4`
- feature table SHA-256: `b128e789fab7240ad791b76d7adbb9e6ae231053a9226800a17c92546e255b39`
- response table SHA-256: `9290cd9af3d9c28e2f15b34c26f237273916b6474427fbd6ef43e15ae05ac4b1`
- extraction receipt SHA-256: `b2a56168de6b1141e1c5910c754b5cc2d20160bcc2f7f4fe60bb596bccbb1473`
- result file SHA-256: `e86833a9a97244d01856f5005b81849d14f59974a12be3994e2c95c829512ef3`
- result content SHA-256: `8aba65a3743ec95b7de1601461a7e710e6a0fb03ea4ca8a1b06fb8edca55c6b7`
- replay command:
  `python -m sigma_theory_compiler.gravity_item9_interior_exterior check`
