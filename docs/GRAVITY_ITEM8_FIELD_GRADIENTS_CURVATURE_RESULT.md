# Gravity roadmap Item 8: field gradients and curvature

## Decision

`REJECT_ITEM8_FIELD_CURVATURE_EXPLORATION`

The frozen projected K-light field-gradient and curvature families do not improve prediction
of fresh 2M++ group velocity dispersions. This is a scoped rejection of the exact tested
representation and formula families, not a rejection of all field-curvature, redirected
gravity, nonlocal gravity, dark-matter, or general-relativity theories.

## Real-data problem

The test used the independent 2M++ group catalog `J/A+A/578/A61`. Before reading any
published group dispersion, the pipeline selected 131 groups with at least eight members
and exact predictor-only member counts. A salted richness-stratified split assigned 98
groups to exploration and sealed 33 for confirmation.

The only predictor columns acquired were group ID, richness, group mean velocity, member
ID, sky position, and K magnitude. The pipeline never queried member redshifts, virial
radius, virial mass, lensing mass, or the confirmation dispersions. Group mean velocity was
used only as a distance proxy. The response was the published group velocity dispersion.

Primary sources:

- 2M++ group catalog: <https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A%2BA/578/A61>
- Catalog paper: <https://doi.org/10.1051/0004-6361/201424716>

## Frozen formula boundary

The preregistration separated four known or nonqualifying controls from four qualifying
families.

Known or nonqualifying controls were the constant model; fixed `sqrt(GM/R)` virial scaling;
flexible mass, size, richness, and distance terms; and ordinary projected axis ratio and
radial concentration. `GM/R^2` and `GM/R^3` amplitudes were explicitly treated as algebraic
mass-size rewrites rather than discoveries.

The qualifying families combined:

- dimensionless center-field imbalance;
- normalized potential-Hessian trace, determinant, norm, and eigenvalue anisotropy;
- normalized third spatial derivative and field/shape alignment;
- radial acceleration-profile slopes and curvature at `0.25`, `0.5`, `1`, and `2 R_rms`;
- one combined family containing all frozen field-derivative variables.

The quantities came from a softened Newtonian point-source field built only from projected
member K light with a fixed K-band mass-to-light ratio. The radial angular integral used
512 directions after a pre-response rotation-invariance test exposed unacceptable error at
16 directions. No formula was created, removed, or changed after response access.

## Leakage boundary

Commit `f6ac47f1245a76b9cbf76f939abd2c286ef64400` froze the sample, formulas,
features, folds, null, and gates before response access. Commit
`9612571a` bound the implementation to that freeze. Acquisition then made exactly 98
one-ID queries for the exploration dispersions and zero confirmation queries.

All 98 exploration groups passed frozen quality. The 33 confirmation dispersions remain
sealed. There were zero member-redshift, virial-mass, virial-radius, lensing-mass, paid-model,
or post-response formula-generation accesses.

## Results

| Evaluation | Held-out MSE (log10 sigma squared) | Held-out R2 |
|---|---:|---:|
| Strongest nonqualifying selector | 0.026990 | 0.352 |
| Qualifying field-derivative selector | 0.029593 | 0.289 |

- The qualifying families increase MSE by `9.65%` rather than improving it.
- The unrestricted selector chooses nonqualifying mass/size or mass/size/shape controls in
  all five outer folds.
- The qualifying selector loses in all three richness strata, both concentration strata,
  and the low-mass stratum; it has only a small positive gain in the high-mass stratum.
- Removing the brightest member does not rescue the result: the qualifying MSE remains
  worse by `0.002631`.
- The 499-permutation richness-stratified test gives `p=0.762`; the observed improvement is
  negative and compatible with the frozen null search.
- Only three of eleven gates pass: complete exploration quality, positive qualifying `R2`,
  and untouched confirmations.

## Failure-space knowledge retained

The exact projected K-light center-gradient, normalized Hessian-invariant, normalized
third-derivative/alignment, and four-radius field-curvature families are now a recorded
failed region for this group-dispersion problem. Algebraic virial and acceleration
amplitudes remain known-equivalent controls, not creative discoveries. Retuning this family
on the 98 opened responses would invalidate the counterexample and is forbidden.

This result does not test complete baryonic mass, member line-of-sight geometry, a covariant
curvature scalar, external structure, boundary focusing, interior/exterior nonlocality,
history, resonance, modified inertia, or alternative field equations. Those remain separate
roadmap regions.

Item 9 now tests a materially distinct hypothesis: whether baryons inside and outside a
radius enter through a universal nonlocal balance or redirection term on a fresh resolved
real-data problem.

## Replay evidence

- config SHA-256: `72c22188c1aacae9d8e56623ec5b9be1236daf2b15a40db8de700afcacead20b`
- sample manifest SHA-256: `10d11c579802b02d62ec4a48152b3cc73e6859a77c9eb6cf0e775320c5abdc07`
- predictor source SHA-256: `91c338cbd58807652313a77d9bdb3f14b542287ece6c31c68c7ca31ce1ed9819`
- primary feature table SHA-256: `0c47539c78faaf51262bd7ef58e175262be8f26a6614c5c723b6da4935d0db0e`
- brightest-removed feature table SHA-256: `d5ee00650004a496e27d56e3aea581e6dcae1051d95b610eca8ccb47824265ea`
- predictor extraction SHA-256: `3812c1600a3bcf72e18c0f545348b431777590574e4c91128552b809ba7bcb51`
- exploration response source SHA-256: `4f171210f2ae0cce360a623dada4c9d26dc6a22bbb1a14a442480a5bae59c7dc`
- result file SHA-256: `2c97f701d4119fb5b8dc628c9f0b2579919cc2c8b3da38bb510e9d889bc3510b`
- result content SHA-256: `d07841c5a10415583613e3ce1583a8d5124995fce90f8fdd2bb411b1751783ff`
- replay command:
  `python -m sigma_theory_compiler.gravity_item8_field_gradients_curvature check`
