# Gravity roadmap Item 5 attempt 2: cluster pressure cross-support result

## Decision

`REJECT_ITEM5_PRESSURE_CROSS_SUPPORT_EXPLORATION`

This is a scoped rejection of the frozen mapping from SPT fixed-aperture thermal SZ
observables and matched-filter coherence to galaxy velocity dispersion. It is not a
rejection of the virial theorem, radially resolved pressure support, every possible
thermal/collisionless coupling, dark matter, or general relativity.

## Why this was a genuinely different second attempt

Attempt 1 used radial H I rotation, dispersion, and surface-density profiles in LITTLE
THINGS dwarf galaxies. Its unsmoothed pressure derivative failed the representation
quality gate for six of eleven exploration galaxies, so the result was inconclusive.

Attempt 2 changes both physical system and measurement pipeline. It asks whether direct
thermal pressure support in galaxy clusters predicts independently measured collisionless
galaxy support:

- predictors: Bleem et al. (2015) SPT-SZ fixed-0.75-arcmin `YSZ`, `e_YSZ`, multi-scale
  detection significance `xi`, and selected matched-filter scale `theta`;
- primary response: Bayliss et al. (2017) measured line-of-sight galaxy velocity
  dispersion;
- response robustness: the separately published Bayliss et al. (2016) and Ruel et al.
  (2014) gapper and biweight estimates.

The source catalogs are the official [SPT-SZ catalog](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/216/27),
[Bayliss 2017 catalog](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJ/837/88),
[Bayliss 2016 catalog](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/227/3), and
[Ruel 2014 catalog](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJ/792/45).

## Frozen sample and leakage boundary

Identifier and predictor-quality metadata gave 80 SPT/Bayliss overlaps. The frozen rule
required at least 20 spectroscopic members, positive `YSZ` and uncertainty,
`YSZ/e_YSZ >= 3`, and a positive spectroscopic redshift. Sixty-two clusters qualified.
A salted split assigned 44 to exploration and 18 to reserved confirmation; five outer
folds hold out whole clusters in counts 9, 9, 9, 9, and 8.

Before velocity-response access, commit
`e7a861b656cda74e9e92ea63cd864cfcfdf2555e` froze the sample, formulas, model families,
cross-validation, permutations, gates, and creativity labels. The following were never
requested or used:

- SPT-inferred `M500c` or its uncertainty;
- Ruel's SZ-mass-derived `sigmaSPT`;
- any lensing, dynamical, or fitted halo mass as a predictor;
- cluster identity or per-cluster coefficients;
- any of the 18 reserved confirmation velocity responses.

Confirmation predictors had been seen during response-blind sample auditing, so the
receipt truthfully marks them as not predictor-blinded. Their velocity responses remain
sealed, which is the inferential boundary that matters here.

## Frozen derivation and creativity boundary

The known self-similar control was

`log10(sigma) = intercept + 0.2 log10[YSZ D_A(z)^2 E(z)]`.

Flexible regressions on raw SZ amplitude, distance, redshift, significance, and filter
scale were nonqualifying empirical controls. Pressure amplitude divided by filter area
was also nonqualifying because it is a compactness-style rewrite.

The potentially new synthesis was the nonlinear interaction among:

- physical filter extent `L = log10[D_A sqrt(theta^2 + 0.25^2)]`;
- pressure-surface proxy `Q = log10[YSZ/(theta^2 + 0.25^2)]`;
- detection/aperture coherence `C = log10[xi/(YSZ/e_YSZ)]`;
- aperture/filter ratio `R = log10[0.75/sqrt(theta^2 + 0.25^2)]`;
- the frozen interactions `C L`, `C Q`, `C R`, `Q R`, and `C^2`.

These ideas qualified only as a potentially new cross-domain synthesis of known
observables. The test does not establish historical novelty.

## Acquisition audit

The acquisition had two transparent mechanical failures after the scientific freeze:

1. Bayliss labels three exploration systems `ACT-CLJ` although the same systems have SPT
   identifiers. The first run stopped at the first such object after issuing eight primary
   response requests, seven of which returned rows.
2. After the identifier-only repair, all 44 exploration records were retrieved, but the
   canonical receipt layer rejected unstringified floating-point values before writing.

Neither repair changed a sample member, formula, model, fold, or gate. The successful
manifest therefore records the cumulative history: 96 primary response requests, 95
returned primary rows including duplicates, 192 robustness requests, 44 unique primary
exploration records, and zero confirmation requests. All raw floating values are stored
as canonical decimal strings. This duplication is an audit cost, not extra evidence.

## Results

All 44 exploration clusters passed the frozen quality contract. The source yielded 92
alternative gapper/biweight response values for robustness.

| Evaluation | Held-out MSE (dex squared) | Held-out R2 |
|---|---:|---:|
| Unrestricted selector | 0.0103073 | -0.0404 |
| Strongest nonqualifying selector | 0.0103073 | -0.0404 |
| Qualifying pressure-coherence selector | 0.0102098 | -0.0306 |

The qualifying selector's apparent MSE gain over the foldwise strongest baseline is only
`0.0000975`, or `0.946%`. It loses directly to both the flexible-SZ and redshift-only
baselines. The unrestricted selector chooses no qualifying family in any of five folds.

Additional failure evidence:

- qualifying held-out `R2` is negative overall;
- qualifying `R2` is negative in both frozen redshift strata (`-0.155` below `z=0.6` and
  `-0.042` at or above `z=0.6`);
- the redshift-stratified 499-permutation test gives `p=0.43`;
- on 92 alternative dispersion estimates, the qualifying family is worse by
  `0.000307 dex^2` MSE;
- only the representation-quality and sealed-confirmation gates pass: 2 of 9.

## What the result means

The simple direct SPT summaries do not reliably predict object-by-object galaxy velocity
dispersion in unseen clusters in this sample. Nonlinear coherence and phase-balance terms
do not rescue the relation. The likely limitations include the fixed angular aperture,
the discrete matched-filter scale, a restricted high-mass selected population, projection,
member-velocity scatter, and the fact that these are not full radial pressure profiles.

The scientifically useful output is therefore an excluded equation region: self-similar
fixed-aperture scaling, flexible raw-main-effect scaling, pressure/area compactness, and
the exact frozen nonlinear coherence interactions cannot be promoted on these data. A
future idea must add materially new measurements or dynamics, not retune these opened
responses.

## Replay evidence

- config SHA-256: `7f35de22fec23d986457a02c6293daf684252edf10445bc52f327e104557c03b`
- sample manifest SHA-256: `54b8778b60701af1da8cfd5e98511acc321f4a80ebd882ead5cae57ef61f0397`
- source manifest SHA-256: `31b904b669a58514a6a002d3ab4788bf4399571cd35b1f555a3bd0a0c3fea67d`
- feature table SHA-256: `002bcc2f73cd1f9828047e661b47b7c6da3c441285da95a3d695b6484ecc6583`
- extraction summary SHA-256: `0c3a745101ea89b972c9a2c9875a2339047289d87a26184b1433dffb41ab5492`
- result receipt SHA-256: `f37bb3d5deb941e01fe2d1034892fab6626a9ef02d515a2de4c78009b3167835`
- result content SHA-256: `957ce9138b7b22bae5d7fba46284b8fb9147360b40a3ae2836dc7599b7afbfdb`
- replay command:
  `python -m sigma_theory_compiler.gravity_item5_pressure_cross_support replay`

No paid model calls or direct-lensing likelihood evaluations were made.
