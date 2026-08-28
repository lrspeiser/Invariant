# Gravity roadmap Item 6: thermodynamic-state result

## Decision

`REJECT_ITEM6_THERMODYNAMIC_STATE_EXPLORATION`

The frozen cooling-state family is **not promoted**, because it fails model-selection and
permutation gates. It is nevertheless retained as a promising, explicitly nonconfirmed
lead because it improves several held-out diagnostics. This distinction is important:
failure to promote is not the same as evidence that the idea is worthless.

## Real problem and data

The experiment asks whether intracluster thermodynamic state predicts independently
measured collisionless galaxy support after the familiar cluster temperature scale is
controlled.

- thermodynamic predictors: the [ACCEPT Chandra archive](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/182/12)
  mean temperature and fitted entropy profile
  `K(r)=K0+K100(r/100 kpc)^alpha`;
- response: the [Hectospec Cluster Survey](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJ/767/15)
  published projected galaxy velocity dispersion;
- direct nuisances and controls: redshift, ROSAT X-ray luminosity, and number of
  spectroscopic members.

Coordinate and normalized-name matching produced 28 real clusters with usable response-
blind thermodynamic predictors. A cool-core-stratified salted split reserved eight
confirmation clusters and assigned 20 exploration clusters to five whole-object folds of
four clusters each.

## Leakage and confirmation boundary

Commit `4c318784364b222927a2c2484c1d0c0e195ff94a` froze the sample, formulas,
creativity labels, models, folds, permutation scheme, and gates before any HeCS dispersion
value was requested. The experiment never acquired or used:

- HeCS caustic or NFW mass profiles;
- `M200`, `M100`, a lensing mass, or any fitted halo mass;
- cluster identity or a per-cluster coefficient;
- any of the eight reserved confirmation dispersions.

Confirmation predictors were visible during the response-blind match and quality audit;
their velocity responses remain sealed.

## Known formulas versus creative synthesis

The following are known or nonqualifying:

- the virial/beta-spec scaling `log10(sigma)=intercept+0.5 log10(kT)`;
- flexible main effects of temperature, X-ray luminosity, redshift, and member count;
- linear main effects of `K0`, `K100`, and entropy slope `alpha`.

The qualifying synthesis combined known thermodynamic ingredients into frozen nonlinear
state variables:

- approximate cooling proxy `tc = 1.5 log10(1+K0)-log10(T)`;
- core-pressure proxy `pc = 2.5 log10(T)-1.5 log10(1+K0)`;
- entropy-gradient proxy `gK = alpha[log10(K100)-log10(1+K0)]`;
- interactions `tc*alpha`, `pc*alpha`, entropy contrast times temperature, gradient times
  temperature, and `tc^2`.

These are labeled a potentially new synthesis, not a historically novel law. The cooling
quantity is a frozen bremsstrahlung-motivated proxy, not a directly measured cooling-time
profile.

## Acquisition audit

The first exploration query exposed an incomplete duplicate ACCEPT row: a `flat` fit with
blank `Nbin` accompanied the complete `extr` fit. One A1201 response had already been
requested when parsing stopped. The committed mechanical repair ignores incomplete
duplicate rows exactly as required by the preregistered `Nbin>=5` rule; it changed no
sample, formula, model, fold, or gate.

The final manifest therefore records 21 cumulative primary requests and 21 returned rows
for 20 unique exploration clusters, with zero confirmation requests.

## Results

All 20 exploration clusters passed the frozen representation and response-quality rules.

| Evaluation | Held-out MSE (dex squared) | Held-out R2 |
|---|---:|---:|
| Unrestricted selector | 0.007382 | 0.359 |
| Strongest nonqualifying selector | 0.007382 | 0.359 |
| Qualifying cooling-state selector | 0.006297 | 0.453 |

The qualifying selector used `cooling_state_coupling` in all five folds. Relative to the
foldwise strongest nonqualifying selector, it reduces MSE by `14.7%`. The gain is positive
for both cool-core and non-cool-core clusters, for both low- and high-temperature strata,
and for both the upper and lower published response-error envelopes.

However, two decisive gates fail:

- the unrestricted inner selector chooses the simpler temperature/luminosity nuisance
  model in all five folds, not a qualifying family;
- the frozen 499-permutation test gives `p=0.326`, far from the required `p<=0.05`.

Seven of nine gates pass. The unchanged result is therefore rejection, not confirmation.

## Interpretation and retained creative lead

There is a repeatable-looking held-out pattern in this small sample: adding the cooling and
core-pressure proxies improves predictions across several prespecified slices. But the
inner model selection and permutation distribution say the gain is not yet distinguishable
from a chance flexible fit with enough confidence.

Invariant should preserve this exact cooling-state family in its novelty archive with the
label `NONPROMOTED_POSITIVE_LEAD`. It must not be silently discarded, and it must not be
retuned on these 20 opened responses. It can become interesting again only through a
materially independent, frozen dataset or later action-derived theory that predicts the
same interaction before seeing new responses.

## Replay evidence

- config SHA-256: `a35b9802dac73a2763a5d195e8671fa517a580374daa548007b6aa39a2381962`
- sample manifest SHA-256: `9efd38e48dcffcee484537cc4c757073cb9d0c092d5163f44d5fff4a84b90f8d`
- source manifest SHA-256: `b03f1772d116c65970acf303d1536169420db6d053d202c63809b57b7c00fce3`
- feature table SHA-256: `44ae344743ea84cc9b46ad4afda7ba5992da9b747848cad1f5dffe49378180c2`
- extraction summary SHA-256: `1256c05866815aade4e3a83212bfd155a3b902fcaf1b7a30624450cbfbe02daa`
- result receipt SHA-256: `beed9e79a12ca53c62f3b6e8678dfab0e4f71f295bbe401a35598f4704d6085d`
- result content SHA-256: `c111d943b654d7650864548494e9783edd2551a95c0e35f23f5cb96bcdae9c71`
- replay command:
  `python -m sigma_theory_compiler.gravity_item6_thermodynamic_state replay`

No paid model calls or direct-lensing likelihood evaluations were made.
