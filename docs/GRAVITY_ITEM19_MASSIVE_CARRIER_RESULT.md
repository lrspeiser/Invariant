# Gravity roadmap Item 19: massive gravitational particles

## Decisions

- Universal-gravity track: `REJECT_ITEM19_MASSIVE_CARRIER_GRAVITY_EXPLORATION`
- Phenomenon/publication track: `NO_ITEM19_EMPIRICAL_PUBLICATION_LEAD`

The frozen massive-carrier library predicts the fresh galaxy responses worse than the strongest
ordinary baryonic baseline. This closes the exact static linear Yukawa region tested here; it does
not make other massive phases, interference mechanisms, polarizations, nonlinear screening, or
time-dependent carriers less viable.

The run also leaves a useful failure-space certificate: a nonnegative mixture of attractive
Yukawa exchanges fades with separation. At the four exponential-disk component radii used here,
the verified Yukawa disk response is between zero and the Newtonian response, so an attractive
mixture cannot produce more force than calibrated GR with the same baryonic masses. This is a
scoped mathematical exclusion, not a historically novel theorem or a paper claim by itself.

## Two equally viable mechanism niches

The weak-field actions were frozen before any selected rotation response was opened:

`S_scalar = integral[-(partial phi)^2/2 - m^2 phi^2/2 + q phi rho] d4x`

and

`S_vector = integral[-F_mn F^mn/4 + m^2 A_m A^m/2 + q A_m J_b^m] d4x`.

The first gives attractive exchange and the second gives repulsive exchange between like
universal charges. For either sign, the point potential and force kernel are

`Phi = -G_bare M [1 + s alpha exp(-r/lambda)] / r`

and

`f(u) = (1 + u) exp(-u)`, where `u = r/lambda` and `s` is `+1` for the
attractive scalar or `-1` for the repulsive vector.

After normalizing to the measured short-distance Newton constant, an arbitrary admitted mixture
has

`mu(r) = [1 + s sum_j alpha_j f(r/lambda_j)] / [1 + s sum_j alpha_j]`.

Both signs were treated as live hypotheses. The repulsive branch was not mislabeled as a new
formula. Algebraically,

`mu(r) = 1 - sum_j [s alpha_j/(1+s sum alpha)] [1-f(r/lambda_j)]`.

For `s=-1`, this is exactly a positive sum of Item 16's subtracted-Yukawa bases. The one-carrier
branch is labeled `ALGEBRAIC_REWRITE_OF_ITEM16_SUBTRACTED_YUKAWA_WITH_DISTINCT_ACTION_SIGN`; the
two-carrier branch is a known-family combination. The scalar branches are also known formulas or
known-family combinations. Historical novelty is explicitly false.

## Exponential-disk derivation

The experiment did not multiply a Newtonian rotation curve by an arbitrary radial factor. It
convolved the massive propagator with the frozen exponential stellar and gas disks. For each
component,

`gY(R) = 2 pi G integral_0^infinity dk k^2 J1(kR) Sigma_tilde(k)
         / sqrt(k^2 + lambda^-2)`.

The final one-radius prediction was

`gpred = [gN_star + gN_gas
          + s sum_j alpha_j (gY_star_j + gY_gas_j)]
         / [1 + s sum_j alpha_j]`.

The primary radius was `2.2 hR`; the gas scale was `2 hR`, with `1.5 hR` and `3 hR`
replays. The dimensionless kernel table covered `Rd/lambda = 10^-6` through `10^12` at all four
required component radii. Its massless-limit error was `3.14e-7` and its maximum scaled
interpolation error was `2.05e-4`.

For point sources, `f'(u) = -u exp(-u) <= 0`. For the evaluated disk kernels, the verified ratio
was `6.30e-13 <= gY/gN <= 1`. Convexity then certifies that every nonnegative attractive mixture
in this domain is no stronger than same-calibration GR. A numerical audit of 4,096 attractive
cells found a maximum log-velocity excess of `-4.44e-13`.

## Frozen fresh-data boundary

Corrected scientific freeze `f6726c7271c01ac87368baeedefa656bf665602c` fixed the actions,
formula grammar, local filters, disk convolution, candidates, baselines, gates, and two-track
decisions. Sample freeze `7dc0bcdeeede74af01349d94dd4e0429f03c5412` then fixed every identity
before response access.

Predictors came from:

- DiskMass 2010 photometry and disk geometry, VizieR `J/ApJ/716/198/table2`;
- Springob et al. 2005 corrected integrated H I flux, VizieR `VIII/77/table3`;
- the 2025 DiskMass K-band/distance table, VizieR `J/ApJS/276/59/sample`.

The predictor request explicitly omitted every H I line width. The response phase later requested
only the 2025 H-alpha velocity-field parameters needed to compute
`V(2.2hR) = Vrot tanh(2.2hR/hrot)`.

There were 66 complete predictor joins. Before roles were assigned, the code excluded:

- all 43 Item 18 identities, including both exploration and reserved confirmation roles, of which
  27 occurred in this join;
- all earlier response-exposed or search-snippet-exposed identities, of which 19 occurred in this
  join.

That left 20 fresh objects. Predictor-only mass strata and HMAC ranks assigned 16 exploration
objects, four per outer fold, and four reserved confirmations. No confirmation query was issued.

### Predictor-only correction before the final freeze

An initial predictor-only preparation attempt found that six nominal joins lacked `B-K`, although
the sample rule did not require color. No rotation response had been requested. The nuisance
baseline was therefore corrected from five predictors to the same four complete predictors used
in Item 18: K luminosity, H I mass, disk scale, and surface brightness. The exact contract was
refrozen at the corrected scientific commit above before the final sample was prepared. This was
an outcome-blind schema correction, not post-response tuning.

## Search and physical filters

PCG64 seed `191901` generated 262,144 programs across four equally represented niches:

- attractive scalar, one carrier;
- attractive scalar, two carriers;
- repulsive vector, one carrier;
- repulsive vector, two carriers.

The ranges spanned `10^-9` to `10^4 kpc` before filtering, corresponding to finite carrier masses
through `m c^2 = hbar c/lambda`. Couplings, ranges, component weights, a universal stellar
mass-to-light ratio, and a universal gas scale were all fixed program parameters; none varied by
galaxy.

The pre-response filter required positive mass squared, positive net short-distance force, no more
than `1e-5` fractional variation at the frozen Solar-System radii, a laboratory-normalization
error below `1e-12`, a finite large-distance enhancement below 20, and positive predicted force
over the disk domain. It admitted 183,848 cells, all of which are exact parameter-equivalence
classes; an independent tuple audit found zero duplicates. Family counts ranged from 44,308 to
47,975, so neither sign nor carrier count was starved.

## Quality result

Ten of the 16 exploration galaxies met every frozen response-quality rule, exactly the frozen
minimum. The held-out folds retained 4, 1, 2, and 3 galaxies. Five failures carried the published
low-inclination flag; one failed both velocity-field asymmetry thresholds. The uneven and small
valid folds limit the reach of any result, even though the formal quality gate passes. All four
confirmations remain sealed.

## Held-out empirical result

| Model | Held-out MSE | Massive-carrier change |
|---|---:|---:|
| Frozen massive-carrier selector | 0.47910 | — |
| Fixed baryon-only GR | 0.40215 | **19.13% worse** |
| Globally calibrated baryon-only GR | 0.38405 | **24.75% worse** |
| Baryonic Tully-Fisher | 0.42603 | **12.46% worse** |
| Flexible four-predictor nuisance | 1.55433 | 69.18% better |

The flexible nuisance model was unstable and poor in this ten-object, four-fold problem; calibrated
GR was the strongest ordinary baseline. Nine of ten valid galaxies were individual
counterexamples where the selected carrier prediction had larger squared error than calibrated
GR.

The negative result was broad:

- low- and high-mass halves were 36.03% and 23.43% worse than calibrated GR;
- low- and high-gas-fraction halves were 67.28% and 22.26% worse;
- the `1.5 hR` and `3 hR` gas-profile replays were 28.54% and 16.03% worse;
- the selection-aware 199-trial residual permutation gave `p = 0.925`;
- the primary carrier ranges did not cluster across folds.

Three folds selected attractive carriers, with total couplings `6.73` to `9.59` and component
ranges near `0.08` to `1.87 kpc`; the fourth selected the known
repulsive rewrite with coupling `0.767` and range `10.08 kpc`. The attractive cells paired their
weaker force with much larger stellar mass-to-light ratios. Calibrated GR instead selected the
minimum frozen stellar mass-to-light ratio in three folds. This is the expected baryonic-calibration
degeneracy, not a stable massive-particle signature.

Eight of 16 universal-gravity gates and three of seven phenomenon/publication gates passed. The
carrier model therefore advances on neither empirical track.

## Controls and compute

The synthetic repulsive-carrier injection was recovered exactly in all four folds, with the correct
polarity and range, and improved on GR by 100%. The pure-GR control did not spuriously prefer a
carrier. CPU and GPU log predictions agreed to `8.88e-16`.

The RTX 5090 evaluated 183,848 admissible cells on the ten valid galaxies and performed an
estimated 1,176,627,200 null-inclusive training-residual evaluations. Matrix construction took
`0.161 s`, the 199 full-search null screens took `2.257 s`, and the complete run took `3.841 s`.
There were zero paid model calls and zero API spend.

## What is excluded, retained, and still open

Retain as a failure-space result:

1. A positive-weight attractive point-source Yukawa spectral mixture fades with separation.
2. For the frozen exponential-disk radii and kernel range, any nonnegative attractive mixture is
   no stronger than same-calibration GR.
3. The frozen one-/two-carrier scalar and repulsive-vector library does not predict these fresh
   galaxy amplitudes beyond ordinary global baryonic calibration.

Do not generalize this result to:

- a massive mode active only in a derived phase or period (Item 20);
- interference or crossover beyond the algebraically equivalent static sums (Item 21);
- independently derived scalar, vector, or tensor polarization and lensing responses (Item 22);
- nonlinear screening, environmental mass, time dependence, memory, or nonlocal modes;
- full resolved rotation curves, galaxy clusters, direct lensing, binary pulsars, gravitational
  waves, stability, causality, strong fields, or cosmology;
- baryonic mass estimates outside the frozen global calibration grid.

The next numbered experiment is Item 20, massive phases or periods, on a new response. The exact
Item 19 formulas and all ten opened valid responses must not be retuned, and the four confirmations
must remain sealed. Separately, the Item 12/13 age association remains active on its independent
phenomenon/publication track; this Item 19 negative does not change its viability.

## Replay evidence

- result receipt: `runs/gravity/roadmap/item-19-massive-carrier-v1.json`;
- source receipts: `runs/gravity/roadmap/item-19-massive-carrier-v1-source/`;
- replay command: `python -m sigma_theory_compiler.gravity_item19_massive_carrier check`;
- paid API calls: `0`;
- reserved-confirmation target accesses: `0`.
