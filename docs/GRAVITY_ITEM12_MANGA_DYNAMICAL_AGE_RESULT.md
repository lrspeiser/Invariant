# Gravity roadmap Item 12: dynamical age

## Decision

`PASS_ITEM12_MANGA_DYNAMICAL_AGE_EXPLORATION`

The frozen MaNGA exploration passes all 13 preregistered gates. A consensus of integrated
stellar-population clocks, interacting with a stellar surface-density proxy, adds reproducible
held-out information about stellar velocity dispersion beyond the structural baseline. This is
a candidate association for independent confirmation, not a causal age measurement, a novel
formula claim, or evidence for modified gravity.

## Frozen test

Before reading any galaxy-level MaNGA predictor or response row, commit
`a50af163f819ec757e8aeedb5aaa1124f5b47c2f` froze:

- the 10,782-row SDSS DR17 MaNGA DAPall/DRPall join and exact DAP type;
- predictor/response separation and zero-quality-bitmask requirements;
- exclusions against prior MaNGA identities and all declared predecessor coordinates;
- a 1,000-galaxy sample balanced by Dn4000 half and stellar-mass half;
- nine structural controls and fixed, non-data-derived clock normalizations;
- 262,144 PCG64 candidates across twelve provenance-labeled age/settling mechanisms;
- five-fold nested selection, one scalar clock coefficient, quality floors, and all gates.

The predictor-only query returned all 10,782 rows. Of these, 6,333 passed the frozen predictor
rules. The independence audit left 6,099 eligible galaxies after excluding 69 prior MaNGA
identities, 20 predecessor-coordinate matches, and duplicate observations or identities. The
exact 750 exploration and 250 reserved-confirmation identities, together with every candidate
cell, were committed at `5d70c33d99edfde710b2cd765a4ae3191f7e19d1` before any selected
dynamical response was requested.

## Variables and creative grammar

The structural ridge baseline uses stellar mass, angular half-light radius, their surface-density
proxy, axis ratio, Sérsic index, `g-r` color, redshift, surface brightness, and signal quality.
It does not use object identity, stellar velocity dispersion, or velocity span as a predictor.

The creative grammar combines integrated Dn4000/D4000, H-delta, H-gamma, H-beta, H-alpha
equivalent width, specific star formation, and a mass-size crossing proxy through:

- monotonic age and young-burst transforms;
- post-starburst and quenching boundaries;
- depletion and age/SFR coherence clocks;
- settling, crossing-time, hysteresis, and rejuvenation terms;
- multi-clock consensus and log-periodic clocks.

Every clock uses a center and scale frozen in the config, so no held-out predictor distribution
or response determines its normalization. Cells are labeled `KNOWN_FORMULA_TRANSFORM`,
`KNOWN_FAMILY_COMBINATION`, `COMBINATION`, or `UNRESOLVED`; 108,642 cells are labeled
`UNRESOLVED`. No historical novelty is claimed, and no formula was generated after response
access.

## Quality result

Ten exact response queries of 75 exploration identities returned all 750 requested rows. No
reserved-confirmation identity was queried. Of those responses, 585 pass every frozen quality
check, a 78% retention rate above the 450-galaxy and 60% floors. The five held-out folds contain
124, 120, 116, 105, and 120 valid galaxies. Most failures are the preregistered signal-quality
floor; six fail the dispersion range and a small number fail missing-value, velocity-span, or
fit-quality checks.

## Held-out result

| Model | Held-out MSE | Held-out R2 |
|---|---:|---:|
| Structural ridge baseline | 0.00919765 | 0.7949 |
| Nested selected spectral-clock term | 0.00751216 | 0.8325 |

The selected term reduces MSE by `18.33%`. The mean per-galaxy MSE gain is positive with paired
sign-flip `p=0.001`. It is positive in both halves of Dn4000, stellar mass, Sérsic index, and
redshift. All 13 gates pass.

All five outer folds independently choose the `spectral_clock_consensus` family with
`stellar_surface_density` modulation and positive fitted coefficients from `0.0416` to `0.0444`.
In schematic form, the selected family is:

`log10(sigma*) = structural baseline + b * standardize[C(clock consensus) * tanh(surface proxy)]`

where `C` is a bounded signed-power transform of a fixed-normalized Dn4000/D4000-minus-Balmer
consensus. This is an existing-family combination. Exact threshold, scale, power, and selected
ordinal vary across folds, so no single post-response parameter cell is promoted.

The RTX 5090 screened `3,067,084,800` candidate-galaxy validation combinations in `4.64`
seconds with CuPy 13.5.1. The maximum CPU/GPU component discrepancy was `2.84e-14`.

## Retained lead and counterexamples

Record `SPECTRAL_CLOCK_CONSENSUS_TIMES_STELLAR_SURFACE_DENSITY` as
`PROMOTED_TO_INDEPENDENT_CONFIRMATION_QUEUE`, with origin `COMBINATION`.

Do not retune it on the 585 opened responses or open the 250 sealed confirmations. A later
confirmation must freeze a deterministic family-level parameter rule before a fresh response
and distinguish the lead from these alternatives:

- stellar-population-dependent mass-to-light or stellar-mass errors;
- survey, redshift, aperture, angular-size, or signal-quality systematics;
- ordinary merger and relaxation history;
- star-formation history that correlates with dynamics without causing it.

The following observations bound the lead: 165 selected responses fail quality checks, exact
parameter cells are not identical across folds, and the response is integrated stellar
dispersion rather than a resolved rotation curve, cluster observable, or lensing map.

## Boundaries

- Integrated spectral indices are proxies, not direct formation or settling times.
- The surface term uses mass divided by angular radius squared, with redshift separately
  controlled; it is not a calibrated physical surface density.
- This test does not predict galaxy rotation curves or address galaxy clusters or lensing.
- No dynamical-age cause, alternative to general relativity, or historical novelty is
  established.
- Zero confirmation responses were opened and zero paid model calls were made.

## Next real test

Advance to Item 13, relaxation and mergers, using a fresh response. Freeze asymmetry,
disturbance, close-pair, merger-stage, and kinematic-relaxation predictors before response
access. The test must determine whether these ordinary-history variables explain or preserve
the Item 12 spectral-clock association while retaining mass, geometry, color, redshift, and
survey controls.

## Replay evidence

- result file SHA-256:
  `d134a8c9f5cd3e87d25bdb0cf38b390ca45ac85a9387abae412d95b5f109d2bc`
- result content SHA-256:
  `d2638ae1c05f4fa96124b91c79aec6f73a73a225eab1b86b772fd91cdd34f62c`
- response-source SHA-256:
  `91a095d21b2e252c8e707aefa4ad8b1378144f1a97dfef8cc1a77b223df42544`
- replay command:
  `python -m sigma_theory_compiler.gravity_item12_manga_dynamical_age check`
