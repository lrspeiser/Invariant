# Gravity roadmap Item 15 attempt 1: galaxy timescale ratios

## Decision

`INCONCLUSIVE_ITEM15_MANGA_TIMESCALE_QUALITY`

The frozen galaxy-timescale grammar shows a positive diagnostic direction but cannot be promoted.
Only 123 of 240 fresh exploration galaxies pass the joint stellar/H-alpha quality rules, below
the frozen floor of 150 and 62.5% retention. On the valid subset, nested timescale selection
improves held-out stellar MSE by 5.39% and transfers a 0.79% gain to H-alpha without reselection,
but the primary paired test gives `p=0.225`, several broad strata regress, and fold formulas are
not stable.

Record this as an inconclusive, nonpromoted hint. It is not a formula discovery, a causal clock,
or evidence for modified gravity. Item 15 remains open because this galaxy source has no direct
hot-gas cooling time; the frozen contract explicitly requires a separate cooling-time lane before
a broad Item 15 synthesis.

## Frozen test

Before opening any fresh velocity map, commit
`bfccc9118a637dc5438f60a7f935044170c8a396` froze:

- all predecessor identity and 60-arcsecond coordinate exclusions;
- physical definitions and constants for baryonic dynamical, crossing, orbital, free-fall,
  stellar mass-doubling, two-body relaxation, and cosmic times;
- the rule that orbital, crossing, free-fall, and baryonic dynamical times form one algebraic
  equivalence class rather than four independent laws;
- a strong baseline containing every mass, size, star-formation, age-proxy, morphology, and
  redshift variable from which the ratios are derived;
- twelve provenance-labeled timescale families and 262,144 seeded cells;
- nested five-fold selection, H-alpha transfer without reselection, strata, quality floors,
  permutation testing, and claim boundaries;
- zero confirmation access, zero post-response formula generation, and zero paid model calls.

The response-free source contained 2,366 fresh eligible galaxies after excluding 1,655 prior
identities, 17 coordinate overlaps, 1,543 unsuitable axis ratios, and one specific-SFR floor.
The exact 320-galaxy sample has 80 galaxies in each fast/slow-growth by lower/higher-mass cell,
with 240 exploration and 80 reserved confirmation. Every fold has 48 exploration galaxies.
The identities, all derived timescales, and all candidate cells were committed at
`73b9dc23f6951c8e0e490b9a4a55394c0b7509fb` before any selected MAPS payload was requested.

## Timescales and equivalence

For stellar mass `M` and half-light radius `R`, the baryonic dynamical clock is

`t_dyn = sqrt(R^3 / (G M))`.

Under the frozen definitions, crossing time is `t_dyn`, orbital time is `2*pi*t_dyn`, and
uniform-density free-fall time is `pi/sqrt(8)*t_dyn`. They therefore cannot add independent
information or count as distinct laws. The qualifying ratios instead combine:

- mass-doubling time `1/sSFR` with dynamical and cosmic clocks;
- classical two-body relaxation `0.1 N/ln(N) * t_dyn` as a collisionless null;
- cosmic age in a frozen flat `Omega_m=0.3`, `Omega_Lambda=0.7`, `H0=70` cosmology;
- clock hierarchy, crossover, localized-shell, entropy, coupling, phase, and log-periodic forms.

The baseline already contains the source variables. A candidate can therefore help only through
a stable nonlinear organization of those variables, not by silently reintroducing mass, size,
specific SFR, redshift, or the prior age lead.

## Quality result

All 240 exploration MAPS files were downloaded and hash recorded. Only 132 contain both usable
stellar and H-alpha annuli; 101 galaxies fail specifically for too few qualifying inner H-alpha
measurements. After span and ratio cuts, 123 galaxies pass, or 51.25%.

Every fold retains 20–30 galaxies, but the short-dynamical-time stratum retains only 17, below its
frozen floor of 20. Thus both the total/retention gate and a stratum gate fail. No quality rule was
weakened after response access.

## Prespecified diagnostic

| Response and model | Held-out MSE | Held-out R2 | Relative change |
|---|---:|---:|---:|
| Stellar source-variable control | 0.00618508 | 0.2285 | — |
| Stellar control + selected timescale | 0.00585170 | 0.2701 | 5.39% better |
| H-alpha source-variable control | 0.00782634 | -0.0143 | — |
| H-alpha + inherited stellar-selected timescale | 0.00776446 | -0.0062 | 0.79% better |

The primary paired mean gain is positive but not significant (`p=0.225`). Fast-growth and
lower-mass galaxies improve, while slow-growth and higher-mass galaxies regress. Both dynamical
time halves improve, but one contains only 17 objects. The lower-cosmic-age half regresses
slightly. Four folds select different log-periodic cells and one selects a localized timescale
shell; fitted stellar coefficients reverse sign across folds. Four of five stellar/H-alpha
coefficient signs agree, but the H-alpha model remains below zero held-out `R2`.

The RTX 5090 evaluated `644,874,240` candidate-galaxy score combinations in 3.22 seconds with
CuPy 13.5.1. The maximum CPU/GPU component difference was `2.22e-15`.

For context, the quality subset has median clocks of 0.070 Gyr dynamical, 69.9 Gyr mass-doubling,
13.01 Gyr cosmic age, and 5.59 million Gyr classical two-body relaxation. The enormous relaxation
time is exactly why it is treated as a null rather than a plausible ordinary galaxy-aging clock.

## Boundaries

- Specific SFR gives a mass-doubling time, not a direct stellar formation age.
- Classical two-body relaxation is far too slow to be an ordinary galactic settling mechanism.
- This source contains no direct hot-gas density/temperature cooling-time measurement.
- A log-periodic selected cell is a screened empirical transform, not a detected periodic force.
- The response is an annular line-of-sight velocity span, not deprojected circular speed.
- The sample and response remain within the SDSS MaNGA ecosystem despite fresh identities.
- Zero confirmation responses were opened, zero formulas were generated after response access,
  and zero paid model calls were made.

## Required next test

Keep Item 15 open. Freeze a materially independent cluster or hot-gas test with direct cooling,
free-fall/crossing, and cosmic times, plus a lensing or cluster-dynamical response. The test must
exclude every prior cluster identity, preserve the galaxy attempt as a nonpromoted hint, and
decide Item 15 only after the direct-cooling lane has a replayable real-data result.

## Replay evidence

- result file SHA-256:
  `2308098ec13e76459cd925e4d27e3c4ffc4edbe94d700cd931bf94864ec20702`
- result content SHA-256:
  `eff43e00423c51e2f06e2a6c6db275cc77ad0d1c328ae5a30887b3ff16efb351`
- response-source SHA-256:
  `8c979711870c95f99d177c400f40c98b48a80903441211801c1da52c18ef5890`
- extraction-summary SHA-256:
  `57d73780ac96c681af4589d80109a4d8747092e08688cfa33dd87ffecf06b297`
- sample-manifest SHA-256:
  `b960f9fe364e0544517d4643a35c26e70f048dfb17ee8921898b506da8b07a6f`
- predictor-source SHA-256:
  `db761d6ec60bbb2dbe2e6cd4b094220582d03781b65aa37ff72b7ba3a745a747`
- candidate-manifest SHA-256:
  `23b81a8bd008582f74e8d1d9b91e6ba0af9035cc8c45b37c6430a44dbce2ec75`
- replay command:
  `python -m sigma_theory_compiler.gravity_item15_manga_timescale_ratios check`
