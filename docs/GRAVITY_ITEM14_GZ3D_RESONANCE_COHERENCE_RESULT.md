# Gravity roadmap Item 14: resonance and coherence

## Decision

`REJECT_ITEM14_GZ3D_RESONANCE_COHERENCE_EXPLORATION`

The frozen static spiral/bar mask grammar does not improve held-out prediction of resolved
outer-to-inner stellar velocity-span ratios beyond structural, catalog-vote, and frozen-age
controls. It makes stellar MSE 2.58% worse, with paired `p=0.827`, and makes the preregistered
H-alpha transfer MSE 0.10% worse without candidate reselection.

The resolved phenomenon itself is real in this sample: the median outer-to-inner line-of-sight
velocity-span ratio is 1.356 for stars and 1.231 for H-alpha. The rejection means that the tested
static image geometry does not explain which galaxies have larger ratios. It does not reject
true time-dependent resonance, pattern-speed physics, other collective modes, or modified
gravity.

## Frozen test and sample revision

Before reading any GZ3D metadata row, morphology-mask pixel, or MaNGA velocity-map payload,
commit `095c7ddfc4ca16c8ae1e4aeaafbfbd180e902197` froze:

- the exact official GZ3D metadata/checksum sources and MaNGA MAPS response definition;
- a resolved stellar primary response and independent H-alpha secondary response;
- all predecessor identity and 60-arcsecond coordinate exclusions;
- twelve provenance-labeled mask-geometry families and 262,144 seeded candidate cells;
- nested five-fold selection, fixed normalizations, quality rules, strata, and admission gates;
- a strict prohibition on confirmation access, post-response formulas, and paid model calls.

The initial metadata-only feasibility pass exposed two implementation facts before any mask or
velocity pixels were read. First, GZ3D metadata filenames require the official `gz3d_` prefix and
`.gz` suffix to join the checksum registry. Second, the original bar-vote/mass cutoffs produced
only 40 low-bar, higher-mass galaxies, below the frozen 80-per-cell design. A disclosed,
response-blind grid over only those two cutoffs uniquely maximized the smallest cell at bar vote
`0.28` and log stellar mass `10.1`, yielding predictor-only cell counts of 148, 137, 136, and 170.
That revision was committed at `63afedca6b276ccdfe1d3eb11ee769df6da294fd` before any mask
pixel or response was opened.

The mask pass then rejected four files with too few nonzero spiral pixels and selected exactly
320 galaxies: 80 in every bar/mass cell, 240 exploration and 80 sealed confirmation. Each outer
fold has 48 exploration galaxies. The exact identities, 320 checksum-verified mask features,
and all candidate cells were committed at
`f05ee9e367d781538c745c28ce0f62c5e2b65dc4` before any selected MAPS payload was requested.

## What was tested

The primary response is

`log10(stellar outer annular velocity span / stellar inner annular velocity span)`,

where the inner annulus is `0.2–0.8 Re`, the outer annulus is `0.8–1.5 Re`, and each span is the
unweighted 90th-minus-10th percentile of unique stellar-continuum bins. Quality requires at least
eight unique bins and three azimuth quadrants in each annulus, valid inverse variance and masks,
stellar S/N at least five, and bounded fit quality and span ratios.

The secondary response applies the fold-selected formula, without reselection, to the analogous
H-alpha ratio. It requires at least 20 qualifying spaxels and three quadrants per annulus,
amplitude-to-noise at least five, equivalent width at least 3 angstroms, and bounded local fit
quality.

The structural control contains mass, size, surface density, axis ratio, Sérsic index, color,
redshift, surface brightness, signal quality, morphology, concentration, spiral/bar vote
fractions, and the frozen Item 12 age lead. The creative component searches fixed-normalized
transforms and combinations of:

- two-, three-, and four-arm amplitudes and their hierarchy;
- radial phase locking, pitch/twist coherence, and radial mode persistence;
- bar/spiral phase locking and bar-to-arm radial ratios;
- harmonic coupling, mode-entropy suppression, and log-periodic arm phase;
- no modulation or modulation by bar vote, surface density, mass, prior age, or arm coverage.

Of the 262,144 seeded cells, 87,356 are labeled `UNRESOLVED`, 65,618 known formula transforms,
65,403 known-family combinations, and 43,767 combinations. These are screening cells, not
262,144 mathematically independent laws; inert parameters and numerically equivalent components
remain one empirical equivalence class.

## Quality and held-out result

All 240 exploration MAPS files were downloaded and hash recorded; none of the 80 confirmation
files were requested. Two hundred six files produced both raw ratios, and 204 pass the full
frozen quality rules. The 85% retention rate exceeds every total, fold, bar, mass, and age floor.
Most failures lack enough qualifying inner H-alpha measurements.

| Response and model | Held-out MSE | Held-out R2 | Relative change |
|---|---:|---:|---:|
| Stellar structural/age control | 0.00671501 | 0.0602 | — |
| Stellar control + selected mask geometry | 0.00688823 | 0.0360 | 2.58% worse |
| H-alpha structural control | 0.00546124 | 0.0847 | — |
| H-alpha + inherited stellar-selected geometry | 0.00546662 | 0.0838 | 0.10% worse |

The stellar paired mean MSE gain is negative and gives `p=0.827`. Both mass halves, both age
halves, and the high-bar half regress; the low-bar gain is only `0.0000067`. Four folds select a
bar-to-arm-radius family, one selects four-arm amplitude, and one fitted stellar coefficient
reverses sign. Thus neither predictive gain nor stable mechanism survives held-out testing.

The RTX 5090 evaluated `1,069,547,520` candidate-galaxy score combinations in 4.00 seconds with
CuPy 13.5.1. The maximum CPU/GPU component difference was `2.22e-15`.

## What the observed ratio means

The 204 quality galaxies have these descriptive distributions:

- stellar median outer/inner span: `1.3556`; only 14.7% are within 20% of unity;
- H-alpha median outer/inner span: `1.2307`; 40.2% are within 20% of unity.

A span is the range of line-of-sight velocities across an annulus. It is not the circular speed
of one star, and a larger outer span does not by itself show that every outer star travels at the
same speed as interior stars. Inclination, spatial sampling, noncircular motion, asymmetric drift,
and tracer distribution all enter. The pattern is retained as a real target for Item 15, not as
a formula discovery.

This test also does not decide whether age-dependent stellar mass-to-light estimates are wrong.
The prior age feature was included as a fixed control, and mask geometry regressed in both age
halves. That only says the tested morphology terms add no predictive value to this response; it
neither confirms nor refutes an age-dependent baryonic-mass correction.

## Counterexamples and boundaries

- Static masks cannot measure temporal pattern speed, corotation, or Lindblad resonances.
- GZ3D volunteer masks summarize visible projected arms and bars, not a full 3-D mass field.
- Stellar and gas annuli have different physical sampling even when the geometric cuts match.
- Morphology and kinematics are both from the SDSS MaNGA ecosystem.
- The response is not a gravitational-lensing map, cluster mass profile, or field equation.
- Zero confirmation responses were opened, zero formulas were generated after response access,
  and zero paid model calls were made.

## Next real test

Advance to Item 15, timescale ratios, on fresh identities. Before response access, freeze
dimensionless orbital, crossing, free-fall, star-formation, cooling, settling, and cosmic-time
ratios plus null controls. Test whether one universal timescale law predicts resolved
outer/inner motion and transfers to an independent tracer. Do not retune the 204 opened responses
or open the 80 Item 14 confirmations.

## Replay evidence

- result file SHA-256:
  `484f8c2e3d6ab72f570acac9d346525a75109bb287282dfda2ad766769709935`
- result content SHA-256:
  `e5061b71c8fa18d7788b09ea0d47dffbdd062bb7092dbe95458f8754f032cf98`
- response-source SHA-256:
  `d1088ff8213ed1b2c0231bd51fd364f5073d018ea58f062423d3aa10ddf05dd5`
- extraction-summary SHA-256:
  `55bbc396fe1172cd155233104a42876a2e4fcb692339faf61281ac5a53395126`
- sample-manifest SHA-256:
  `d4b529884a43a374832b5e2890a7891deb6632ba77b0cfcbe246da099f1894cd`
- synthesis file SHA-256:
  `e371d0b29fa848963005c573fb7172585ddf8317f12215a15069a0d9a3baf72b`
- synthesis content SHA-256:
  `55028a43aca6e3c40bdb468071dbdd9db9c0f6c2544a4003aad19851237003ca`
- replay commands:
  `python -m sigma_theory_compiler.gravity_item14_gz3d_resonance_coherence check`
  and `python -m sigma_theory_compiler.gravity_item14_synthesis --check`
