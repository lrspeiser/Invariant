# Gravity roadmap Item 22: multiple gravitational polarizations

## Bottom line

This frozen real-data test does **not** support the tested scalar/vector/tensor mixture as a
universal replacement for GR, and it does **not** yet produce a phenomenon/publication lead.
The best candidate improves on GR with one shared baryonic calibration by `8.45%`, but it is
`3.20%` worse than ordinary GR with separate dynamics/lensing calibration and `16.53%` worse
than a fixed flexible photometric model. The guarded selection-aware permutation result is
`p=1.0`.

The attempt nevertheless preserves one specific, preregistered partial pattern rather than
discarding it: four of five outer folds select the new three-polarization interaction niche,
and the candidate beats the flexible model by `21.03%` in the high-luminosity half and
`17.08%` in the large-size half. It fails badly in the complementary halves, so this is a
regime-dependent replication hypothesis, not a discovery or paper claim.

## What was tested

For an extra gravitational mode with Newtonian potential contribution `Psi_h` and spatial
potential contribution `Phi_h`, define

`eta_h = Phi_h/Psi_h`, and p_h = (1 + eta_h)/2`.

The frozen weak-field projectors were:

| Polarization | `eta_h` | response of light relative to matter `p_h` |
|---|---:|---:|
| conformal scalar | `-1` | `0` |
| longitudinal vector | `0` | `1/2` |
| transverse tensor | `1` | `1` |

Each mode has the same local-off massive-force basis

`B_h(r) = 1 - (1 + r/lambda_h) exp(-r/lambda_h)`.

For amplitude `A`, polarity `s`, normalized helicity weights `w_h`, and the optional frozen
interaction `I(r)`, the matter and light responses are

`log mu_D(r) = s A [sum_h w_h B_h(r) + I(r)]`,

`log mu_L(r) = s A [sum_h p_h w_h B_h(r) + I_L(r)]`.

The exponential completion keeps both responses positive and reduces to an additive weak-field
pole sum for small coupling. It is an ansatz, not a proved covariant action.

The search assigned exactly `65,536` raw candidates to each of four niches:

1. one-helicity controls already behaviorally covered by Item 16;
2. two-helicity/two-range controls already behaviorally covered by Item 16;
3. three helicities with three strictly distinct ranges;
4. three helicities plus a frozen pair-interaction vertex.

Every niche contained exactly `32,768` enhancing and `32,768` suppressing candidates. The
first two niches were retained as rewrite controls. Only the last two were allowed to count as
empirically new relative to Item 16. No historical novelty is claimed without a separate
literature review.

## Data and leakage boundary

The response source was the [Amante et al. 204-lens compilation](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/498/6013),
whose early-type lenses have spectroscopic velocity dispersions, Einstein radii, and lens/source
redshifts. Independent baryonic predictors came from the [Legacy Survey DR9 Tractor catalog](https://www.legacysurvey.org/dr9/catalogs/):
extinction-corrected `g/r/z` model fluxes, half-light radius, morphology, ellipticity, and quality
diagnostics.

Before response access, the pipeline:

- excluded `95` Item 16/17 identities by normalized name, including matches that a coordinate-only
  audit missed;
- excluded `26` identities whose response values had appeared during source previews;
- excluded six remaining systems without usable source coordinates;
- made `77` predictor-only Legacy queries and retained `50` systems after fixed morphology,
  centering, shape, signal-to-noise, masking, and blending checks;
- froze `40` exploration roles and ten unqueried confirmations, with eight exploration systems
  in each outer fold and two confirmations in each predictor-only redshift stratum;
- froze all `262,144` candidates before any selected velocity dispersion or Einstein radius was
  read.

Only the 40 exploration identities received exact per-name response queries. Confirmation
response values read: `0`. Post-response formulas generated: `0`. Paid model calls and spend:
`0` and `$0`.

The two predicted channels were evaluated through the same stellar mass-to-light factor:

`log[5 R_e sigma^2/(G L_z)] = log(M/L_z) + log mu_D(R_e)`,

`log[pi R_E^2 Sigma_crit/(L_z f_H(R_E))] = log(M/L_z) + log mu_L(R_E)`.

The Einstein radius is therefore a response/evaluation radius, not an input to formula generation.
This is stricter than fitting a published total lensing mass but is not yet direct image-level
lensing.

## Frozen results

All 40 exploration systems passed response quality.

| Comparison | MSE | candidate improvement |
|---|---:|---:|
| selected polarization candidate | `0.288031` | — |
| shared-calibration baryonic GR | `0.314631` | `+8.45%` |
| separate dynamics/lensing GR calibration | `0.279093` | `-3.20%` |
| flexible photometric nuisance model | `0.247179` | `-16.53%` |

By response channel:

| Channel | vs shared GR | vs separate calibration | vs flexible model |
|---|---:|---:|---:|
| stellar dynamics | `+3.66%` | `-4.90%` | `-15.87%` |
| Einstein-radius lensing | `+17.50%` | `+0.36%` | `-18.00%` |

The error-weighted candidate MSE is `0.246769`, compared with `0.224532` for the flexible
model, so measurement weighting does not reverse the conclusion.

The best outer-fold niches were four three-helicity interaction candidates and one known
single-tensor control. The interaction candidates used suppressing polarity, mostly destructive
scalar/tensor or vector/tensor phases, and sub-kpc transition ranges. Their exact parameters were
not stable, and the fitted global `M/L_z` values ranged from `2.64` to `9.31`, indicating that
photometric calibration remains a serious competing explanation.

There are `20/40` individual counterexamples where the selected candidate loses to the flexible
ordinary model.

## Two-track decision

### Universal-gravity track: do not advance

The candidate misses the frozen `10%` shared-GR improvement threshold, loses to both stronger
ordinary baselines, fails the low-redshift half, and has guarded permutation `p=1.0`. A stable
mechanism label cannot compensate for failed held-out predictions.

### Phenomenon/publication track: no formal lead yet; preserve a replication hypothesis

The overall candidate and both complete response channels lose to the flexible model, so the
frozen publication gates fail. The high-luminosity and large-size halves were declared before
responses and improve over the flexible model, while the low-luminosity and small-size halves
regress by `45.45%` and `62.11%`. This makes the exact interaction formula and high-mass/large-size
domain scientifically testable on fresh data, but multiple related slice checks, only 20 systems
per half, and failure in the full sample prevent a paper-level claim now.

The correct next paper-track action is an **unchanged** replication of the selected interaction
family on a wholly independent high-luminosity/large-size strong-lens sample with better stellar
population masses and resolved dynamics. The ten Item 22 confirmations remain sealed because the
formal lead gate did not pass.

## What this closes—and what it does not

This result rejects the exact spherical, photometric, exponentiated subtracted-Yukawa search region
as a promoted universal law on these fresh lenses. It also supplies counterexamples against the
idea that merely adding more polarization degrees of freedom will automatically reconcile motion
and light bending.

It does not reject:

- polarization laws derived from a complete covariant action;
- non-spherical or resolved Jeans dynamics;
- direct image-plane lensing likelihoods;
- different scalar/vector/tensor projectors or matter couplings;
- nonlinear screening, environmental dependence, or radiative polarization effects;
- the preserved high-luminosity/large-size interaction hypothesis before independent replication.

The numbered roadmap therefore advances to Item 23, bimetric or multi-metric gravity, while the
partial Item 22 pattern remains available on its separate phenomenon/publication track.

## Reproduction

- Config: `configs/gravity_item22_polarization_superposition_v1.json`
- Runner: `src/sigma_theory_compiler/gravity_item22_polarization_superposition.py`
- Immutable result: `runs/gravity/roadmap/item-22-polarization-superposition-v1.json`
- Source and freeze receipts: `runs/gravity/roadmap/item-22-polarization-superposition-v1-source/`
- Validation: `python -m sigma_theory_compiler.gravity_item22_polarization_superposition validate`

The RTX 5090 evaluated `8,388,608,000` frozen training residuals including all null searches in
about `2.12` seconds of measured screening time. CPU/GPU maximum absolute disagreement was
`8.88e-16`; the one-AU deviation ceiling was passed at `3.80e-14`; the injected interaction was
recovered in all folds; and the pure-GR control did not falsely prefer a nonzero polarization.
