# Gravity roadmap Item 9 synthesis: interior/exterior balance

## Decision

`REJECT_ITEM9_TESTED_STELLAR_LIGHT_OCCUPANCY_ADVANCE_ITEM10`

Item 9 is complete at the scope actually tested. The exact normalized stellar-light
interior/exterior occupancy formulas are not promoted as a universal law. Their positive
structure is preserved for future derivations; materially different gas-inclusive,
vector/tensor, action-derived, causal-history, and boundary mechanisms remain open.

## Why the family is not promoted

Attempt 1 was a genuine large-sample lead. On 823 quality-passing non-SPARC PROBES-I
galaxies, the frozen qualifying selector improved held-out MSE by `13.30%` over a flexible
local profile control, reached `R2=0.691`, and gave paired `p=0.002`. All five outer folds
selected the same acceleration-occupancy `I_in-I_out` operator. Promotion was blocked because
the prespecified 28-galaxy SHIVir subset regressed.

Attempt 2 performed the required zero-tuning transfer. It copied the five selected cells and
the exact earlier SPARC cell, excluded every predecessor identity, froze 233 PROBES-II
objects and filenames before response opening, and selected no formula or coefficient from
PROBES-II velocities. The formal attempt is inconclusive because 136 valid galaxies missed
the frozen 150-galaxy floor.

The valid-set diagnostic explains why further retuning is not justified:

- all five inherited cells improve over fixed stellar RAR;
- their median ensemble improves fixed stellar RAR by `6.86%`;
- the exact earlier SPARC cell improves fixed stellar RAR by `4.17%`;
- the median ensemble is `66.59%` worse than the OOF flexible local profile control;
- it loses in both halves of distance, stellar mass, surface density, and inclination;
- it regresses in three of four source families meeting the frozen ten-galaxy gate;
- paired `p=1.0` against the strongest baseline;
- six of fifteen gates pass.

The repeated RAR improvement shows that the inherited perturbation is not random numerical
noise. Its failure against local profile information shows that the exact term is not a
competitive universal explanation across these datasets.

## Failure-space record

The following region is now labelled `FAILED_UNIVERSAL_STELLAR_LIGHT_OCCUPANCY_REGION`:

- the acceleration-threshold-one, log-radius-scale-one `I_in-I_out` operator with the five
  inherited logistic amplitude maps;
- the exact earlier surface-brightness focusing cell;
- algebraic rescalings or renamed copies that do not add new physical information;
- attempts to rescue those cells with dataset-specific amplitudes, identity, or opened
  alternate responses.

Equivalent candidates should be rejected without retesting. Reconsideration requires a
materially new baryonic field, operator, field equation, response, or independent frozen
dataset—not a new fit to the opened responses.

## Positive structure retained

- One acceleration-occupancy operator converged in all five PROBES-I folds.
- The operator family improves the fixed stellar RAR in two nonidentical data pipelines.
- The likely useful question is therefore not “does this exact correction fit?” but “what
  local or boundary variable produces the reproducible correction, and why does its amplitude
  become dataset-dependent?”
- Gas, plasma, noncircular support, exact disk geometry, and a covariant gravitational field
  were not included. Their absence is a limitation, not permission to refit the same cells.

## Boundaries preserved

- All 323 PROBES-I confirmations remain sealed.
- All 229 unselected PROBES-II alternate rotation entries remain sealed.
- No paid model calls or post-response formulas were used.
- The synthesis does not reject every interior/exterior theory, general relativity, dark
  matter, or every alternative-gravity theory.
- It does not establish a new law, a causal gravity mechanism, or historical novelty.

## Next real test

Advance to Item 10, baryonic boundaries. Freeze genuinely new edge, shell, interface,
finite-domain, vector-focusing, and boundary-action terms before accessing a new response.
The Item 10 grammar must prove that its candidates are not algebraic rewrites of the failed
Item 9 occupancy cells. The real test must predict held-out resolved dynamics and retain
counterexamples; no PROBES-I confirmation or PROBES-II alternate curve is authorized.

## Replay evidence

- attempt-1 receipt SHA-256:
  `e86833a9a97244d01856f5005b81849d14f59974a12be3994e2c95c829512ef3`
- attempt-2 receipt SHA-256:
  `09e2896cf78bd9782835ebbc31d833261dd8293ff3ebef967937add41b37de4f`
- synthesis file SHA-256:
  `e343ea8835373b8802bb287cd3a7699973dec66d67c362cc6bf8931a5cd7f5fb`
- synthesis content SHA-256:
  `1d361a37383376bf52c953f45afc1255e4c0f26eebf652002ca1ffd46aef140b`
- replay command: `python -m sigma_theory_compiler.gravity_item9_synthesis --check`
