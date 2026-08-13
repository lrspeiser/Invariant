# Semantic discovery contrast: blind guessing and exact constrained recovery

> Both campaigns are prospectively sealed and receipt-validated. Their hidden worlds
> are distinct, so this is a methodological contrast—not a matched-target scorecard.

## Receipt bindings

| Receipt | Path | File SHA-256 | Content SHA-256 |
| --- | --- | --- | --- |
| Blind structured guessing | `runs/math/semantic-formula-proof-holdout-tournament/campaign.json` | `401780e27d8006800ffceeb323a7938a6fa3305816a4a93041c28255171915d0` | `42149516825bd8fc4a78d78d5ba99b038f2db440cd036cfbe63691364b66d798` |
| Constraint-conditioned recovery | `runs/math/constraint-conditioned-semantic-recovery-tournament/campaign.json` | `cb5b0963c52887c93008a23d87925dea8b9fd394063ff5d724b56c0a587ef54e` | `300eee2de03b328da94cad8337aba11f3e31dde17ffb97e016d6cc922a43f37d` |

Both JSON receipts were reopened and replayed through their native campaign validators before this projection was built.

## The exact contrast

| Campaign | Worlds | Candidates | PASS | REJECT | BLOCK | Evidence after unseal | Pareto eligible |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Blind structured guessing | 3 | 21 | 0 | 21 | 0 | 21 exact counterexamples | 0 |
| Constraint-conditioned recovery | 3 | 21 | 21 | 0 | 0 | 21 exact certificates | 21 |

The blind campaign produced structured, target-free formulas, but every one of its 21 candidates failed exact equality after unseal. The recovery campaign gave the same seven native families public semantic constraints and then applied one generic exact operator; all 21 lineage-bound candidates passed.

## Chronology

Each campaign completed **21 generation events** across three worlds and seven native families before one atomic unseal. Pre-unseal target access was **0**; the batch disclosed **3 targets**. Post-unseal generation and tuning were both zero.

The recovery constraints were committed publicly before generation. The hidden target records remained separately committed until the atomic unseal.

## What public evidence was added

| Recovery world | Declared basis terms | Public evidence | Evidence count | Constraint SHA-256 | Closed form public? |
| --- | ---: | --- | ---: | --- | --- |
| `constraint.hidden_quartic` | 5 | `exact_input_output_examples` | 5 | `0671b1163577eb24816ba0f028b3385a611c35eae051d11cda3083dd541ca0bb` | no |
| `constraint.hidden_partial_fraction` | 3 | `exact_input_output_examples` | 3 | `fee3826f0d5c80abf841ef7ebdb880812ac3906bd8cf9f708518c2323de26df5` | no |
| `constraint.hidden_recurrence` | 4 | `exact_base_and_successor_recurrence_axioms` | 2 | `7f0fe4de903c11d14a01e2dfe0876ce9fcbe5a67d746438d4f8c0d5b4e83eb28` | no |

Public target-derived exact examples or recurrence axioms constrain declared bases without publishing a closed-form target or its coefficient vector.

## The generic recovery operator

The declared grammar is `linear_combination_of_declared_basis` over exact `rationals` coefficients. The operator converts evaluation examples or recurrence coefficient identities into one rational linear system. It classifies rank exactly: a unique solution emits a candidate, an underdetermined system BLOCKs, an inconsistent system REJECTs, and malformed input BLOCKs.

The operator ran **21 times**. Every world has seven native lineage candidates but exactly one recovered expression across those lineages. Native proposals determine provenance and row order; they do not supply the recovered coefficients. The semantic recovery therefore belongs to the generic solver plus the public constraints, not to seven independent discoveries.

## Exact certificates after unseal

The blind receipt contains three validated reference certificates—{'invariant-math-induction-certificate-1.0': 1, 'invariant-math-rational-identity-certificate-1.0': 2}—and 21 exact candidate counterexamples. The recovery receipt contains 21 checked candidate certificates—{'invariant-math-induction-certificate-1.0': 7, 'invariant-math-rational-identity-certificate-1.0': 14}—plus three reference certificates—{'invariant-math-induction-certificate-1.0': 1, 'invariant-math-rational-identity-certificate-1.0': 2}. Polynomial and rational worlds use exact rational-identity certificates; the recurrence world checks its base case and symbolic successor identity.

Certificates check equality or induction after unseal inside the registered grammar; they do not prove general discovery or scientific significance.

## Fail-closed negative controls

| Control | Expected | Observed | Exact reason | rank(A) | rank([A|b]) |
| --- | --- | --- | --- | ---: | ---: |
| `malformed_unknown_symbol` | BLOCK | BLOCK | `malformed_constraints:ValueError` | 0 | 0 |
| `underdetermined_rank_deficient` | BLOCK | BLOCK | `underdetermined_exact_constraints` | 1 | 1 |
| `noisy_inconsistent_duplicate` | REJECT | REJECT | `inconsistent_exact_constraints` | 2 | 3 |

## What the 21/21 result does—and does not—mean

The success shows that exact, sufficient semantic conditions can turn a registered linear grammar into a determinate synthesis problem. It does not show that unconstrained native generators independently found the formulas. It does not cover out-of-basis targets, nonlinear coefficient searches, noisy-data inference, external proof kernels, scientific significance, novelty, or promotion.

Because the blind and recovery cohorts use different hidden targets, subtracting 0/21 from 21/21 is not a matched-world effect estimate. The next sharp gate is a matched, independently authored, out-of-basis campaign with an external proof kernel and the same one-unseal chronology.

Report content SHA-256: `30f8c91f7db8fda718abfb43f22e45db0667334a19e23b02fd27d5679baec6d1`.
