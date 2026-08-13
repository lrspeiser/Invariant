# Prospective blind tournament: success, rejection, and dependence

> This report is generated from validated receipts. A bounded PASS is not general
> discovery, stable replay is not truth, and corpus absence is not novelty.

## Receipt bindings

| Evidence | Path | File SHA-256 | Content SHA-256 |
| --- | --- | --- | --- |
| Tournament | `runs/math/prospective-blind-cross-generator-tournament/campaign.json` | `6f8b26b4d8bb9b088df6ef8f640f6fab775da1f9e0acc35dd9927eab10922339` | `2e4a49cc192cfbeba2b66ae616bd817b70034752581a9f2058826a552f1a315a` |
| Robustness and ablation | `runs/math/prospective-tournament-robustness-ablation/campaign.json` | `0660c4fa383e422888803da6751dedde1326155cb3bc959c0dd46e5909c55160` | `8f735fd9f196c25928d30165e7bf1f9d40f270178ed93546de7614ec2fa5f6d4` |
| Prior-art seed | `configs/equation_universe/gravity_seed_v1.json` | `e1fb33e7f79f43faf8a1c5ab4e8e7755b35d6abddb5199d90cc05fb90bd9a629` | import `184a77ab27dd95d34a26c592ab4cce20cfb485c7ac4c1151ca57faf7d7791a92` |
| Prior-art policy | `configs/equation_universe/source_policy.json` | `5ebb942fa6f2f10908883fe69c456c94926fc62a73747f3a70bcbf432e2983e9` | graph `3cc26065d80f543e59619b8945dfb6ed94640a35e9cde9eab807f24ac5dbd558` |
| Prior-art audit | `runs/equation-universe/audit-report.json` | `3bcb2ac87ffc5f841aa00a3e79762bc53d7df1fbb18a3df1d7747a49a7c028bc` | import validated |

## Chronology: what was hidden, and when

The campaign preregistered **3 worlds** and **7 native generator families per world**. All **21 generation events** completed before the single atomic unseal. The pre-unseal target-access count was **0**. That one batch disclosed **3 target records**; post-unseal generation and tuning both remained zero.

## A bounded PASS

In `prospective.modular_affine`, the sealed target was hypothesis **7**. The Bayesian proposal selected the same registered hypothesis and survived both hard gates. It was the only surviving family in this world. This proves only that one target-blind proposal hit one finite registered target under the fixed budget.

## The honest fixed-budget REJECT

In `prospective.finite_difference`, the target was hypothesis **1**. The seven target-blind proposals were `bayesian=10`, `cross_domain=5`, `egraph=0`, `evolutionary=5`, `grammar=3`, `llm=0`, `symbolic=9`. None matched, so the world terminated as `reject_fixed_budget_exhausted_without_holdout_match`, with no metric receipt and no Pareto front. The REJECT is evidence about this search budget, not a proof that the target is undiscoverable.

## Replay robustness

Across **8** preregistered order perturbations, all **8** CPU replay passes were exact: **168** candidate evaluations, **336** gate comparisons, and **28** Pareto recomputations. Candidate overlap was 21/21 for every seed; gate statuses and fronts were stable. This is replay invariance of a sealed candidate set, not evidence that new search seeds discover the same mathematics.

## Bayesian-only dependence

Removing Bayesian proposals changed **2** worlds from PASS to REJECT and removed **2** front members. Removing any of the other six families caused **0** decision changes and **0** front changes. The present successes are therefore Bayesian-dependent, not portfolio-robust.

## Prior art: absence is not novelty

The validated static equation-universe import contains 9 sources and 18 equations. All **21** tournament candidate hashes returned `absent_from_this_corpus`. This means only that those exact content identities are absent from that exact snapshot. It does not establish semantic novelty, historical priority, equivalence-class novelty, usefulness, or truth.

## Sharp boundary

The tournament provides two bounded PASS worlds, one honest REJECT world, exact replay, and a dependency diagnosis. It does not establish general discovery, scientific truth, novelty, or promotion eligibility. The next meaningful gate is an independently authored external world with a substantive proof oracle.

Report content SHA-256: `921cf82f3e44c61e3f257651c3f2390d48fa3dfc0e321542ae8e946f7e9a43a9`.
