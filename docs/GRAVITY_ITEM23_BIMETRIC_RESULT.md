# Item 23: bimetric or multi-metric gravity result

## Decision

Item 23 is a **scoped reject on both formal tracks**. The frozen interacting-metric search does
not advance as a universal gravity theory and does not yet produce a standalone
phenomenon/publication lead on this small fresh sample.

This is not a judgment that bimetric gravity, age, or any other roadmap hypothesis was less
viable at the start. All four Item 23 mechanism niches received exactly 65,536 raw candidates and
the same target-blind data, held-out scoring, counterexample, locality, and provenance rules. The
result changes only the status of the precise representations tested here.

In plain language: the search repeatedly found one way for two interacting metrics to improve the
stellar-motion side of the problem. But the same formula made gravitational lensing worse. A much
simpler explanation—ordinary calibration differences between the motion and lensing
measurements—fit better, and an ordinary photometric regression fit much better. The system has
therefore found a stable way this theory family *tries* to solve the mismatch, plus evidence that
it is not the common cause on these lenses.

## Frozen action and matter coupling

The action-adjacent contract starts from the Hassan–Rosen two-metric structure,

`S = m_g^2 ∫√-g R[g] + m_f^2 ∫√-f R[f] - 2m^4 ∫√-g Σ beta_n e_n(√(g^-1 f)) + S_m[g_eff,psi]`,

with the composite matter metric

`g_eff = alpha^2 g + 2 alpha beta g√(g^-1 f) + beta^2 f`.

Around a proportional background, the two perturbations diagonalize into a massless and a massive
spin-2 mode. The matter/mass-eigenstate angle mismatch `delta` gives the nonnegative massive
residue `A=(4/3)tan(delta)^2`. The same matter coupling fixes both observables: the massive pole's
null-lensing projection is frozen to three quarters of its nonrelativistic-motion response.

The proportional-background linear regions are not new formula families. They reduce exactly to
the massless-plus-massive two-pole response already certified in Items 19 and 21. Only the two
nonlinear regions remain behaviorally distinct:

1. a monotone screened branch obtained from `u+c3 u^2+c4 u^3=(rV/r)^3`;
2. a baryon-triggered exchange in the metric mixing angle, using only the baryon-predicted
   acceleration before any response is opened.

Positive kinetic norms, positive Fierz–Pauli mass squared, an acyclic two-metric interaction,
unique monotone nonnegative branch solutions, positive motion/light responses, and a fractional
one-AU deviation below `1e-5` were imposed before fitting. This is still a static spherical
effective-field test, not an exact covariant galaxy solution or a proof of full nonlinear,
radiative, gradient, or cosmological stability. Primary theory references include the
[Hassan–Rosen action](https://doi.org/10.1007/JHEP02(2012)126), the
[effective composite matter metric](https://arxiv.org/abs/1408.1678), the
[composite-coupling constraint analysis](https://arxiv.org/abs/1409.1909), and the
[mass-eigenstate/GR-degeneracy result](https://arxiv.org/abs/1409.3146).

## Fresh-data boundary

The identity source is the public 2,000-object SuGOHI VI catalog. The target-blind audit found:

- 121 Grade-A systems;
- 32 with both spectroscopic lens and source redshifts;
- seven excluded because a prior preview or sample exposed the same coordinate-name core;
- five more excluded by a three-arcsecond coordinate match to Items 16, 17, or 22 under a
  different catalog name;
- 16 of the remaining systems with a matching SDSS DR16 galaxy spectrum;
- 15 with acceptable independent Legacy DR9 r/z light and geometry.

An HMAC rule sealed three confirmation identities and assigned 12 exploration identities to four
folds of three before either target was read. The target columns were SDSS DR16 stellar velocity
dispersion and the official SuGOHI master-list Einstein radius. Eleven exploration systems have
both valid values; one has no published Einstein radius and is excluded by the frozen rule. The
three confirmations remain unqueried.

The sample is deliberately small. It is sufficient to falsify a large effect in these exact
representations, but any positive pattern would require unchanged replication on fresh systems
before a paper claim.

## Search and controls

- 262,144 frozen raw programs, exactly 65,536 per mechanism niche;
- 168,233 programs survive the stability and Solar-System gates;
- 18,689 singly coupled and 18,472 composite-coupled linear controls survive, while all 65,536
  cells in each nonlinear niche survive;
- 499 full selection-aware null replays;
- 5,551,689,000 held-out training-residual evaluations including nulls;
- NVIDIA GeForce RTX 5090 through CuPy;
- 0.81 seconds to build the response matrix and 3.68 seconds for selection plus null replays;
- maximum CPU/GPU log-response disagreement `3.33e-16`;
- maximum admitted one-AU fractional deviation `3.00e-6`;
- the frozen branch-exchange injection is recovered in all four folds and beats the ordinary
  flexible control;
- the known-GR synthetic control does not prefer a nonzero bimetric response;
- zero paid model calls and zero API spend.

## Held-out result

| Predictor | Joint held-out MSE | Candidate improvement |
|---|---:|---:|
| Bimetric candidate | 0.256136 | — |
| Shared-calibration baryon-only GR | 0.304742 | **+15.95%** |
| Separate ordinary motion/lensing calibration | 0.218289 | **-17.34%** |
| Flexible ordinary photometric regression | 0.137658 | **-86.07%** |

The two observable channels expose the failure:

| Channel | Candidate vs shared GR | Candidate vs separate calibration | Candidate vs flexible model |
|---|---:|---:|---:|
| Stellar motion | **+26.41%** | +9.43% | **-165.28%** |
| Einstein-radius lensing | **-7.23%** | **-113.06%** | **-27.97%** |

All luminosity, size, and redshift halves improve over the rigid shared-GR calibration, but every
half loses to the flexible ordinary model. Seven of eleven lenses are individual counterexamples
relative to that strongest baseline. The raw selection-aware null-tail fraction is `0.97`; because
the observed candidate does not improve the strongest baseline, the guarded value is `1.0`.

## What was stable, and what it means

All four folds select the potentially new baryon-triggered metric-ratio-exchange niche. Three
folds select the exact same cell: massive range `316.23 kpc`, reference-mass Vainshtein scale
`0.3 kpc`, outer mixing angle `0.5 rad`, high-acceleration angle ratio `0.25`, transition
acceleration `3e-10 m/s^2`, transition power `4`, `c3=1`, and `c4=2`. The fourth fold selects the
same niche but a materially different branch direction and scale.

That family-level stability is real, but it is not a paper lead by itself. Its response acts mainly
like a structured way to give stellar motion and lensing different effective calibrations. Since a
plain separate-calibration model is better—especially by more than a factor of two in lensing
MSE—the data do not support interpreting the stable selection as a new gravitational cause.

The reusable result is a failure-space statement: healthy linear bimetric relabelings add no new
static formula beyond the prior two-pole class, and the tested nonlinear branch/exchange family
cannot jointly reconcile motion and lensing on this sample. The exact selected family is retained
for future independent comparison, not promoted or silently pruned.

## Two-track disposition and next action

- **Universal-gravity track:** do not advance this exact Item 23 representation.
- **Phenomenon/publication track:** no empirical lead yet, because neither channel nor any broad
  regime beats the strongest ordinary model. Retain the stable selection as a documented
  replication hypothesis only if an independent future dataset directly motivates it.
- **Confirmations:** keep all three sealed.
- **Numbered roadmap:** advance to Item 24, **different behavior of time**, on a fresh response.
  Freeze clock-rate, lapse, temporal-propagation, retardation, and history-dependent candidates
  that derive both motion and light from one rule.

Clerical erratum: the immutable machine receipt's `exact_next_action` sentence accidentally calls
Item 24 “emergent or entropic gravity.” The stable roadmap controls; emergent gravity is Item 38,
while Item 24 is different behavior of time. The frozen receipt is preserved rather than rewritten
after responses.

The machine-readable receipt is `runs/gravity/roadmap/item-23-bimetric-v1.json`.
