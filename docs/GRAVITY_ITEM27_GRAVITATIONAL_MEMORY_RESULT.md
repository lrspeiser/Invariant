# Gravity roadmap Item 27: gravitational memory

## Decision

**INCONCLUSIVE QUALITY; NEGATIVE PRIMARY DIAGNOSTIC; ITEM COMPLETE.**

The frozen CALIFA test required at least 45 exploration galaxies with usable stellar
kinematics in all three disk-scaled annuli. Only 9 of 64 met that rule. The gate was not
lowered. Twenty-six galaxies retained the unchanged primary annulus, so the full frozen
search ran only as a non-promotable diagnostic. On those 26 galaxies the selected memory
formulas were **28.68% worse** than the simple instantaneous stellar-baryonic baseline,
the full-search permutation result was `p=0.75`, both stellar-mass halves regressed, and
the outer replay remained unavailable. Item 27 therefore passes neither the universal-
gravity track nor the phenomenon/publication track.

This result does not reject age, history, or every gravitational-memory theory. It rejects
promotion of the four tested fading-kernel representations on this CALIFA observable and
sample. The separate Item 12/13 age/surface-density association remains an active
same-survey phenomenon lead awaiting unchanged cross-source replication.

## Equal-viability and two-track policy

Age/history received no preferred status. The numbered roadmap position controlled only
execution and leakage. Four memory mechanisms each received exactly 65,536 raw cells,
the same target-blind rules, and the same held-out evaluation:

1. a single exponential kernel;
2. a stretched exponential kernel;
3. a positive two-timescale exponential mixture;
4. an integrable scale-free fading tail.

Every result received two independent judgments. A formula did not need to solve gravity
to become scientifically interesting, but a phenomenon lead still had to beat a strong
ordinary baseline, survive the full-search null, behave consistently across broad slices,
and have enough fresh data. Item 27 did not meet those conditions.

## Frozen physical construction

For lookback ages `tau_j` and Pipe3D fossil luminosity fractions `f_j`, each candidate
computed

`Q_K = sum_j f_j K(tau_j) / sum_j f_j`,

then modified the instantaneous stellar-baryonic prediction by

`mu = exp[s A Q_K / (1 + (g_bar/a_transition)^p)]`

and

`log10(v) = log10(v_baryon) + 0.5 log10(mu)`.

The fossil state was independently allowed to be global, inner, outer, or outer-minus-
inner. Every admitted `K` had only past support, `K(0)=1`, nonnegative values, monotonic
fading, and a finite nonnegative integral. The low-acceleration window was applied
unchanged at Solar-System acceleration, and the one-AU fractional response had to remain
below `1e-5`.

These kernels are known response families, not new formulas. Using them with resolved
fossil histories and a low-acceleration gravitational response is at most a potentially
distinct synthesis until prior art, an action or state realization, and independent data
say more. Causal/nonlocal gravity itself is established prior art in
[Mashhoon's nonlocal-gravity framework](https://arxiv.org/abs/1101.3752) and in
[Rahvar and Mashhoon's galaxy/cluster tests](https://arxiv.org/abs/1401.4819).

## Fresh data and leakage boundary

The predictors came from the public
[Pipe3D CALIFA DR2 products](https://arxiv.org/abs/1602.01830): stellar mass, SFR,
population age and metallicity, SDSS photometric geometry, and 39 SSP-age luminosity-
fraction planes. The response came from the separately published
[CALIFA V1200 stellar-kinematics maps](https://arxiv.org/abs/1609.06446).

The predictor audit intersected 119 identities advertised by both products, excluded 12
predecessor-name overlaps and four additional coordinate overlaps, and froze 79 eligible
galaxies. HMAC ranking sealed 15 confirmations and assigned 64 exploration identities to
five folds. No V1200 value was read until the science, formulas, identities, roles, folds,
thresholds, and candidate injections were committed. The 15 confirmation files remain
unqueried.

The stellar-population record is a luminosity-fraction fossil reconstruction, not a direct
mass-weighted history of all baryons. Stellar `v_rms` is not the circular speed of every
star and is not a lensing observable. Those limits were frozen before response access.

## Candidate-space and compute receipt

- Raw cells: `262,144`, exactly `65,536` per kernel niche.
- Admissible cells: `192,661` after equivalence, normalization, monotonicity,
  integrability, positivity, domain, and local-response gates.
- Admissible counts: exponential `57,360`; stretched `57,301`; positive mixture
  `20,538`; scale-free tail `57,462`.
- All four frozen synthetic injections recovered their correct niche in all five folds.
- GPU: NVIDIA GeForce RTX 5090 through CuPy; CPU/GPU maximum difference `0`.
- Training residual evaluations: `2,103,858,120`.
- Full-search null trials: `99`.
- Wall time: `94.02` seconds for the final replay.
- Paid model calls: `0`; paid API spend: `$0`.

## Quality outcome

- Frozen exploration identities: `64`.
- Required all-annulus valid galaxies: `45`.
- Primary-annulus valid: `26`.
- All three annuli valid: `9`.
- Formal quality gate: **failed**.
- Confirmation values read: `0`.

The official README labels the kinematic columns `Vp/Sp/DVp/DSp`, while the binary FITS
tables expose uppercase names. The first exploration-only acquisition downloaded the 64
permitted files and aborted on that case mismatch without writing a response artifact.
The correction accepted column names case-insensitively. After the all-annulus shortage
was observed, a second disclosed correction retained the 45-object gate, admitted only a
primary-only diagnostic above the already-frozen 20-object diagnostic floor, and disabled
promotion. Neither correction changed a formula, identity, response value, threshold,
confirmation boundary, or decision gate.

## Diagnostic result

At the primary `1.3–2.3` disk-scale annulus:

- candidate MSE: `0.0144337`;
- instantaneous stellar-baryonic MSE: `0.0112164`;
- candidate versus instantaneous: **28.68% worse**;
- flexible ordinary-model MSE: `0.0212394`;
- candidate versus flexible: `32.04% better`;
- guarded selection-aware permutation: `p=0.75`;
- individual counterexamples versus flexible: `11/26`.

The apparent win over the flexible model is not a positive result: on this small sample
that high-dimensional model was itself much worse than the simple baryonic baseline. The
candidate lost to the stronger comparator, and 75% of guarded null searches did at least
as well as the observed gain against the simple baseline.

Four folds chose the stretched-exponential niche and one chose the two-timescale mixture,
but exact timescales, spatial states, and acceleration transitions varied. The selected
amplitudes all hit `A=2`, which is another warning that the search favored a boundary
rather than identifying a stable formula.

Broad diagnostic slices reinforce the negative decision:

- low-mass galaxies: `29.996%` worse than instantaneous;
- high-mass galaxies: `26.941%` worse;
- younger fossil half: `56.615%` worse;
- older fossil half: `13.151%` better;
- inner-annulus replay: `0.614%` worse;
- outer-annulus replay: only 9 objects and four folds, so formally unavailable.

The older-half improvement is retained as scoped failure knowledge, not a lead: it was
not paired with a full-sample gain, selection-aware significance, formula stability, or
adequate radial data.

## Lay interpretation

The test asked whether a galaxy's present stellar motion looks as if gravity still carries
a fading echo of when and where its stars formed. On the small usable sample, the answer
was “not in a dependable way.” A memory formula could outperform an unnecessarily complex
ordinary regression, but it could not outperform the simpler rule based on present stellar
mass and disk size. Randomly reassigned histories often produced equally convincing
formulas. That is exactly the pattern expected from flexible searching plus too little
data, not from a discovered physical memory law.

## Exact next action

Preserve the complete Item 27 failure region and the older-half diagnostic without
retuning or opening the 15 confirmations. Keep the Item 12/13 age association on its
separate publication track. Advance the equal-priority numbered roadmap to Item 28,
periodic gravity, using a fresh response and equal raw capacity for spatial, temporal,
log-scale, and phase-oscillation mechanisms.

Machine-readable result:
`runs/gravity/roadmap/item-27-gravitational-memory-v1.json`.
