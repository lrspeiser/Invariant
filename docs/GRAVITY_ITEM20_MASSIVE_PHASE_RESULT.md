# Item 20: baryon-driven massive-field phase result

## Decision

Item 20 is **rejected as a universal-gravity result** and does **not yet pass the frozen
phenomenon/publication gates**. It nevertheless produces a promising, tightly scoped exploratory
signal worth testing unchanged on an independent dataset. That signal is retained rather than
pruned.

In plain language: a rule that turns on an extra attractive force when the baryon-predicted
orbital clock approaches a universal carrier clock predicts these galaxies much better than the
frozen physical baselines. It is not better than a flexible ordinary statistical regression, and
one of five frequency bands reverses, so the current data do not distinguish a new physical phase
from a compact way of representing ordinary galaxy correlations.

## Frozen question and data boundary

The experiment froze three mechanisms and both force polarities before reading any SFI++
linewidth:

1. a standard damped linear resonance (`KNOWN_FORMULA_REAPPLICATION`);
2. a compact Landau-style window (`KNOWN_FAMILY_COMBINATION`);
3. two-harmonic coherence (`POTENTIALLY_NEW_SYNTHESIS_OF_KNOWN_COMPONENTS`).

All are in the broad Item 15 timescale-resonance equivalence class. Their new test context is an
action-linked massive carrier with Compton frequency `c/lambda`, convolved with the measured
baryonic disk. No historical novelty is claimed.

The trigger is computed only from baryonic predictors:

`Omega_b = sqrt(v_bar^2) / R`, `z = n Omega_b lambda / c`, and
`v_pred^2 = v_bar^2 + s A P(z) v_Y^2`.

The observed linewidth never enters the phase trigger or candidate generator. The exact
three-catalog join produced 407 eligible galaxies. Four coordinate matches to earlier roadmap
samples were removed, leaving 403 fresh identities. A target-blind HMAC split locked 300
exploration galaxies and 100 sealed confirmations across five baryonic-frequency strata. All 300
exploration responses passed the frozen quality rules; zero confirmation responses were opened.

## Search and controls

- 262,144 raw PCG64 programs, distributed equally across six family/polarity niches;
- 262,143 passed the pre-response Solar-System phase bound;
- 15,728,580,000 residual evaluations including 199 within-frequency-stratum permutations;
- NVIDIA GeForce RTX 5090 through CuPy; 11.89 seconds for the primary replay;
- bounded-phase, exact phase-off GR, known-GR, and digest-selected synthetic phase-injection
  controls all pass;
- the injected cell is recovered exactly in all five folds;
- the maximum admitted Solar-System phase-force fraction is `7.13e-6`.

## Held-out result

| Predictor | Held-out log-velocity MSE | Candidate improvement |
|---|---:|---:|
| Massive-phase candidate | 0.028476 | — |
| Fixed baryonic GR | 0.201349 | 85.86% |
| Globally calibrated baryonic GR | 0.046583 | 38.87% |
| Baryonic Tully–Fisher | 0.037337 | 23.73% |
| Flexible seven-variable nuisance regression | 0.028330 | **-0.52%** |

The full selection-aware permutation result is `p=0.015`. Every fold chooses an enhancing force;
four choose two-harmonic coherence and one chooses standard linear resonance. Four carrier ranges
cluster within 0.75 dex. The selected cells improve both baryonic-mass halves, both size halves,
and four of five frequency strata relative to calibrated GR. The highest-frequency stratum is
3.47% worse. Changing the gas-disk scale does not change the primary result.

Against the strongest ordinary regression, the candidate improves the high-mass half by 1.64%
but is 3.18% worse in the low-mass half. It improves frequency strata 1 and 2, while strata 0, 3,
and 4 regress. These failures are why the phenomenon/publication gate does not pass despite the
strong physical-baseline result.

## What is retained

The retained object is not “a new theory of gravity.” It is the frozen empirical proposition that
a baryon-derived orbital-frequency/carrier-frequency interaction—especially the two-harmonic
form—compresses global galaxy linewidth structure substantially better than calibrated baryonic
GR and BTFR on this fresh sample.

That proposition is interesting enough for a separate paper-track experiment because its folds,
polarity, carrier range, and permutation result are coherent. The next test must be preregistered,
use a genuinely independent source and preferably a resolved rotation observable, keep the exact
selected formula fixed, and compare against a flexible ordinary model. The sealed 100 SFI++
objects must not be opened merely to tune the current result. Cluster dynamics and lensing remain
later universal-gravity gates; this global-linewidth experiment supplies neither.

The immutable machine-readable receipt is
`runs/gravity/roadmap/item-20-massive-phase-v1.json`.
