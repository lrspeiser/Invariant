# Item 21: mixed massless and massive modes result

## Decision

Item 21 is a **scoped reject on both empirical tracks**. The nonlinear mixed-mode search is not a
universal-gravity result and does not produce a standalone phenomenon/publication lead on this
dataset. Its strongest reusable result is a linear-equivalence certificate: every stable
quadratic mixture with one massless and one massive pole reduces to the same two-pole static
response already tested in Item 19.

In plain language: combining a permanently long-range gravitational field with a finite-range
field can sound like a new mechanism. At linear order it is not a new curve family—changing field
coordinates turns it into an ordinary sum of one massless and one Yukawa response. The genuinely
different nonlinear ideas tested here chose one very stable avoided-crossing formula, but that
formula predicted new galaxies worse than baryonic Tully–Fisher and much worse than an ordinary
multivariable regression.

## Linear failure-space certificate

For any healthy quadratic two-field scalar sector, the Fourier-space kinetic matrix can be put in
the form

`K(k) = [[a k^2, epsilon k^2], [epsilon k^2, b(k^2+m^2)]]`.

Contracting `K^-1` with any universal linear matter-coupling vector produces a rational source
response with denominator `k^2 (C k^2 + D)`. Partial fractions therefore give exactly

`R0/k^2 + Rm/(k^2+m_eff^2)`.

Positive kinetic determinant, positive mass squared, and positive residues give the healthy
massless-plus-massive pole family. Its static radial response is a Newton kernel plus a Yukawa
kernel, not a new interference law. Negative residues introduce a ghostlike or repulsive branch;
complex or negative mass squared introduces unstable or oscillatory/tachyonic behavior. Thus the
healthy linear class is behaviorally equivalent to Item 19 and is stored as such rather than
counted as historical novelty.

The nonlinear search remained eligible because environment-dependent diagonalization or mixing
can no longer be removed by one global linear field redefinition.

## Frozen nonlinear mechanisms

All mechanisms and both polarities began with equal raw candidate counts:

1. healthy normalized two-pole response — `KNOWN_FORMULA_EQUIVALENT_TO_ITEM19`;
2. avoided level crossing — `KNOWN_FAMILY_COMBINATION`;
3. adiabatic mode exchange — `KNOWN_FORMULA_REAPPLICATION_FROM_TWO_STATE_MIXING`;
4. Fano mixed-mode interference — `POTENTIALLY_NEW_SYNTHESIS_OF_KNOWN_COMPONENTS`.

The nonlinear trigger used only the baryon-predicted acceleration and the convolved ratio of the
massive to massless disk response. Neither the observed line width nor an object identifier could
enter a candidate. Positive kinetic determinant, positive carrier mass squared, bounded mixing
probability, positive predicted velocity squared, and a four-radius Solar-System deviation below
`1e-5` were imposed before responses.

## Fresh real-data boundary

The predictor-only source was the 1,715-object VizieR
`J/ApJ/809/146/table1` isolated low-mass galaxy catalog. Frozen uncertainty and inclination cuts
left 1,041 objects. Thirteen coordinates matching earlier roadmap samples and three identities
whose target values appeared during source discovery were removed. A target-blind HMAC rule chose
800 of 1,025 fresh identities, with 600 exploration and 200 sealed confirmation galaxies balanced
across five baryonic-acceleration strata.

All 600 exploration responses were requested individually; 587 pass the frozen response-quality
rules. No confirmation response was requested or read.

## Search and controls

- 262,144 raw programs across eight equal family/polarity niches;
- 172,784 locally and formally admissible programs;
- eight structural equivalence classes;
- 20,284,841,600 residual evaluations including 199 nested null replays;
- NVIDIA GeForce RTX 5090 through CuPy; 14.23 seconds for the final replay;
- maximum CPU/GPU log-velocity disagreement `4.44e-16`;
- the digest-selected synthetic mixed-mode injection is recovered exactly in all five folds;
- the known-GR control fails: the flexible mixed-mode search improves its nearly-GR synthetic
  control by 2.63%, demonstrating a false-positive capacity that counts against promotion;
- the maximum admitted Solar-System deviation is `9.997e-6`.

## Held-out result

| Predictor | Held-out log-velocity MSE | Candidate improvement |
|---|---:|---:|
| Mixed-mode candidate | 0.074620 | — |
| Fixed baryonic GR | 0.117961 | 36.74% |
| Globally calibrated baryonic GR | 0.089864 | 16.96% |
| Baryonic Tully–Fisher | 0.072568 | **-2.83%** |
| Frozen RAR-at-characteristic-radius proxy | 0.570074 | 86.91% |
| Flexible ordinary nuisance regression | 0.061728 | **-20.88%** |

Every fold independently selects the same enhancing avoided-level-crossing cell: carrier range
`183.90 kpc`, transition acceleration `4.79e-13 m/s^2`, field power `0.5`, massive-ratio power
`2`, effective amplitude `2.10`, stellar scale `0.5`, and gas scale `1.0`. That stability is real,
but stable selection does not make the formula predictive enough.

Relative to the strongest ordinary regression, the candidate loses in both mass halves and all
five acceleration strata. It remains about 20.9% worse under both alternative gas geometries.
There are 319 individual counterexample galaxies. Because the candidate does not improve the
strongest baseline, the one-sided selection-aware permutation result is `p=1.0`; the raw
unguarded null-tail diagnostic (`0.015`) is retained only to document why a non-improvement guard
is necessary, not as evidence for the candidate.

## Meaning and next action

This result closes the tested healthy linear two-pole class by equivalence to Item 19 and rejects
the exact global-linewidth representations of avoided crossing, adiabatic exchange, and Fano
interference. It does not reject nonlinear mixed fields generally, resolved radial effects,
multiple polarizations, or a covariant action whose nonlinear solution differs materially from
these bounded proxies.

No Item 21 paper-track claim is warranted from the empirical fit. The equality/stability proof for
the linear class is independently useful as a failure-space result because it prevents the engine
from repeatedly renaming kinetic mixing as a novel force law.

The machine-readable receipt is `runs/gravity/roadmap/item-21-mixed-modes-v1.json`.
