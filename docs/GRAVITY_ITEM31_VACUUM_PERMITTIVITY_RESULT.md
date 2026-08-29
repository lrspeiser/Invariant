# Item 31 vacuum polarization and gravitational permittivity result

Date: **2026-08-28**

## Decision

`SCOPED_ITEM31_REJECT`

Both formal tracks are **not promoted**. The quality gate passes, all frozen controls pass, and no
broad partial slice survives. This is therefore a clean negative result for the tested integrated
permittivity families, rather than a data-quality failure.

The immutable receipt is
`runs/gravity/roadmap/item-31-vacuum-permittivity-v1.json`. Forty confirmation objects remain
sealed. No paid API or model call was made.

## Frozen real-data experiment

Item 31 inherited only the response-blind, predecessor-vetoed MaNGA/GEMA predictor table frozen by
Item 30. Before any response access it excluded all 1,000 Item 30 roles, including Item 30's sealed
confirmations. Of the 511 remaining objects, a declared predictor-only `snr_med_g >= 5` rule retained
345. Median internal baryonic acceleration crossed with median five-Mpc GEMA tidal strength gave
four cells containing 82--91 objects each.

The committed role manifest selected exactly 50 objects per cell:

- 160 exploration objects, 40 per cell and 32 per outer fold;
- 40 sealed confirmations, ten per cell; and
- zero response values used for exclusions, roles, folds, formulas, or gates.

SkyServer returned all 160 declared exploration rows. A frozen response-quality rule retained 156
objects, or **97.5%**, above both the 140-object floor and the 87.5% retention floor. Two objects
failed the stellar-fit statistic, one failed the dispersion range, and one failed the velocity-span
range. No replacement identity was selected.

## Equal-capacity formula search

The grammar generated exactly **262,144** raw formulas, with **65,536** cells and matched enhancing
and suppressing polarity in each niche:

1. **Refracted density permittivity** -- a known-family control inspired by
   [Matsakos and Diaferio](https://arxiv.org/abs/1603.04943).
2. **Low-acceleration polarization** -- a known-family control overlapping MOND-like and dipolar
   medium phenomenology. The cited [Blanchet--Le Tiec construction](https://arxiv.org/abs/0807.1200)
   contains a dipolar dark medium, so it cannot establish literal vacuum polarization or a
   no-new-matter theory.
3. **Nonlocal constitutive polarization** -- a bounded one-object environment projection inspired
   by [nonlocal gravitational constitutive kernels](https://arxiv.org/abs/2209.05817), not a causal
   spacetime kernel.
4. **Frustrated dual-invariant vacuum response** -- a potentially new synthesis that activates
   when low-acceleration and low-density classifications disagree. It is behaviorally distinct from
   Item 30's conjunction by construction, but historical novelty is not claimed.

Every cell uses the same integrated projection

```text
div(epsilon grad(Phi)) = 4 pi G rho_b
mu_eff = epsilon_eff^-1 = 1 + s A chi
Delta log10(sigma) = 0.5 log10(mu_eff).
```

Finite-domain, positive-response, materiality, direction, and one-AU gates admit **228,926** cells:
63,466 density, 56,930 acceleration, 54,574 nonlocal, and 53,956 frustrated. Their 228,926 exact
parameter signatures collapse to **214,090** adversarial behavioral classes at the frozen precision;
14,836 duplicate behaviors are counted rather than presented as extra creativity. The maximum
admitted one-AU fractional response is `9.9970e-6`.

## Measured result

All five whole-object folds select an **enhancing low-acceleration polarization** control. Four
folds set environment coupling to exactly zero, and every fold selects the maximum allowed
amplitude `A=0.5`. Two parameter cells repeat across four folds; the remaining fold chooses the
same known niche with weak environment coupling. Neither the nonlocal nor the newly synthesized
frustrated branch is selected.

| Prediction | Held-out MSE | Item 31 relative result |
|---|---:|---:|
| Calibrated baryonic virial gravity | 0.0172318 | Candidate is **19.48% better** |
| Structural ridge control | 0.00825772 | Candidate is **68.02% worse** |
| Flexible structure + stellar population + environment | 0.00632529 | Candidate is **119.35% worse** |
| Selected permittivity candidate | 0.0138747 | — |

The guarded 99-trial full-search permutation result is **`p=0.94`**. Ninety-one of 156 individual
objects are counterexamples relative to the flexible ordinary model. Every low/high mass,
acceleration, density, and external-tide half improves over the simple virial estimate, but every
one is substantially worse than both stronger ordinary baselines. No partial slice clears the
predeclared 5% threshold against the flexible model.

## Controls and compute

- All four synthetic niche injections are recovered in all five folds. The nonlocal injection is
  required to use maximum environment coupling so it cannot collapse to the declared
  environment-zero acceleration rewrite.
- A zero-modification GR control produces no false improvement.
- The local, positivity, direction, and behavioral-equivalence checks pass.
- CPU FP64 and RTX 5090 results agree to a maximum absolute difference of `1.94e-16`.
- The GPU evaluated 35,712,456 candidate-observable matrix values and
  **14,284,982,400** observed-plus-null training residuals. Matrix construction took 0.121 seconds.
- Confirmation responses read: **0**. Post-response formulas generated: **0**. Paid calls: **0**.

## Interpretation

In plain language, a low-acceleration boost can repair part of an intentionally simple baryonic
gravity estimate, but ordinary information about galaxy size, structure, stellar populations, and
environment explains the measured dispersion much better. The search does not detect a vacuum
medium. It also gives no support to the newly synthesized invariant-mismatch response on this
observable.

The exclusion is deliberately scoped. It applies to these bounded one-effective-radius weak-field
projections using stellar-mass and GEMA observables. It does not reject all refracted gravity,
nonlocal gravity, vacuum polarization, resolved vector fields, action-derived constitutive laws, or
direct galaxy and cluster lensing predictions.

## Next action

Advance in order to **Item 32, boundary focusing**, on fresh data. Preserve all four Item 31
failure regions and their 214,090 behavioral classes for equivalence rejection. Item 32 must test a
derived vector or tensor field that redirects gravitational flux toward baryon-rich boundaries,
not another scalar radial boost. It must freeze source geometry, divergence/curl observables,
ordinary morphology controls, and motion/light consequences before response access.

Do not retune the 156 opened Item 31 responses or query the 40 confirmations. Continue the separate
Items 12/13, 20, 22, 25, and 29 replication tracks unchanged.
