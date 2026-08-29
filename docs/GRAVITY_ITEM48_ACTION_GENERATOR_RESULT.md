# Gravity Item 48 action-generator result

Date: **2026-08-29**

## Decision

`NONPROMOTED_ITEM48_ACTION_RESULT_RETAINED`

Item 48 materially changes how Invariant generates gravity candidates. Every candidate now begins
as a frozen normalized weak-field action, its radial field equations are obtained by automatic
Euler variation, malformed or nonpositive cells are rejected, and only an action-derived response
is scored on the data. All five outer folds selected the exact same long-range two-field action.

The empirical result is nevertheless negative relative to the strongest existing candidate. The
action has balanced loss **1.8116**, versus **0.7615** for Item 45, so it is **137.91% worse**.
The two-sided object sign-flip value is `p=0.00115`; here the small value measures a stable
*disadvantage*, not evidence for the action. It is worse than Item 45 on both S4TM galaxy lenses and
CLASH clusters and under all four global baryonic-mass variants.

The result is not vacuous. The selected action is **3.53% better** than the cross-fitted ordinary
coordinate ridge overall and all five folds independently choose the same action, source, range,
mixing, amplitude, exponent, and transition. It is therefore preserved as a reproducible formal
and empirical clue. It is not promoted, called novel, or erased.

The immutable aggregate receipt is
`runs/gravity/roadmap/item-48-action-generator-v1.json`.

## Action-first search

Six action classes received exactly 65,536 raw cells each:

1. source-conditioned permittivity;
2. a screened auxiliary scalar;
3. a two-scale bi-Helmholtz auxiliary scalar;
4. two mixed screened fields;
5. a convex nonlinear auxiliary scalar; and
6. a baryon-adaptive gradient action.

These are normalized static radial actions in `x=log(r)`. They are not asserted to be covariant
four-dimensional theories. Known action-based modified-Poisson and bi-potential constructions are
explicit prior art ([AQUAL review](https://arxiv.org/abs/0901.1524),
[QUMOND](https://arxiv.org/abs/0911.5464)), as are auxiliary-field representations
([Rodrigues et al.](https://arxiv.org/abs/1101.5028)). The design also records the known warning
that localized nonlocal actions can introduce ghost-like modes
([De Felice and Sasaki](https://arxiv.org/abs/1412.1575)). The two nonlinear/adaptive lanes are
only potentially new observational syntheses; historical novelty is not claimed.

Every action uses the common gravitational sector

```text
L_Phi = epsilon_c (d_x Phi)^2/2 - source_b Phi,
```

where `epsilon_c` is positive and constructed only from the frozen baryonic source and auxiliary
solution. Automatic variation gives

```text
d_x(epsilon_c d_x Phi) + source_b = 0.
```

After radial flux integration,

```text
epsilon_c g = g_bar,
g = nu_c g_bar,
epsilon_c = 1/nu_c.
```

The candidate cell is part of the action through

```text
nu_c = 1 + A u^(-p)/(1+u/u_t) (0.05+0.95 H_action),
u = g_bar/a0.
```

This is action-first in the scoped radial sense: the actions, sources, coefficients, variations,
and admission gates were committed before scoring. It is not a relativistic derivation of the
zero-slip lensing closure.

## Formal and numerical admission

- **393,216** raw action cells were generated.
- **32,256** survived; each of the six classes retained exactly **5,376**.
- All six symbolic action templates have exact zero Euler residuals.
- All six source-free gravitational shift identities are exact.
- Six malformed controls were rejected: negative permittivity, missing source, indefinite field
  mixing, negative quartic coupling, incorrect bi-Helmholtz coefficient, and response leakage.
- The response-blind profile stage solved **4,608** action/profile systems.
- The maximum relative discrete Euler residual is `2.36e-12`, below the frozen `1e-8` ceiling.
- The smallest reduced static Hessian eigenvalue is `0.01813`, strictly positive.
- All **96** symbolic recipes produce distinct behaviors on the development predictors.

These checks prove exact normalized radial variation and reduced static convexity. They do not
prove a bounded four-dimensional Hamiltonian, healthy propagating degrees of freedom,
hyperbolicity, causality, or nonlinear relativistic stability.

## Selected action

The selected full-data candidate is `245512`, a known-family combination. Its reduced source action
is

```text
E = integral [
  (h^2 + chi^2 + ell^2 h_x^2 + ell^2 chi_x^2)/2
  - kappa h chi
  - J h
] dx,
```

with baryonic source `J` equal to the bounded logarithmic mass-profile slope, `ell=1.8`, and
`kappa=0.8`. Its automatically derived auxiliary equations are

```text
(1-ell^2 d_x^2) h - kappa chi = J,
(1-ell^2 d_x^2) chi - kappa h = 0.
```

The gravitational coefficient uses

```text
H_action = 0.5 + 0.5 h/(1+abs(h)),
nu = 1 + 6 u^(-0.2)/(1+u/10) (0.05+0.95 H_action).
```

Every fold selected this exact action, source, `ell`, `kappa`, `A`, `p`, and `u_t`. That is unusually
stable selection, but the selected family is built from known screened and mixed auxiliary-field
ideas and is not historically new.

## Galaxy-and-cluster result

The primary score gives equal weight to S4TM and CLASH and equal weight to each object within a
population.

| Model | Balanced loss | S4TM | CLASH |
|---|---:|---:|---:|
| Item 45 geometry-density interaction | **0.7615** | 0.1878 | **1.3351** |
| Item 47 exterior operator | 0.7789 | **0.1635** | 1.3943 |
| Item 46 dimensionless generator | 0.8701 | 0.2475 | 1.4927 |
| Item 44 scale hierarchy | 0.9160 | 0.1664 | 1.6655 |
| Matched scale-free action | 0.9501 | 0.1606 | 1.7396 |
| **Item 48 action generator** | **1.8116** | **0.2048** | **3.4184** |
| Ordinary coordinate ridge | 1.8778 | 0.2605 | 3.4952 |
| Direct source-conditioned action | 2.1361 | 0.1855 | 4.0867 |

Relative to Item 45, the selected action is 9.05% worse on S4TM and 156.04% worse on CLASH. Relative
to the ordinary ridge it is 21.37% better on S4TM and 2.20% better on CLASH. The action-derived
auxiliary fields also improve the balanced result by 15.19% over the direct source-conditioned
action, although the direct action is slightly better on S4TM alone.

All four frozen baryonic-scale audits remain worse than Item 45. Every leave-one-object comparison
and the trimmed comparison retains the negative sign.

## Counterexamples and evidence policy

Relative to Item 45, 31 of 48 objects are raw counterexamples to the Item 48 action. Fifteen remain
mismatches across all four mass-scale audits: two S4TM lenses and thirteen CLASH clusters. These
objects are preserved with their identities and influence records.

The executable counterexample policy returns `QUALITY_LIMITED_EVIDENCE_RETAINED`. One mismatch is
never a kill switch, and a count from correlated, exposed, model-dependent data cannot prune an
action family. Conversely, retention is not success: the exact action is decisively noncompetitive
with Item 45 on this development representation.

## Compute and claim boundary

The RTX 5090 performed **21,224,448** candidate-point-fold evaluations. CPU and GPU selected losses
agree within the frozen tolerance. There were zero sealed-response accesses, zero post-evaluation
candidate cells, zero paid calls, and zero LLM-generated action cells.

Item 48 establishes that Invariant can generate actions first, derive their radial equations,
reject malformed actions, solve the survivors, and compare their unchanged motion/light response
across galaxy and cluster data. It does not establish a covariant action, a healthy relativistic
field theory, an alternative to GR, the absence of dark matter, or historical novelty.

The next ordered task is Item 49: seeded pseudorandom exploration. The action grammar, selected
two-field clue, failed cells, equivalence signatures, and every mismatch should become eligible
inputs to that reproducible stochastic search without retuning Item 48 on these responses.
