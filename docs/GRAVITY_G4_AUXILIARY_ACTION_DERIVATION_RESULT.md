# G4 auxiliary-action derivation result

Date: **2026-08-27**

## Decision

`BLOCK_G4_AUXILIARY_ACTION_DERIVATION`

The sealed receipt is `runs/gravity/g4/auxiliary-action-derivation-v6.json`. This is a scoped
derivation success inside an overall G4 block: one effective radial action generates both terms of
the fold-stable v5 parent and predicts its coefficient and scale from a post-v5 disk-dimension
hypothesis. It does not produce a covariant gravity theory or pass the observational gate.

## The effective action

Let `x = log(r)` and define the inherited baryonic occupancy

```text
q(x) = (g_bar/g_dagger) / ((g_bar/g_dagger) + 0.1).
```

The action contains the radial gravitational potential `Phi`, a screened field `psi`, a directed
occupancy field `chi`, and a Lagrange multiplier `eta`:

```text
A(chi) = 1/(1 + chi/D)
J(chi,psi) = g_dagger psi / (D(1 + chi/D))

L_Phi = A(chi) (d_x Phi)^2/2 - J(chi,psi) d_x Phi - source_b Phi
L_psi = (psi^2 + ell^2 (d_x psi)^2)/2 - q psi
L_chi = eta (chi + ell d_x chi - q).
```

Exact symbolic variation gives

```text
psi - ell^2 d_x^2 psi = q
chi + ell d_x chi = q.
```

The first equation has a symmetric exponential Green function; the second has an interior-directed
exponential solution. The integrated radial flux equation is

```text
A(chi) g - J(chi,psi) = g_bar,
```

which simplifies exactly to

```text
g = g_bar (1 + chi/D) + g_dagger psi/D

V^2 = V_bar^2 + [V_bar^2 chi + r g_dagger psi]/D.
```

All three symbolic residuals—the screened Euler equation, directed constraint, and integrated flux
identity—are exactly zero. The radial auxiliary quadratic Hessian is `diag(1, ell^2)`, so this
limited static sector is positive for nonzero real `ell`.

## Dimension closure and numerical reproduction

The post-v5 closure hypothesis takes the effective baryonic support dimension of a thin disk to be
`D = 2` and sets

```text
beta = 1/D = 1/2
ell = 1/(2D) = 1/4.
```

No velocity was used to choose `D` inside this derivation. With boundary-normalized discrete Green
operators, the action reproduces the v5 parent with **zero chi-square difference** and the same
prediction-manifest hash. It has zero galaxy-specific fitted gravitational constants.

The frozen dimensional counterfactual was:

| Support dimension | Predicted beta | Predicted ell | Chi-square |
|---:|---:|---:|---:|
| 1 | 1 | 1/2 | 1,157,380.924 |
| **2** | **1/2** | **1/4** | **138,636.609** |
| 3 | 1/3 | 1/6 | 242,539.345 |

`D = 2` therefore beats both declared alternatives decisively. This is useful internal support for
the closure relationship, but it was proposed after v5 exposed the winning numerical values and is
not independent confirmation or a fundamental derivation of spatial dimensionality.

## Remaining blockers

The action remains **6.06% worse than RAR** and improves 67 of 139 exploration galaxies. It exceeds
the NFW-plus-slack ceiling by **105,177.802**. More fundamentally:

- the transition `g_bar/g_dagger = 0.1` is inherited from v5 and not derived;
- `x = log(r)` is a radial coordinate, so the directed constraint is not causal time evolution;
- there is no four-dimensional covariant completion or Noether conservation identity;
- radial auxiliary positivity is not a complete ghost, gradient, or nonlinear stability proof;
- the same fields have not generated cluster dynamics or lensing;
- Solar-System and gravitational-wave limits remain unresolved.

No confirmation, cluster, or lensing data were accessed. No alternative to GR or historical novelty
is claimed.

The next derivation gate is now narrower: derive the occupancy transition from a field potential or
symmetry-breaking scale, embed `Phi`, `psi`, and the directed response in a covariant action, and
obtain the weak-field radial action plus a same-field lensing equation as controlled limits. If that
cannot be done without an arbitrary preferred radial direction or an unseen matter-like stress
tensor, this parent should be rejected rather than further fitted.
