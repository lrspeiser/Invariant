# Sigma generated-formula priority report

## Outcome

The complete declared grammar contains 1,088,651,720 actions. Generator v2 retained 17,540,440
after its structural and original sampled-static gates. The RTX 5090 dense-static tier rejected
12,642,541 more on a 343-point lattice, leaving 4,897,899 actions in 36,047 structural families.

The first eight Pareto layers contain 124 discovery representatives. Two one-term families are kept
separately as controls. These are work priorities, not probabilities of truth.

## First Pareto layer

| Family | Ordinal | Representative correction |
|---|---:|---|
| `GF-53dd40c1364288d8` | 110 | `-(z)+(q)` |
| `GF-78308d0577387874` | 737 | `+(q)-(sqrt(1+(x*z))-1)` |
| `GF-bb30199987071bd0` | 739 | `+(q)+(sqrt(1+(x*z))-1)` |
| `GF-cb4ebf3da5a74582` | 723 | `+(q)+(q**2)` |
| `GF-d39051e5155d04d2` | 32,885 | `+(q)-(x)+(x*z)` |
| `GF-e8e7932f868cbc42` | 111 | `+(z)+(q)` |

Here `x = D^2/a_sigma^2`, `q = L_sigma^2(partial D)^2/a_sigma^2`, and
`z = Z_b^2/Z_0^2`. The listed expression is the sparse correction with the frozen shared coupling;
it is not by itself a covariantly complete theory.

## Required next test

The first layer should enter a covariant-lift/variation queue. Each proposed lift must then pass
kinetic-rank, Hamiltonian, constraint, degree-of-freedom, characteristic, covariance-identity, GR,
and Solar-System hard gates. No galaxy or cluster fit should be used to rescue a failure.

## Audit status

- All 17,540,440 compact survivor records verified.
- All 260 dense-status hashes verified.
- GPU/CPU status agreement: 508/508 deterministic witnesses.
- Floating-point ambiguity-band count: zero.
- Dark matter, redshift-derived distance, supernova distance, and derived GR/NFW targets: excluded.

The machine-readable queue is `generated-priority-dense.json`; its records include family counts,
ordinals, term IDs, signs, expressions, Pareto layers, mechanism tags, and historical theory signals.
