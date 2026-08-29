# Gravity Item 44 scale-hierarchy result

Date: **2026-08-29**

## Decision

`NONPROMOTED_ITEM44_SCALE_HIERARCHY_RESULT_RETAINED`

Item 44 found the clearest cross-scale pattern in the recent mechanism search, but
it remains a retrospective lead rather than a discovery. Across 28 S4TM galaxy
lenses and 20 CLASH clusters, a formula based on the ratio between measurement
radius and the baryonic transition length improves the matched scale-free formula
by **9.14%** in balanced whole-object cross-validation. It improves in both
populations separately, beats the ordinary coordinate ridge, and all five outer
folds independently select the same mechanism niche.

The pattern does not pass promotion. Its object-level paired sign-flip value is
`p=0.1323`, both datasets were already exposed before Item 44, and a `-0.25 dex`
S4TM stellar-mass audit changes the 9.14% nominal advantage to a **0.09% deficit**.
The data therefore support preserving and prospectively testing this exact idea,
not claiming a new law.

The complete aggregate receipt is
`runs/gravity/roadmap/item-44-scale-hierarchy-v1.json`. No sealed response or paid
model call was used, and no formula family was pruned.

## The first-principles question

Item 43 showed that a direct finite-horizon factor could fit galaxy lenses modestly
but collapse at cluster radii. Item 44 asked whether the physically relevant input
is not one absolute distance, but the *ordering of several lengths*:

- baryonic size `R_b`;
- measurement or lensing aperture `R`;
- transition length `r_M=sqrt(G M_bar/a0)`;
- Schwarzschild length `r_s=2 G M_bar/c^2`;
- weak-field curvature radius `L_K=sqrt(R^3/r_s)`;
- acceleration wavelength `lambda_a=c^2/a0`; and
- background horizon `R_H=c/H(z)`.

These ingredients are known. The potentially creative part is asking the generator
to combine their dimensionless orderings and then requiring one unchanged mapping
to work for both galaxies and clusters. Algebraic novelty is not a historical
novelty claim.

## Formula search

Exactly 262,144 formulas were divided equally among four mechanisms:

1. baryonic size versus transition length;
2. aperture radius versus transition length;
3. curvature radius versus the geometric mean of acceleration wavelength and
   horizon; and
4. a four-scale closure combining size, transition, curvature, wavelength,
   aperture, and horizon.

Each raw ratio `x` is converted to a bounded matching coordinate

`H = 1/[1 + |log10(x)|]`,

which is largest when the two sides of the proposed hierarchy are comparable. The
universal weak-field grammar is

`nu = 1 + A u^(-p)/(1 + u/u_t) [0.05 + 0.95 H^s]`,

with `u=g_bar/a0`. The high-acceleration Newtonian limit and positivity gates admit
201,828 cells. A matched scale-free search uses the same `A`, `p`, and `u_t` space
with `H=1`; this isolates whether the length ratio itself adds information.

## Joint real-data result

For S4TM, the target is projected Einstein mass relative to the Chabrier stellar
mass inside the aperture. For CLASH, the target is published total acceleration
relative to baryonic acceleration. The primary score gives equal weight to the two
populations and equal weight to every whole object inside a population. Five folds
hold out whole lenses and whole clusters together.

| Model | Balanced loss | S4TM | CLASH |
|---|---:|---:|---:|
| Baryonic Newton | 67.65 | 0.91 | 134.39 |
| Fixed MOND/RAR | 21.00 | 0.78 | 41.21 |
| Ordinary coordinate ridge | 1.99 | 0.27 | 3.72 |
| Matched scale-free search | 1.01 | 0.18 | 1.84 |
| Scale-hierarchy search | **0.92** | **0.17** | **1.67** |

All five folds choose the aperture-transition niche with `s=0.2` or `0.3`. The
formula selected on all development data is candidate `125057`:

`nu = 1 + 4 u^(-0.5)/(1 + u/10) [0.05 + 0.95 H^0.3]`,

where

`H = 1/[1 + |log10(R/r_M)|]`.

In plain language, the extra gravitational response is strongest when the radius
being measured is near the length at which the enclosed baryonic mass naturally
crosses the acceleration scale `a0`. It weakens gently when the aperture is much
smaller or much larger. This is closely related to known acceleration-scale ideas;
the result is a scale-hierarchy reformulation and cross-population test, not a new
fundamental derivation.

The hierarchy improves the matched scale-free law by **8.01%** on S4TM and **9.25%**
on CLASH (differences rounded from the recorded losses). Its overall advantage
survives every leave-one-object calculation and a 10% trimmed calculation. There
are 23 raw object-level counterexamples, because an aggregate improvement need not
help every object. Six remain mismatches across the four global mass-scale variants.
The executable policy returns `QUALITY_LIMITED_EVIDENCE_RETAINED`: count alone is
not decisive, and exposed model-dependent data cannot terminally reject or confirm
the formula.

## Mass-scale sensitivity

The frozen-fold formula and matched control were replayed without reselection under
four global baryonic-mass shifts:

| Audit | Hierarchy vs scale-free |
|---|---:|
| S4TM stellar mass `-0.25 dex` | **0.09% worse** |
| S4TM stellar mass `+0.25 dex` | 14.10% better |
| CLASH baryonic scale `-0.10 dex` | 11.25% better |
| CLASH baryonic scale `+0.10 dex` | 0.68% better |

This asymmetry is exactly why imperfect data must not be treated with a one-strike
rule. The nominal hierarchy could reflect a real scale ordering, a Chabrier stellar
mass offset, or a mixture of both. Better stellar masses can distinguish those
possibilities; this dataset cannot.

The first retrospective evaluator omitted this required audit. After detection,
the four predeclared shifts were executed with every fold selection fixed. No
formula, coefficient, object, or nominal result changed, and no post-evaluation
candidate was added.

## Compute and claim boundary

The NVIDIA GeForce RTX 5090 performed **114,806,160** candidate-point-fold
evaluations. CPU and GPU selected losses agree to `1.11e-16`. There were zero paid
calls, zero sealed confirmation accesses, and zero post-evaluation candidate cells.

Item 44 does not establish modified gravity, an alternative to general relativity,
the absence of dark matter, a carrier field, or a historically new formula. It does
establish a concrete, reproducible cross-scale lead: `R/r_M` carries incremental
information beyond the same scale-free acceleration formula in these exposed
summaries.

The correct next test is prospective. Freeze candidate `125057` unchanged and test
it on genuinely new galaxy lenses plus independently reduced cluster lensing with
better baryonic-mass constraints. Keep every mismatch in the counterexample store.
Regardless of that future test, continue the ordered roadmap to Item 45, universal
interaction variables.
