# Gravity Item 18: gravitational antiscreening

## Outcome

**SCOPED REJECT; ITEM COMPLETE.** The frozen positive weak-field antiscreening family does
not predict fresh DiskMass galaxy rotation measurements beyond ordinary baryonic-mass
calibration. It improves dramatically over one deliberately fixed mass normalization, but
is **4.25% worse** than Newtonian gravity after the same universal stellar and gas scales are
selected on training galaxies, and **15.64% worse** than a fixed flexible baryonic/structural
control. The selection-aware permutation result is `p=0.71`.

This rejects one finite, saturating, acceleration-dependent constitutive family in its
one-radius disk approximation. It does not reject all gravitational antiscreening, a full
nonlinear disk-field solution, or relativistic theories that predict both motion and light.

## Frozen physical proposal

The test used

`nu(g_bar) = 1 / [1 - A / (1 + (g_bar/a0)^p)]`,

with `g_pred = nu(g_bar) g_bar`, `0 < A < 1`, one universal transition acceleration `a0`,
and one universal exponent `p`. Its strong-field limit is GR and its weak-field limit is a
finite enhancement `1/(1-A)`. The proposal was motivated by a QUMOND/AQUAL-adjacent
auxiliary-potential action. It is labeled **known-family combination**, not a novel theory.

The pre-response physics filter required a fractional deviation no larger than `1e-5` at
the Sun's one-AU acceleration and a positive constitutive denominator. It admitted 621,750
of 750,300 exact parameter cells. The grid also included universal, not per-galaxy, stellar
mass-to-light and gas-mass nuisance scales.

## Real-data boundary and blindness

Predictors came from three public primary catalogs:

- the 2010 DiskMass sample for disk scale length and photometric structure;
- the corrected ALFALFA 100% catalog for directly measured integrated H I mass;
- the predictor-only columns of the 2025 DiskMass H-alpha release for updated distance and
  K-band luminosity.

The response was the later 2025 H-alpha velocity-field fit, transformed at the frozen radius
as `V(2.2 h_R) = V_rot tanh(2.2 h_R / h_rot)`. No H I width, dark-halo fit, old DiskMass
mass fraction, or confirmation response entered the candidate.

The source audit disclosed and removed two contaminated identity sets before freezing:

- all 30 PPak galaxies whose older 2013 mass-fraction values had been viewed while assessing
  candidate sources; and
- nine additional 2025 rows whose rotation values appeared in a web-search snippet.

The remaining exact predictor join contained 43 galaxies. Predictor-only HMAC rules reserved
eight confirmations and assigned 35 exploration galaxies evenly across five folds. The
response query requested only those 35 exploration identities. Twenty-three passed the
preregistered quality rules; eleven rejected rows carried the catalog's low-inclination
flag, with one of those also failing the azimuthal-asymmetry limit, and one further row failed
the rotation-curve-asymmetry limit. The frozen minimum was 20, so the quality gate passed.
All eight confirmations remain unqueried.

## Predictive result

The primary loss was equal-galaxy mean squared log-velocity residual.

| Model | Held-out MSE | Antiscreening change relative to model |
|---|---:|---:|
| Fixed baryonic GR (`Upsilon_K=0.5`) | 0.19309 | **59.80% better** |
| Calibrated baryonic GR | 0.07447 | **4.25% worse** |
| Fixed-slope baryonic Tully-Fisher | 0.08340 | 6.92% better |
| Flexible baryonic/structural nuisance | 0.06713 | **15.64% worse** |
| Antiscreening candidate | 0.07763 | — |

The candidate also lost to calibrated GR in both mass halves (`-3.68%`, `-5.77%`) and both
gas-fraction halves (`-4.98%`, `-3.25%`). Its gas-disk geometry replay changed sign:
`-5.99%` for a `1.5 h_R` H I scale and `+6.65%` for `3 h_R`. That sensitivity is a concrete
warning that integrated H I mass plus an assumed radial profile is not yet a sufficient
geometry measurement.

All five outer folds selected `p=4`, transition accelerations between `1.0e-11` and
`3.16e-11 m/s^2`, and positive enhancement. This is a stable direction, but not a predictive
success. Every fold put stellar `Upsilon_K` at the frozen lower boundary `0.2` and the gas
scale at `1.0`; calibrated GR independently chose `Upsilon_K=0.2` in four folds and `0.242`
in the fifth. Thus the apparent gain over fixed GR is dominated by baryonic calibration,
not evidence for antiscreening.

The selected family did worse than calibrated GR on 12 of the 23 valid galaxies. The full
selection repeated under 99 null permutations gave `p=0.71`. A frozen synthetic
antiscreening injection was recovered by the exact injected cell in all five folds with
100% improvement over GR, while the known-GR control was not spuriously improved. The
negative real-data result is therefore not caused by an evaluator unable to recognize its
own hypothesis.

## Failure-space contribution

This attempt excludes the following region as a promoted explanation on this data
representation:

- positive-only, finite low-acceleration enhancement;
- one universal transition acceleration and exponent;
- one universal stellar and gas calibration;
- algebraic application to exponential stellar and H I disk accelerations at `2.2 h_R`;
- one-AU agreement with GR at `1e-5`.

The result leaves open full nonlinear field solves using measured radial gas and stellar
profiles, density- or environment-dependent antiscreening, gravitational slip, and
relativistic lensing predictions. Those are materially different mechanisms and must not be
declared equivalent to this failed cell family.

## Reproducibility and next action

- Science freeze: `9fc36c647c0a70aea5b2921d7d3ff640a43b4f9d`
- Sample freeze: `01987ebcabd00b757af67d6a3fd68610b840d0e6`
- Result receipt: `runs/gravity/roadmap/item-18-diskmass-antiscreening-v1.json`
- Source and compute receipts: `runs/gravity/roadmap/item-18-diskmass-antiscreening-v1-source/`
- Compute: RTX 5090/CuPy, 5,906,625,000 null-inclusive training residual evaluations in
  3.68 seconds; zero paid API calls and zero API spend.

Advance to Item 19, massive gravitational particles. Preserve this exact finite
antiscreening family as a tested known-family region. Do not retune the 23 opened valid
responses or query the eight reserved confirmations.
