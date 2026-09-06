# Gravity relay concepts: first comparative experiments

The best mathematical starting point is a distributed, conservative secondary
response around ordinary matter. Finite memory is worth keeping as a separate
extension. Absorption alone does not supply extra attraction in the tested model.
These findings rank proposed equations, not explanations established in nature.

Status: THEORY_BENCHMARK_ONLY. Three parallel agents tested absorption and
redirection, distributed sources, and memory/feedback. The coordinator reproduced
published halo profiles, fitted fixed return formulas and checked disk geometry.
Twenty combined tests pass. No observed galaxy velocity, cluster or lensing
response was newly scored. These small deterministic tests ran on CPU; they did
not need the RTX 5090. Neither an RL search nor an exhaustive search of all
possible gravity laws was performed.

## Findings and repairs

| Direction | What worked | What challenged it | Specific next change |
|---|---|---|---|
| Distributed mini-generators | A flat manufactured disk produces an increasingly round extra field. At 64 kpc it reaches 98.64–99.89% of the point-calibrated halo strength; direction differs by at most 0.40 degrees. | At 8 kpc it reaches only 72.95–74.11%, with up to 7.95 degrees direction difference. Kernel strength and length were supplied from a fitted halo. | Predict those parameters from ordinary-matter observables, then test extended real sources without halo-based retuning. |
| Partial absorption and re-emission | Same mean opacity, different clumping: 90.00% transmission for 10% covering fraction versus 36.79% for a uniform screen. | Both remain below vacuum transmission. Ten lossless half-forward relays at opacity one leave 2.24% forward. | Retain the arrangement diagnostic; add a specified conservative redistribution mechanism if seeking enhancement. |
| Delayed, repeated response | Stable feedback can build a larger secondary source and favor broad spatial modes. | A gain of 100 comes with a relaxation time of 100 tau; after 10 tau only 9.52% of equilibrium exists. Feedback at or above one has no bounded equilibrium in the normalized toy system. | Use finite kernels, subcritical feedback and explicit source histories. Explore saturation as a new model, not a silent repair. |
| Finite cumulative return | One two-parameter shape matches the concentrated NFW profile to 1.39% RMS at interleaved radii; another matches cored Burkert to 4.92%. | Outer-radius RMS errors rise to 21.98% and 34.90%; one fixed central behavior does not match both. | Test physically motivated central response and finite outer transition, with untouched galaxies for selection. |
| Scalar boost of disk gravity | It matches the halo direction in the disk midplane. | At R=4 kpc,z=2 kpc even the best scalar has 41.61% vector error. Spatially varying boosts generally introduce curl. | Construct the extra field from a potential or complete dynamics; do not merely multiply existing acceleration point by point. |
| Outer reflecting shell | A numerical Newtonian shell control matches the analytic solution to 3.91e-15. | A uniform exterior shell of ordinary Newtonian secondary sources gives no interior force. | A genuine reflection model needs its own coupling law; inward travel alone does not establish inward attraction. |

Radii and percentages above describe manufactured fields or published fitted
profiles. They are not measured dark halo shapes or confidence intervals.
The forward-relay bound excludes cross-ray focusing and repeated backscatter;
it is not a general theorem against local enhancement by redistribution.

## Candidate equations

For the leading spatial candidate, an ordinary source element dm at separation
s contributes

`dPhi_extra = -G eta dm log(1+s/L)/s`.

Taking minus its gradient gives the additional acceleration. Summing the
potential over all ordinary matter determines both strength and direction.
This is a translation-invariant pair interaction, so it does not select a
special center. The equivalent spatial kernel is

`q(s) = eta/[4 pi L^3 x(1+x)^2]`, where `x=s/L`.

The single-point equivalence to NFW is exact by construction. The extended disk
test is where that equivalence ceases to be exact. The total integrated kernel
weight diverges logarithmically without a cutoff; this weight is not an energy.
A cutoff at 10L preserves the point-source force inside 10L and restores an
inverse-square outer force. At 100L it yields 41.1% of the untruncated target.

For the memory extension, the separately tested formula is

`tau dS/dt + S = K(rho_b + alpha S)`.

S is an effective source and K a finite spreading operator. The tested normalized
ring is a manufactured system, not a galaxy. A mode with eigenvalue lambda has
transfer function `lambda/(1-alpha lambda+i omega tau)`. The accumulated source
is not itself a measured force multiplier. Combining this temporal equation with
the distributed spatial kernel remains a next experiment; it was not validated
by testing the two pieces separately. Finite-speed propagation, physical energy
accounting and a photon metric have not been derived.

Two simple return shapes are also useful descriptive controls:

`g_extra = A/(1+r/L)^2` (finite central acceleration),

`g_extra = A (r/L)/(1+r/L)^3` (acceleration rising linearly from zero).

A has acceleration units. The first resembles NFW near the center; the second
resembles a cored Burkert halo. The radius-squared acceleration saturates in
both models, unlike an untruncated NFW/Burkert enclosed source. Their shape
parameters were fitted on r/rs=0.03–2, tested at interleaved radii and then at
3,5,8,12 rs. The fit was performed once per profile shape; rescaling it across
many galaxies supplies no new independent validation.

## Published-data foundation

We retained 525 original parameter rows (three selected halo/prior combinations
for each of 175 galaxies) from [Li et al. 2020](https://arxiv.org/abs/2001.10538),
and the author's Milky Way model from
[McMillan 2017](https://arxiv.org/abs/1608.00971). The source URLs and exact hashes
are in the sibling `mond-atlas-halo-return-001/source-receipts.json`.
All fitted stellar mass-to-light ratios, distances, inclinations, errors and fit
quality are preserved. Thirty rows have infinite published reduced chi-square;
they remain flagged, not removed. Different halo/prior fits use different
baryonic nuisance parameters; these cannot be mixed arbitrarily.

The exact cumulative-return identity H=r^2 g_h, H'=4 pi G rho_h r^2 reproduced
504 three-dimensional pilot vectors to 5.55e-14 relative by density integration.
Independent galpy verification agrees within 1.69e-12. This proves numerical
equivalence given a fitted halo function, not a first-principles explanation.
The full catalog generated 11,046 conditional profile targets, not that many
observations. Spherical continuation above/below a disk remains an assumption.

## Recommended next order

1. Keep the conservative distributed kernel as the baseline. Freeze one
   ordinary-matter rule for strength, scale and finite extent before scoring.
2. Evaluate stellar plus atomic and molecular source maps with their geometry
   uncertainty. Require converged fields and the existing noise/motion controls.
3. Compare the static model against distinct held-out galaxies. Use both central
   and outer behavior; selecting formulas only for outer agreement can miss a
   central failure.
4. Add memory only where source history is constrained. Time since a mass
   rearrangement is the relevant clock, not automatically stellar age.
5. Derive Solar System, cluster and lensing predictions before claiming one
   theory covers them. This milestone does not resolve those use cases.

## Reproducibility and detailed evidence

- [Absorption report](absorption/FINDINGS.md), 560 relay cases and shell supplement.
- [Distributed-source report](distributed/REPORT.md), 56 converged spatial points.
- [Memory report](delay/FINDINGS.md), 150 parameter cases and 48 spatial modes.
- [Combined verification](combined-verification.json), 20 tests and independent geometry identities.
- [Independent halo review](distributed/parent-review.json), exact source-row and force replay.
- [Halo comparison plot](../mond-atlas-halo-return-001/run002/halo-shape-comparison.png).

Run `python scripts/verify_mond_atlas_relay.py` for combined tests. Each branch's
report gives its runner and overwrite protection. Preserve existing outputs;
do not overwrite a published receipt. The failed first halo parser attempt is
retained in the sibling run001/FAILURE.md; it rejected published infinite fit
quality values before any target scoring. The corrected run002 preserves them.
