# Gravity roadmap Item 56: disk-galaxy gate

## Outcome

Item 56 is complete. The fixed Item 45 geometry-density formula did **not** pass the
disk-galaxy predictive gate on the 139 admitted SPARC exploration galaxies. Its
equal-galaxy standardized loss was `839.1780`, compared with `378.4520` for baryonic
Newton, `33.5560` for the empirical RAR, and `7.1221` for the fold-trained NFW
performance ceiling.

The decision is:

`ITEM56_DISK_GALAXY_GATE_NOT_PASSED_LEAD_AND_FAILURES_RETAINED`

This is a strong failure of the unchanged Item 45 parameterization on this tested
dataset. It is not a terminal rejection of every geometry-density interaction, because
the SPARC exploration responses were previously exposed elsewhere in the repository,
their published errors omit important covariance, and no unchanged independent
replication has yet been run. No formula family was pruned.

## What was frozen before evaluation

- Scientific-freeze commit: `53a890ecee7fc5912b2bd4c39175e5ea733179c6`
- Commit-bound config: `a28676a125f72848ff71a09c7391a233d53cb4d9`
- Serialization-only repair after the first non-persisted attempt: `44dcff5dd4593bfc1ecbe2102f1b548e27baea46`
- Candidate: Item 45 candidate `135082`, with no SPARC fit and no galaxy-local
  gravitational parameter.
- Formula inputs: radius, the published gas/disk/bulge baryonic velocity components,
  fixed stellar mass-to-light ratios, and the published 3.6-micron effective radius.
- Forbidden predictors: galaxy identity, observed velocity, its uncertainty, halo
  parameters, and target-derived discrepancy.
- Population: 139 admitted SPARC exploration galaxies, 2,720 radial measurements, and
  693 contiguous radial folds.
- Sealed SPARC confirmation responses opened: 0 of 35 galaxies.
- Post-evaluation formula cells added: 0.

The first evaluation calculated the numerical result in memory but stopped before
writing it because a NumPy Boolean was not JSON serializable. The one-line repair only
converted that Boolean to the built-in JSON type. It changed no formula, threshold,
split, score, or outcome.

## Tested equation

For radius `R`, published half-light radius `Reff`, and baryonic acceleration ratio
`u=g_bar/a0`, the frozen coordinates were

`x_g = log10(R/Reff) / [1 + |log10(R/Reff)|]`

`x_d = log10(u) / [3 + |log10(u)|]`

`I = x_g tanh(2 x_d)`

`H = 0.5 + 0.5 tanh(2 I)`.

The unchanged response was

`nu = 1 + 6 u^-0.6 / (1 + u/100) (0.05 + 0.95 H)`

and `V_pred=sqrt(V_bar^2 nu)`. The formula was selected previously on galaxy-lens and
cluster-lens development data, not on SPARC rotation speeds. SPARC is nevertheless not
fresh confirmation at the repository level because its exploration responses had
already been studied by other branches.

## Main results

| Model | Equal-galaxy loss | Pooled chi-square | Two-sigma coverage |
|---|---:|---:|---:|
| Item 45 geometry-density | 839.1780 | 3,296,352.7 | 3.16% |
| Baryonic Newton | 378.4520 | 1,697,326.4 | 13.20% |
| Empirical RAR | 33.5560 | 130,714.7 | 40.74% |
| NFW performance ceiling | 7.1221 | 28,018.8 | 68.27% |

Relative to baryonic Newton, the candidate's equal-galaxy loss was 121.74% worse. It
beat Newton for 23 of 139 galaxies and beat RAR for 3: `UGC07399`, `NGC2915`, and
`NGC1705`. The one-sided paired sign-flip result against Newton was `p=1.0`, reflecting
that the aggregate direction was opposite the intended improvement.

The likely numerical symptom is over-enhancement: the amplitude `6` and low-acceleration
power `u^-0.6`, which were useful in the lens summaries, generally boost disk velocities
far too much. That is a diagnosis from the residual direction, not a derived physical
explanation.

## Radial holdouts

The fixed candidate was never refit, so every radius was prediction-only. The contiguous
blocks still expose whether a success or failure is confined to one part of a galaxy.

| Radial block | Role | Item 45 | Newton | RAR | NFW |
|---:|---|---:|---:|---:|---:|
| 0 | inner | 217.70 | 44.25 | 17.36 | 11.66 |
| 1 | intermediate | 706.13 | 264.97 | 30.53 | 8.25 |
| 2 | intermediate | 1,131.69 | 436.03 | 35.18 | 3.55 |
| 3 | intermediate | 1,283.11 | 656.87 | 46.12 | 5.12 |
| 4 | outer | 793.72 | 461.67 | 36.70 | 8.21 |

The candidate lost to both Newton and RAR in all five radial blocks. This is not merely
an edge-of-galaxy miss.

## Data-quality and nuisance audits

Four frozen sensitivity variants were run. The candidate beat Newton only when all
baryonic mass components were shifted down by `0.25 dex`; even there its equal-galaxy
loss was `373.64`, versus `78.40` for RAR. It did not beat Newton under a `+0.25 dex`
baryonic shift or at either inclination-uncertainty endpoint.

The candidate was worse than RAR for 136 of 139 galaxies nominally. Of those, 113 stayed
worse in all four sensitivity variants. Every record is retained, including the three
nominal wins and all uncertainty-limited failures. The counterexample assessment is
`QUALITY_LIMITED_EVIDENCE_RETAINED`: no single object, no count of objects, and no finite
SPARC exploration sample is treated as a global family veto.

## What remains scientifically interesting

The exact Item 45 law is no longer promotion-eligible after Item 56. The broader
geometry-density question remains worth testing because Item 55 showed real predictive
reliance in the lens-development data, but the disk result says that its present
amplitude, transition, scale definition, or even its transfer assumption is wrong.

The three SPARC wins are useful diagnostic cases, not evidence for a subpopulation law.
Any repair must be declared as a new formula version and cannot retroactively turn this
frozen test into a pass. Item 57 should first test the unchanged law on an independent
galaxy pipeline; that can determine whether this broad failure replicates under different
measurements. The sealed SPARC confirmation set must remain closed.

## Compute and reproducibility

- Backend: NumPy on CPU; GPU not needed for one fixed formula.
- Nominal model-point predictions: 10,880.
- Fixed-candidate point predictions including four systematics: 13,600.
- Systematic model-point predictions: 32,640.
- NFW scale-training point evaluations: 695,808.
- Paid model calls and API cost: 0 and `$0.00`.
- Evaluation content SHA-256:
  `b77c3f75319adda54628beda58c50430b6941cc4e803fd611a2fc5215e4e4393`.
- Aggregate content SHA-256:
  `2f65f10b9a42343e5058b430830cac5aec5055a844822522ed3396fc8cc8e563`.

Replay with:

`python -m sigma_theory_compiler.gravity_item56_disk_galaxy_gate replay`
