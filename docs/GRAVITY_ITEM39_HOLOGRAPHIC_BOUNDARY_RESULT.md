# Gravity Item 39 holographic/boundary-gravity result

Date: **2026-08-29**

## Decision

`NONPROMOTED_ITEM39_HOLOGRAPHIC_BOUNDARY_MIXED_DIAGNOSTIC`

The search found one universal boundary-dependent formula that is worth preserving, but Item 39
does not pass promotion. On the usable WALLABY rotation curves, the formula is much better than
baryonic Newtonian gravity and the frozen Item 38 formula, and marginally better than fixed MOND
RAR. It is nevertheless **24.37% worse** than a matched ordinary geometry model, **2.48% worse**
than a flexible radial/surface model, and has paired whole-galaxy `p=0.896`. Only 23 of the 60
selected exploration galaxies pass the frozen curve-quality rules, below both quality thresholds.

The same formula, unchanged, is encouraging on a six-object SWELLS strong-lensing aperture
diagnostic: its mean squared log-mass error is **42.51% lower** than fixed MOND RAR and lower than
all three fixed controls. That advantage survives four of five frozen systematic variants, but a
`+0.25 dex` stellar-mass shift reverses it. The SWELLS calculation was unblinded after the WALLABY
selection and uses published aperture summaries rather than raw lens images. It is therefore a
useful cross-domain consistency observation, not an independent confirmation or a derived
relativistic lensing theory.

The complete aggregate receipt is
`runs/gravity/roadmap/item-39-holographic-boundary-v1.json`. No formula or formula family is
pruned by these data.

## What was tested

The hypothesis was that nested baryonic boundary information could generate one gravitational
multiplier that works for both the motion of stars and the bending of light. The weak-field test
closure fixes `Phi=Psi`, so dynamics and lensing cannot receive separately tuned responses.

Exactly 262,144 formula cells were divided equally among four mechanism niches:

1. surface/bulk equipartition mismatch;
2. Brown–York-like quasilocal contrast;
3. an entanglement-wedge cross-ratio proxy; and
4. nested-screen renormalization flow.

The admissibility rules retain 107,325 cells and 107,268 behavioral equivalence classes. The
first two lanes are known holographic families plus phenomenological observational projections.
The last two are labeled only as potentially new observational syntheses. That label does not
establish historical novelty, and no microscopic derivation was found.

## Selected formula

The full exploration selected candidate `173808`, from the entanglement-wedge cross-ratio lane.
Define

- `u = g_bar/a0`;
- `f_M = M_bar(<r)/M_bar,total`;
- `x = r/R_screen`; and
- `H = sqrt[4 f_M (1-f_M) 4x/(1+x)^2]`.

The universal multiplier is

`nu = 1 + 1.25 u^(-0.5) [1 + (u/10^8)^0.2]^(-5) [0.05 + 0.95 sin(pi H/2)^0.2]`.

For motion, `g_pred = nu g_bar`. For the bounded SWELLS aperture proxy, the same zero-slip closure
uses `M_lens,pred = nu M_bar,projected(<R_E)`. The second expression is a diagnostic translation,
not a solution of the full lens equation.

## Fresh WALLABY dynamics experiment

The official WALLABY pilot DR2 source and kinematic tables supplied source-only predictors. Every
galaxy used by the earlier Item 10 WALLABY work, including reserved roles and nearby coordinates,
was excluded before response access. The source stage found 135 fresh eligible profiles. A
response-blind mass-by-screen-ratio design selected 60 exploration galaxies and reserved 15
confirmation galaxies.

Only after the sample-freeze commit were the 60 exploration rotation responses opened. The 15
confirmation responses remain sealed. The frozen extraction rules leave 23 usable galaxies and
170 radial points. Thirty-seven selected galaxies fail without replacement, mostly because their
published curves contain too few usable points. The frozen minimum was 30 passing galaxies and a
65% retention fraction; both gates fail.

The primary score is equal-galaxy mean squared standardized log-velocity error under five outer
folds.

| Model | Out-of-fold loss | Selected formula relative result |
|---|---:|---:|
| Baryonic Newton | 87.2320 | better |
| Frozen Item 38 formula | 41.8279 | better |
| Fixed MOND RAR | 5.4443 | **0.74% better** |
| Flexible radial/surface ridge | 5.2732 | **2.48% worse** |
| Matched ordinary geometry ridge | **4.3451** | **24.37% worse** |
| Selected boundary formula | **5.4040** | — |

The formula loses in several mass and screen-ratio strata. Leaving out the most influential galaxy
does not change the negative sign, and the distance, inclination, and stellar-mass audits do not
rescue it relative to the matched geometry control. The paired sign-flip value is `0.896`, not
close to the frozen `0.05` gate.

There are 13 raw galaxy-level counterexamples relative to the strongest ordinary control; four
remain counterexamples under the frozen uncertainty audits. They are retained as evidence, not
used as one-strike vetoes. Because the overall data-quality gate fails, the executable policy
classifies the dynamics evidence as `QUALITY_LIMITED_EVIDENCE_RETAINED`.

## Unchanged SWELLS light-bending diagnostic

After the WALLABY-selected formula was commit-bound, the transfer used primary SWELLS III and V
tables. The originally entered arXiv identifiers were found to point to unrelated papers and were
corrected to `arXiv:1201.1677v2` and `arXiv:1206.4310v2` before local source acquisition. This
bibliographic correction changed no formula, coefficient, sample score, or gate.

The sample rule takes the complete usable intersection of published lens mass and Einstein radius,
independent Chabrier bulge/disk masses, and component half-light radii after the predeclared
exclusions. It yields six lenses, with no replacements. Projected Sérsic `n=4` bulges and `n=1`
disks determine baryonic aperture mass. The screen radius is fixed at four disk half-light radii.
The published stellar-to-lensing mass-fraction columns are not used as predictors.

| Model | Mean squared log-mass error | Selected formula improvement |
|---|---:|---:|
| Baryonic Newton | 0.10666 | +50.73% |
| Frozen Item 38 formula | 0.10632 | +50.57% |
| Fixed MOND RAR | **0.09141** | **+42.51%** |
| Unchanged boundary formula | **0.05255** | — |

Five of six lenses favor the boundary formula relative to the strongest fixed control. One is a
raw counterexample, and none remains a stable counterexample across all five systematic variants.
The formula remains better under `-0.25 dex` stellar mass, `+0.10 dex` missing gas, and screen
radii of three or five disk half-light radii. Under `+0.25 dex` stellar mass it becomes **8.89%
worse** than fixed MOND. Thus the positive result may be telling us about a boundary response, an
incorrect baryonic mass scale, or both; these data cannot distinguish them.

The complete SWELLS Table 1 was inspected after selecting the WALLABY formula but before freezing
the aperture mapping. This is recorded as an incomplete empirical audit rather than hidden. It
prevents the lensing diagnostic from authorizing promotion even though no formula or coefficient
was retuned.

## Compute and reproducibility

The NVIDIA GeForce RTX 5090 evaluated 18,245,250 admitted candidate-point combinations. The
same-input CPU/GPU maximum absolute log-velocity difference is `4.44e-16`. The first replay audit
identified that CPU and GPU paths had reconstructed one boundary coordinate from differently
rounded representations; both now consume the identical frozen coordinate. Scientific replay
checks compare every deterministic field and ignore only measured wall-clock duration.

The dynamics, lensing, and aggregate receipts replay exactly. There were zero paid model calls,
zero post-response formula cells, zero post-selection lensing formula cells, and zero accesses to
the 15 reserved WALLABY confirmation responses.

## What this means in plain language

The search found a rule that asks not only “how much ordinary matter is inside this radius?” but
also “how is that matter divided between the inside and the galaxy's outer boundary?” That extra
boundary description does not beat a conventional flexible geometry model on the limited usable
rotation sample. However, when carried unchanged to six spiral lenses, it moves the predicted
light-bending mass in the right direction more accurately than the fixed comparison formulas.

That is interesting evidence for further testing, not a solution to galactic gravity. The result
could disappear with better stellar masses, more complete gas measurements, a blinded lens sample,
or direct image modelling. Conversely, surviving those tests would make the shared motion/light
behavior materially more interesting.

## Claim boundary and next action

Item 39 does not establish holographic gravity, modified gravity, an alternative to general
relativity, the absence of dark matter, or a historically new formula. It establishes that the
pipeline can generate and GPU-screen a large boundary-law family, protect confirmation data,
retain rather than kill imperfect-data counterexamples, and transfer an unchanged candidate from
galaxy motion to a separate light-bending diagnostic.

Preserve candidate `173808` as a nonpromoted lead. Its decisive future tests are a fresh rotation
sample that passes the frozen quality floor and a prospectively blinded direct-lensing analysis
with raw image likelihoods, measured gas, and stronger stellar-mass constraints. Keep the 15
WALLABY confirmation galaxies sealed. Advance the ordered mechanism search to Item 40, discrete
or network gravity, without using the Item 39 lensing result to retune Item 39.
