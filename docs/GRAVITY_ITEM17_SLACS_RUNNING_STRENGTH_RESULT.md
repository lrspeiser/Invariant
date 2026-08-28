# Gravity Item 17: one universal running-strength law on SLACS

## Decision

`REJECT_ITEM17_UNIVERSAL_RUNNING_STRENGTH_EXPLORATION`

The exact tested distance-running law is rejected as a promoted explanation. Its nested
out-of-fold predictions are worse than every frozen primary baseline, both response
channels, every broad mass/size slice, and all four stellar-population mass estimates.
This is a scoped rejection of one bounded no-slip running family, not a rejection of all
scale-dependent gravity.

## Real problem and frozen sample

The experiment joined two public catalogs for the original SLACS strong lenses:

- Bolton et al. (2008), VizieR `J/ApJ/682/964`, supplied target-blind redshifts,
  effective radii, luminosity, geometry, and later the exploration-only velocity
  dispersion and SIE Einstein radius;
- Grillo et al. (2009), VizieR `J/A+A/501/461/lens`, supplied four photometric stellar
  mass estimates based on Salpeter/Chabrier/Kroupa IMFs and BC03/Maraston population
  models.

Before either response was queried, the code sealed 45 exploration lenses and 12
confirmation lenses, with nine exploration assignments per outer fold. Response queries
were then issued by exact object name for the 45 exploration objects only. Forty-two pass
the frozen published-good-dispersion and numeric-quality rules; the 12 confirmation
responses remain unrequested and unread.

No total lensing mass, dark-matter fraction, or separate matter/light coupling was used.
The Einstein radius remains a published SIE image-model result rather than a direct image
likelihood, so this is not the later direct-lensing gate.

## One frozen law

Every non-GR cell instantiated the same known-family bounded logarithmic curve:

`mu(r) = G_eff(r)/G = 1 + A L/(1 + B L)`

`L = log[1 + (r/r0)^n]`.

The same `mu(r)` acted on matter and light; gravitational slip was fixed to one. Positive
and negative amplitudes were admitted. The curve returns to GR at short distance and
approaches the finite value `1 + A/B` at large distance.

The Cartesian grid contained 127,260 exact parameter cells. Before responses, 8,329 were
rejected for exceeding `|G_eff/G - 1| = 1e-5` at 1 AU or allowing `G_eff/G < 0.05`
between 1 AU and 10 Mpc. That left **118,931** admissible cells and no post-response
cells.

This is a phenomenological RG-like curve. It is not a microscopic beta-function
derivation, covariant action, stability proof, or quantum theory.

## Results

The primary Chabrier-BC03 result is:

| Comparison | Reference MSE | Running-law MSE | Relative change |
|---|---:|---:|---:|
| GR with one shared stellar-mass scale | 0.08856 | 0.09002 | **1.65% worse** |
| GR with separate dynamics/lensing calibration | 0.08216 | 0.09002 | **9.57% worse** |
| Fixed flexible nuisance model | 0.05737 | 0.09002 | **56.92% worse** |

The full 99-trial selection-aware null test gives `p = 0.62`. The running law is also
worse than shared-scale GR for:

- stellar dynamics: **0.55% worse**;
- Einstein-radius lensing: **3.24% worse**;
- low and high stellar mass;
- low and high effective radius;
- Salpeter-BC03, Salpeter-Maraston, Chabrier-BC03, and Kroupa-Maraston masses, with the
  exact primary formulas replayed and no reselection.

The selected direction is structurally consistent but unhelpful: all five folds choose
gravity growing with distance, all choose exponent `n = 4`, and four transition lengths
cluster near 26--35 kpc. The fifth transition is 3.31 kpc; amplitudes and saturation vary
widely. The required shared stellar-mass multipliers are already large, approximately
2.12--2.45 times the catalog's Chabrier-BC03 masses. Despite using both increased
baryonic mass and increased large-scale gravity, the model generalizes worse than GR.

Seven of fifteen formal gates pass. The failures are shared/separate/flexible baseline
improvement, both-channel improvement, all-strata improvement, permutation significance,
stellar-population robustness, and the originally frozen synthetic-injection control.

## Frozen control defect and audit

The original injected cell (`17017`) was a poor control choice: it produced only
`2.03e-6` GR baseline MSE, effectively a near-GR signal. Its candidate MSE was `2.55e-6`,
so the frozen recovery gate correctly remains failed.

A disclosed post-response diagnostic selected an already-frozen cell solely by maximum
prediction variance on the fixed radii, without consulting any observed response. For
that strong injection, the selector recovered the exact cell (`117261`) in all five folds,
reduced MSE from `1.09415` to `2.03e-6`, and achieved a 99.9998% improvement over GR.
This validates recovery of a strong signal but does not repair the frozen gate, alter the
real-data result, generate a formula, or authorize confirmation access. The exact audit
is preserved in `postresponse-control-audit.json`.

## Compute and replay

- Candidate-observable matrix values: **9,990,204**.
- Candidate training-residual evaluations including nulls: **3,996,081,600**.
- Device: **NVIDIA GeForce RTX 5090**, CuPy 13.5.1.
- Matrix construction: **0.031 s**.
- Observed selection plus 99 null trials: **1.083 s**.
- Maximum CPU/GPU log-response difference: **5.56e-16**.
- Paid calls and API spend: **0**.
- Full receipt replay: **PASS**.

## Failure-space update

The result rejects promotion of the bounded law above over the admitted parameter domain
when it is universal, distance-only, no-slip, positive from 1 AU to 10 Mpc, and evaluated
with this spherical Hernquist/virial SLACS forward model. It does not reject:

- a running law derived from curvature, momentum, density, or environment rather than
  radius alone;
- an action-derived flow with multiple coupled operators;
- independently derived gravitational slip;
- resolved orbital/Jeans models or direct strong-lens image likelihoods;
- antiscreening driven by field self-interaction, which is Item 18;
- a viable law outside the exact bounded logarithmic family.

## Next action

Advance to Item 18 gravitational antiscreening on a fresh real response. Preserve this
exact running family as a known negative comparator and do not open the 12 SLACS
confirmations.
