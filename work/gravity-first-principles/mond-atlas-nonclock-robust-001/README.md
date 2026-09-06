# Coverage correction and broader tests of the other ideas

The earlier work did not robustly test every proposed mechanism. The user was
right to challenge that description. The clock branch received more structural
changes, and parameter-grid size was an inadequate description of breadth.
The [25-mechanism audit](../mond-atlas-coverage-audit-001/README.md) records the
actual coverage, missing observations and required experiments. It is an audit
of preceding work; the additional runs described here followed that audit.

## What this round added

The confirmed radial branches now have continuous global parameter fitting with
three deterministic starting points, wider bounds, and additional structural
choices. All 663 optimizer starts converged, but local convergence does not
guarantee a global optimum. Six pre-response unit/mechanics tests passed. An
independent review reproduced 204 selected fits and 106,176 held predictions.

The source remains the same 102 eligible galaxies and 2212 SPARC radii from the
139 historically exposed identities. No reserved archive member bodies were
parsed. We used three five-fold whole-galaxy splits and separate transfer tests:
train on 44 gas-rich galaxies and predict 58 stellar-rich galaxies, then reverse.
Global parameters use training velocities; individual source inputs contain no
fitted halo mass. Both branches share the same continuous MOND and stellar-mass
control bounds. These are post-hoc development comparisons, not confirmation.

| Branch | Added changes | Result versus continuously fitted MOND |
|---|---|---|
| Surface-density coherence | Continuous strength and density scale; exponent fixed at 1 versus learned in 0.25–4 | Learned exponent improves its own MSE 1.94%, but remains 23.53% worse than MOND. Transfers poorly between gas-rich and stellar-rich groups. |
| Absorption/re-emission | Continuous opacity and re-emission fraction; passive versus active gain; wider common stellar mass factor | Both remain roughly four times MOND MSE. Active gain fails particularly badly in gas-to-stellar transfer. |
| Finite p2/p3 mixture | Continuous core mixture, amplitude, length and interpolation between disk-size and mass-derived scales | Apparent 0.50% MSE gain is fragile and changes sign across splits. Gas-to-stellar transfer is 21% worse than MOND. |
| Truncated point kernel | Continuous amplitude, length rule and cutoff | 5.94% worse MSE. Amplitude hits its upper bound; sampled radii do not constrain the cutoff. |
| Cored finite transition | Separately adjustable central scale, amplitude and outer transition | Apparent 4.51% lower MSE, but its gain disappears when one galaxy's score contribution is omitted. This overlaps the cored clock force class; it is not a new physical mechanism. |

## The consequential robustness finding

The finite transition's apparent 4.51% gain is dominated by UGC07577. That galaxy
contributes 103.92% of the net improvement. Omitting its contribution from the
completed error summary changes the gain to -0.194%. This diagnostic does not
remove the galaxy from training or rerun the fits; the original cohort and all
predictions remain intact. The mixture's tiny gain also changes sign under
single-galaxy omission. Neither gain is a robust winner.

Descriptive galaxy bootstrap intervals include zero gain for both models. These
intervals resample existing scores without refitting and omit model-search,
survey, source and covariance uncertainty. They are not calibrated discovery
significance. The independently registered diagnostic and every galaxy's
contribution are in [the review](review/diagnostics.json) and
[the influence table](review/galaxy-influence.csv).

Population transfer also limits the interpretation. For the flexible mixture,
ordinary split error is 0.10240dex against MOND 0.10265, but gas-rich training to
stellar-rich testing gives 0.10305 against 0.09359. The finite transition transfers
better, but its cutoff hits the maximum bound in every ordinary fold. The data
does not determine a finite reservoir merely because the formula has one.

The truncated kernel's measured radii lie below 6.4% of its fitted cutoff in the
selected solutions. Its cutoff is therefore unidentifiable in this test. This is
a concrete data limitation, not evidence for a measured outer reflecting shell.

## Formulas and interpretation

Surface coherence uses
`g=gb*[1+A/(1+(Sigma/Sigma0)^n)]`.
Sigma is the stellar disk surface-density proxy; it is not total volume density,
line-of-sight opacity, or measured coherence. The preferred exponent averaged
about 0.57. A smoother transition helps modestly, but source-domain transfer shows
that this local proxy is insufficient for a universal prediction.

Re-emission uses
`g=gb*[exp(-tau)+eta*(1-exp(-tau))]`, `tau=k Sigma/100`.
Passive eta<=1 has no enhancement over Newton for fixed mass. Active eta>1 can
increase the phenomenological field but has no derived energy supply here. A
local high-density interaction law tends to put enhancement in the wrong places
for the outer residuals; this is a failure of this formula, not all transport laws.

The finite mixture lets `L=lambda Rd^(1-t) rM^t`, `rM=sqrt(GM/a0)`, and
`gextra=A GM/L^2*[(1-q)/(1+r/L)^2+q*(r/L)/(1+r/L)^3]`.
Its high fitted q favors a softer central response. The extended point kernel
uses that length rule and a fitted cutoff in its cumulative NFW-shaped source.
Neither computes an actual convolution over observed three-dimensional sources.

The finite transition control uses
`gextra=eta sqrt(GMa0)*r/[(r+delta Rd)^2*(1+r/(C rM))]`.
It belongs to the same rational cored-potential class as the repaired clock
formula, with more independent central/outer parameters. Calling it a return
formula rather than a clock formula cannot identify the mechanism from identical
static forces. Its energy dynamics and photon metric remain unspecified.

## What is still not completed

This corrects part of the imbalance, not the complete research program. Actual
distributed source-cell fields, angular reflection and repeated backscatter,
nested arrangements, external-source geometry, spin/current coupling and memory
require distinct spatial or dynamical models and matching observations. They
cannot all be tested by a scalar one-dimensional proxy. Full baryonic uncertainty,
correlated noise, noncircular motion, untouched confirmation, cluster/Solar System
transfer and lensing remain unfinished. Missing tests are not negative findings.

The next fair comparison should finish those source and operator requirements,
not count additional parameter settings as additional physical ideas. Within
the radial families, source systematics for influential objects and held-survey
transfer are more informative than selecting another small development gain.
No galaxy is removed or downweighted here because it influences the result.

## Reproducibility

- [Coherence and re-emission](coherence/README.md): all starts, parameters,
  predictions, domain transfers and independently replayed selected losses.
- [Finite return](return/REPORT.md): all alternative shapes, limits, cutoff
  identifiability, starts and domain tests.
- [Independent cross-branch review](review/): distinct forward equations,
  all-start checks, selected-fit replay and post-hoc influence diagnostics.

Six tests were rerun by the coordinator after the branch runs. All earlier
published artifacts remain unchanged. A target-free quadrature repair around
the sharp kernel cutoff and a review JSON-serialization failure are preserved.
They were numerical/reporting fixes, not changes selected to improve observed
scores. Raw source archives remain outside Git; code, receipts and derived
predictions are published. Overall research goal remains unfinished.
