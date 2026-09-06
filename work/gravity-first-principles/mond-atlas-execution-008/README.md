# MOND atlas: consistent source maps, 3D force checks and noise robustness

The image and gravity calculations now use the same mathematical description
of the source. The revised stellar light fits have **1.21% and 4.80% image RMS
mismatch** for two different assumed depth arrangements. These are conditional
reconstructions, not measured depths or a noise-calibrated posterior.

A useful conditional pattern is that the two source models have quite similar
mean rotational force but much less similar vertical force. At 5 kpc radius,
their QUMOND force-equivalent speeds differ by about **1.7%**, while their mean
downward force 0.25 kpc above the disk differs by about **32%**. Newtonian gravity
shows the same qualitative sensitivity. This is a model comparison, not an
observed force difference or evidence that one gravity law is correct.

The broader background check also exposes a practical limit: **9/12 galaxies
pass every declared split; NGC2841, NGC2903 and NGC3198 do not.** A favorable
single partition had understated that uncertainty. No new galaxy motion
comparison was performed in this phase, and the atlas remains unfinished.

## The source correction

The earlier image inverse used constant brightness inside each map pixel,
whereas the gravity code interpolated between pixel centers. Those are different
light distributions. For the thin stellar case, its reported 0.41% image error
became 5.04% when projecting the distribution actually used by the gravity code.

The new inverse integrates the same bilinear basis used by the field loader.
Independent line-of-sight and analytic finite-pixel tests check that operator.
The source fit, support, coverage weights and regularization are unchanged.

| Source | Earlier constant-pixel fit | Earlier values under gravity basis | Refit common basis |
|---|---:|---:|---:|
| stars_thin | 0.41% | 5.04% | 1.21% |
| stars_mixed | 3.63% | 6.66% | 4.80% |
| atomic | 0.09% | 1.24% | 0.21% |
| co21 | 3.56% | 5.17% | 3.81% |

The stellar alternatives are a single 0.1 kpc exponential layer and a mixture
with 25% of its light in a 0.1 kpc layer and 75% in a 0.4 kpc layer. Gas layers
remain 0.2 kpc. The 5% image diagnostic is only a gross mismatch flag: its weights
describe coverage, not the actual source-noise covariance. The alternatives
also differ in recovered planar structure, so their gravity difference cannot
be attributed to thickness alone.

The catalog now has an explicit [asset-role overlay](asset-role-overlay.csv):
five files historically named `STELLAR_MASS_MAP` contain cleaned **flux in
MJy/sr**, requiring a separate mass-to-light conversion. This follows the
[publisher's P5 description](https://irsa.ipac.caltech.edu/data/SPITZER/S4G/docs/P5_README.html).
Original receipt names and hashes remain intact.

## Numerical checks for both revised sources

A disk-backed implementation reproduces the previous dense calculation to
roundoff. It applies global transforms in smaller groups; it does not split
the galaxy into independently gravitating chunks. The previous mixed-source
failure was resolved by another lateral refinement: 1.06% Newtonian and 0.95%
QUMOND vector RMS difference, below the unchanged 3% aggregate and 5% ring gates.
That success applied to the old coefficients, so both new source fits were
tested separately.

For each new model, the base grid is 0.125 kpc in all directions in a box with
24 kpc half-width. Separate perturbations halve horizontal spacing, halve
vertical spacing, or enlarge the box half-width to 32 kpc. Each is compared
with the same base, over 2–15 kpc. Their joint-refinement cross terms are not bounded.

| Model / check | Newtonian vector RMS | QUMOND vector RMS | 3% aggregate and 5% ring gates |
|---|---:|---:|---|
| common_thin / lateral | 1.378% | 1.245% | pass |
| common_thin / vertical | 0.216% | 0.228% | pass |
| common_thin / box | 0.004% | 0.192% | pass |
| common_mixed / lateral | 1.366% | 1.221% | pass |
| common_mixed / vertical | 0.124% | 0.135% | pass |
| common_mixed / box | 0.004% | 0.187% | pass |

The potentials solve Newtonian gravity and [QUMOND](https://arxiv.org/abs/0911.5464)
from combined baryons, with the same fixed conversion factors and isolated
boundary assumptions. Box convergence does not establish physical isolation.
This is not an AQUAL calculation or an observational motion likelihood.

## The check now includes forces above the disk

The additional audit samples all three force components at 0.25, 0.5 and 1 kpc
above the plane, 14 radii and 72 azimuths. Mirrored points test the vertical
reflection symmetry of the declared source. Quadratic potentials with cross
terms provide an independent exact force-sampling benchmark.

| Model / check / law | Full-vector RMS | Worst radius-height group | Vertical component RMS | Full-vector gates |
|---|---:|---:|---:|---|
| common_thin / lateral / newton | 0.212% | 0.438% | 0.191% | pass |
| common_thin / lateral / mond | 0.189% | 0.409% | 0.175% | pass |
| common_thin / vertical / newton | 0.594% | 1.112% | 0.833% | pass |
| common_thin / vertical / mond | 0.509% | 1.008% | 0.748% | pass |
| common_thin / box / newton | 0.004% | 0.045% | 0.001% | pass |
| common_thin / box / mond | 0.162% | 0.804% | 0.033% | pass |
| common_mixed / lateral / newton | 0.388% | 0.738% | 0.221% | pass |
| common_mixed / lateral / mond | 0.344% | 0.719% | 0.202% | pass |
| common_mixed / vertical / newton | 0.416% | 0.819% | 0.658% | pass |
| common_mixed / vertical / mond | 0.375% | 0.796% | 0.622% | pass |
| common_mixed / box / newton | 0.004% | 0.042% | 0.001% | pass |
| common_mixed / box / mond | 0.168% | 0.773% | 0.039% | pass |

Full-vector gates retain 3% aggregate and 5% per radius-height group. The
vertical-only column has its own denominator and remains a separate diagnostic:
passing a total-vector gate does not mean every component has that accuracy.
Only the stated sample positions and perturbations are certified by these checks.

## What the alternative source arrangements change

These are conditional field comparisons, not observations of a galaxy changing
shape. The following uses the horizontally refined solutions; all numerical
flags above still apply. A force-equivalent speed summarizes mean inward force
as sqrt(radius × mean inward acceleration); it is not an actual circular orbit
in a barred potential.

| Law | Radius (kpc) | Mixed-versus-thin speed change | Thin sideways RMS / mean inward force | Mixed sideways RMS / mean inward force |
|---|---:|---:|---:|---:|
| newton | 2 | -4.26% | 24.74% | 38.29% |
| mond | 2 | -3.31% | 24.42% | 37.60% |
| newton | 5 | -2.32% | 11.39% | 12.40% |
| mond | 5 | -1.68% | 11.08% | 12.06% |
| newton | 10 | -0.91% | 6.32% | 5.73% |
| mond | 10 | -0.66% | 5.77% | 5.24% |

The model totals are 50.43 and 50.18 billion solar masses, only a 0.50% change.
Their distributions differ in depth and in the reconstructed planar structure.
At radius 5 kpc, the mixed source predicts less downward force than the thin
source, with the difference decreasing farther above the disk:

| Law | Height above disk (kpc) | Mixed-versus-thin mean downward force change |
|---|---:|---:|
| newton | 0.25 | -34.06% |
| newton | 0.5 | -20.03% |
| newton | 1 | -6.87% |
| mond | 0.25 | -32.38% |
| mond | 0.5 | -18.59% |
| mond | 1 | -6.29% |

This supplies a concrete direction for future observations: independently
constrained vertical structure or vertical motions may separate models whose
mean rotation predictions differ much less. Converting gas thickness into
gravity already requires a pressure/equilibrium model; it cannot be treated as
a direct, gravity-independent force measurement. The two source alternatives
also fit the image with different errors, so these percentages are not an
observational confidence interval on the galaxy's gravity.

The complete [midplane table](midplane-source-sensitivity.csv) and
[above-plane table](../mond-atlas-offplane-001/conditional-source-sensitivity.csv)
retain signed changes and component values. Distinct depth alternatives can
therefore be tested through directional and vertical information, but these two
source fits are not a calibrated ensemble and do not establish a measured effect.

## Background noise depends on the calibration region

Eight frozen geometry-only partitions were evaluated in both directions,
producing 16 unique splits per galaxy and **192 completed checks**. Spatial
guards, the background annulus, covariance form and diagnostic limits were
unchanged. Galaxy-region values were zeroed before evaluation. We did not select
a winning split. These splits reuse the same background realization; their
failure fractions are neither independent probabilities nor p-values.

| Galaxy | Passing splits | Residual adjacent-channel product range | Failed diagnostic(s) |
|---|---:|---:|---|
| DDO154 | 16/16 | -0.062 to 0.095 | none |
| IC2574 | 16/16 | -0.119 to 0.081 | none |
| NGC2841 | 9/16 | -0.097 to 0.311 | held_channel_lag1 |
| NGC2903 | 14/16 | -0.082 to 0.218 | held_channel_lag1 |
| NGC2976 | 16/16 | -0.021 to 0.068 | none |
| NGC3198 | 10/16 | -0.099 to 0.317 | held_channel_lag1;spatial_quadrants |
| NGC3521 | 16/16 | -0.018 to 0.052 | none |
| NGC4214 | 16/16 | -0.058 to 0.039 | none |
| NGC5055 | 16/16 | -0.102 to 0.103 | none |
| NGC6946 | 16/16 | -0.111 to 0.066 | none |
| NGC7331 | 16/16 | -0.107 to 0.056 | none |
| UGC04305 | 16/16 | -0.103 to 0.047 | none |

The residual adjacent-channel diagnostic is the mean product of neighboring
whitened channels, with an absolute limit of 0.15; it is not a normalized Pearson
coefficient. All global whitened mean-square values pass the existing 0.5–2.0
range. Some NGC3198 quadrant checks also fail. The cause could include covariance
form, nonstationary noise, residual emission or finite calibration samples.
This audit alone does not distinguish them. In particular, NGC2903's two failed
splits prevent treating its earlier single-split pass as robust noise validation.

## Actual atlas coverage and remaining work

- **13,525 object groups** in the identity overlay, with unresolved associations;
  these are not certified distinct galaxies or complete 3D models.
- **175 radial baselines**, with 126 meeting the fixed descriptive cuts. Those
  earlier algebraic radial calculations are not full-field disk solutions.
- **12 resolved seed galaxies / 137 original assets**. Eight pass both the raw
  image astrometry and every current background split; this is only a pair of
  prerequisites. See [pilot readiness](pilot-readiness.csv).
- **22 source-image fits and 29 conditional full-field runs**, including replay
  and convergence runs, still covering only **one field galaxy**.
- **57 passing unit tests; zero admitted full-field galaxy cube likelihoods.**

The next scientific requirements are source covariance and native-pixel/beam
projection, absolute photometry, recovery of the missing raw geometry tables,
independent mass/depth/environment constraints, missing baryonic phases, AQUAL
controls, and a validated motion cube model including pressure and noncircular
motions. The remaining pilots and resolved-sample expansion have not been run.

All new source and numerical packages prospectively declare `SOURCE_BLOCKED`.
That describes admission for observational scoring, not a failure to execute
these diagnostics. The earlier exploratory rotation comparison remains
nonadmitted under the [repository policy](../../../docs/OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md).

## Reproducibility and publication

The [verification record](verification.json), [test log](validation.log),
[field integrity](field-integrity.csv), [status](execution-status.json), and
[input bindings](input-bindings.json) distinguish completed work from remaining
requirements. The [publication manifest](publication-manifest.json) lists only
intended code, configuration and compact results; all raw observations and
large numerical fields stay outside Git.

Ordinary `git fetch origin main` failed because the linked worktree's Git
metadata is outside the writable workspace. The connected GitHub integration
can read the repository, and its remote comparison verified `main` still at
`afc721a1`. Its first blob-creation call was then rejected because the tool
requires approval and this session's approval policy is `never`. No remote
blob SHA, commit or ref update was returned; nothing was published. The
manifest and exact-byte transfer helper are retained for ordinary publication
when permitted. Downloads and the previous CUDA environment remain unavailable.
Computations here use the working bundled CPU runtime.

To replay, choose unused output paths and preserve the named source bindings:

```text
python scripts/run_mond_atlas_common_basis_fields.py --output NEW_FIELD_DIR --private NEW_PRIVATE_DIR
python scripts/run_mond_atlas_noise_robustness.py --output NEW_NOISE_DIR
python scripts/run_mond_atlas_offplane.py --output NEW_OFFPLANE_DIR
python -m unittest discover -s tests -p "test_mond_atlas*.py" -v
```

The off-plane protocol binds the original field directory. To inspect a newly
replayed field set, create a new protocol copy with its explicit path rather
than editing a frozen result.
