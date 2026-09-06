# MOND atlas: selection controls and readiness for machine learning

We have enough data to develop a controlled search for relationships between
observable baryonic structure and motion. We do not yet have enough to identify
the true gravity contribution of every individual component, reconstruct unique
3D mass distributions, or validate a joint motion-and-lensing theory.

The current limitation is reliable inputs and independent tests, rather than
the raw number of pixels or GPU memory. The atlas goal remains active.

| Resource | Verified position | What it supports |
|---|---|---|
| Rotation curves | 175 galaxies; 126 pass the existing descriptive cuts | Broad exploratory comparisons; not general 3D reconstruction |
| Resolved pilot data | All 137 cataloged assets present; 12 HI cubes, stellar images and CO products | Source and instrument development; coverage/detection/conversion still require checks |
| Larger catalog | 13,525 identity groups, not certified distinct galaxies | Coverage and sample expansion; not 13,525 resolved gravity tests |
| Full-field calculations | 29 conditional runs for one galaxy; zero admitted cube likelihoods | Numerical and geometric sensitivity tests |
| Lensing | No lensing assets or source records in the audited atlas catalog | A matched sample must be acquired and validated |
| GPU | RTX 5090, 32,607 MiB, driver 580.88 detected | Hardware available; CUDA ML execution not yet demonstrated in the active runtime |

The [runtime and asset audit](../mond-atlas-ml-readiness-001/readiness.json)
checks the active Python 3.12.14 interpreter. NumPy is present; torch, CuPy, JAX,
SciPy, scikit-learn, XGBoost and LightGBM are not. Other machine environments are
not certified absent by this check. A broader home-directory listing was denied.
No CUDA kernel or ML training was run. Historical lensing readiness is synthetic
and has no real source packets; that historical configuration is evidence about
past execution, not a restriction on the user's current authorization.

Five catalog assets historically labeled STELLAR_MASS_MAP are cleaned stellar
flux, as corrected by execution-008. Those raw role labels in the audit are not
independent mass measurements. HI likewise is only one baryonic component.

## What the next ML experiment should learn

Divide each galaxy into spatially resolved regions, with distributions for
stellar mass, atomic gas, molecular gas and uncertain depth. Use independent
light and gas measurements for those inputs. Infer nuisance motion components
through a forward cube model: ordinary rotation, warp geometry, asymmetric
structure, streaming, pressure support and the instrument's response.

Newtonian acceleration can be decomposed into vector contributions of source
regions. In [QUMOND](https://arxiv.org/abs/0911.5464), first construct the
Newtonian field of the combined mass, then apply the nonlinear field equations.
Separately boosting each region and adding the results does not generally solve
those equations. Changing one region and recomputing the whole field provides
a conditional sensitivity, not a unique observed allocation of gravity.

Fit a modest model to errors in predicted observables, using features such as
stellar concentration, gas fraction, resolved clumping, asymmetry, plausible
thickness, warp strength and neighboring matter. Measure whether each feature
adds prediction beyond acceleration, size, mass and observational quality.
Color can constrain stellar populations; it is not a direct age or mass label.
Motion-derived mass cannot simultaneously serve as independent input mass and
the response used to prove a new gravity law.

Keep entire galaxies, physical groups and eventually surveys out of training.
Split before feature tuning; use nested training validation and galaxy-level
uncertainty. Retain previous development exposure. Nearby pixels, augmented
images and synthetic versions of one galaxy are not independent galaxies.
Test stability across plausible mass conversions, distances, inclinations and
source-depth ensembles. Explicitly test whether noise, masks, resolution or
survey identity alone predict the same effect. Start with interpretable
regression or small trees; GPU emulators and image models are useful after
the forward simulator and labels are validated. Distill reproducible effects
into dimensionally consistent candidate formulas and test untouched systems.
These are next experiments, not results of a completed ML training run.

## Lensing adds another observable, not a 3D gravity meter

A deflected light ray accumulates effects along its path. An observed arc or
shear pattern therefore constrains a projected gravitational field, not the
acceleration vector at a unique point along that path. Different depth
arrangements can produce similar observations. Lensing and motion together can
reduce this ambiguity, with source/lens distances, stellar populations and
foreground/background structure also constrained.

A nonrelativistic MOND acceleration prescription does not by itself specify
light propagation. An explicit metric theory or declared relativistic closure
is needed. [Bekenstein's relativistic MOND paper](https://arxiv.org/abs/astro-ph/0403694)
is a concrete example of connecting particle dynamics and deflection through
the theory's fields; citing it does not validate our models or establish that
that historical theory passes every modern constraint.

Use observed image pixels, multiple-image positions, redshifts and calibrated
shape measurements as responses. A published halo-fitted mass map or a
simulation's assigned lens mass is not an independent baryonic mass label.
Our experimental gravity branches can exclude dark-matter components while
retaining this distinction between measurements and model-dependent inference.

[Euclid Q1](https://www.euclid-ec.org/science/q1/) provides public imaging,
spectroscopy, catalogs, masks and lens-candidate work. It offers an expansion
source, not a locally acquired or matched sample. Confirmed lenses need
source/lens redshifts, PSF/noise, selection characterization and independent
stellar constraints. No match with the twelve nearby HI pilots is established.

## New executed tests: stationary noise does not explain everything

[Smoothing controls](../mond-atlas-smoothing-null-001/summary.json) use the eight
galaxies retaining historical continuum-fit candidates, with actual dimensions,
the verified smoothing/block operator, two spatial noise surrogates and two
spectral surrogates. Each of 32 cases has 256 simulated galaxy statistics:
8,192 total. Continuum subtraction is propagated through A C A-transpose.

The observed geometric mean of center/outer robust noise ratios is 1.0853.
In each of the four surrogate branches, only zero or one of 256 joint reference
draws reaches that statistic. NGC2903's ratio 1.162 exceeds every reference draw
in all four branches; NGC4214 also stands out. These are exploratory conditional
reference comparisons on exposed development data, not observational p-values.
Actual interferometer noise, faint emission and residual continuum are not
uniquely distinguished. No new gravity or motion fit was scored.

## New executed tests: the mask can create strong patterns

[Injection controls](../mond-atlas-mask-injection-001/summary.json) executed
384 noise cubes, 108 source/instrument cases, 13,824 paired injection trials
and 108 noiseless controls. The fixed operator requires three consecutive
channels above twice the known marginal noise scale. Signal and noise both
receive their declared spectral/spatial response. The experiment isolates
that operator; it does not reconstruct the full published THINGS master mask.

| Spectral noise surrogate | Mean fraction of noise voxels selected | Relative to independent |
|---|---:|---:|
| Independent channels | 0.00310% | 1.0 |
| Three-tap Hanning, all channels retained | 0.26653% | 85.8 |
| Three-tap Hanning, alternate channels retained | 0.01506% | 4.85 |

This is a false-selected voxel fraction, not the probability that a detected
astronomical object is false. The marginal noise variance is approximately one
in all branches; correlation causes the large change in the rule's behavior.

For an intrinsic 90-arcsec spatial Gaussian with peak five times the noise
scale, independent-channel trials select the peak about 0.8%, 95.3% and 100%
of the time for intrinsic spectral widths of one, three and six channels.
Their mean retained true-source flux fractions are 0.19%, 25.2% and 31.1%.
Even without noise, the narrow independent-channel source is excluded because
it never supplies three qualifying channels. Broad Gaussian outskirts fall
below threshold. Correlation and spectral broadening change these outcomes.
This synthetic selection effect is not a measured error in a galaxy's gas mass.

Selected noisy flux also includes positive noise, and is reported separately
from retained known source flux. That statistic sums over the whole controlled
cube; its bias can be large for weak compact sources. Neither metric can be
silently substituted for the other or generalized to the published survey mask.

## Verification and remaining work

All 85 atlas unit tests pass. All 501 files in the preceding milestone manifest
were rehashed without mismatches before updating the handoff. The 15 new
prospective experiment bindings also match. Fixed protocols and all surrogate
branches are retained; no threshold was tuned using galaxy gravity residuals.
These packages remain SOURCE_BLOCKED for observational scoring.

Next: identify usable native line-free support and composite spectral histories;
validate a real selection mask with injections through the actual response;
complete source uncertainties, geometry and baryonic conversions; validate
full-field motion predictions and additional pilots; then conduct held-out
structure tests and add a separately validated lensing tier. Synthetic patterns
are useful controls but cannot supply missing observed information.

Reproduce with new immutable output directories:

```text
python scripts/run_mond_atlas_smoothing_null.py --output <new-null-directory>
python scripts/run_mond_atlas_mask_injection.py --output <new-injection-directory>
python scripts/run_mond_atlas_ml_readiness.py --output <new-audit-directory>
python -m unittest discover -s tests -p "test_mond_atlas*.py" -v
```

Publication is local only. Linked Git metadata lies outside the writable root.
The prior GitHub blob write was rejected because approval is required and this
session cannot grant it. No alternate write or successful push is claimed.
See [execution status](execution-status.json), [verification](verification.json)
and [publication manifest](publication-manifest.json).
