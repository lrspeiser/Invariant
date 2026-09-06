# Selection can distort motion-pattern comparisons

Executed a new conditional injection experiment on actual NGC2976 native HI
backgrounds. At the same integrated injected brightness, the streaming template
lost up to **7.49 percentage points more true source flux** than the symmetric
rotation template. Merely moving the symmetric template from stored channel 10
to 20 changed mean retained flux from **77.11% to45.30%** in one fixed case.
These are selection effects on known artificial signals, not measurements of
streaming, missing mass or unusual gravity in NGC2976.

The pipeline now has an executed transfer test across resolved synthetic motion
patterns, beyond the earlier separable Gaussian injections. **SOURCE_BLOCKED
remains: zero observed gravity likelihoods admitted and zero gravity scores.**

## Frozen experiment and sources

[PREFLIGHT.md](PREFLIGHT.md) froze three templates, three spectral branches,
channel centers 10/20/30, amplitudes 5/10, controls and diagnostic gates before
new array access. [run001/pre-access-bindings.json](run001/pre-access-bindings.json)
binds the implementation, sources, prior selection calibration and exact12 patch
locations. All11 bindings were reverified after execution and in the review.

The real source is the public natural-weighted standard THINGS cube
`NGC_2976_NA_CUBE_THINGS.FITS`, SHA256
`e8ce711a354fcf9c76e7ba9afd9e55fa97df6169863d5473140f20f9c8250169`.
Source paper: [Walter et al. 2008](https://arxiv.org/html/0810.2125).
Archive: [THINGS data](https://things.www3.mpia.de/Data.html).
Instrument evidence: [NRAO contemporary VLA manual](https://library.nrao.edu/public/pubs/obsstat/VLAOS_0302.pdf).
The existing native-selection package contains the acquisition/header audit.
This task downloaded nothing and wrote no private pixel arrays.

Actual data enter as background patches and previously fixed western channel
medians/MAD. The eastern injection locations are separated from calibration,
but are already development-exposed and partially overlap. Same-observation
MOM0 screening does not establish absence of emission. No observed source
rotation curve, fitted velocity field or reserved galaxy data were opened.

Each template has the same elliptical Gaussian surface brightness with major
FWHM 30 arcsec and projected axial ratio 0.5; local line width is 2 stored channels.
The rotation field has a smooth rising velocity, the warp template twists its
kinematic orientation with radius, and the streaming template adds a radial
velocity term. The warp does **not** change density geometry; it is a kinematic
twist test, not a dynamically self-consistent warped disk reconstruction.
All amplitudes use the symmetric template's detector peak, so differences are
not removed by independently renormalizing each template's peak. Integrated
reference-flux differences between morphologies are below 1.52e-12 fraction.
Five times the median channel MAD is not five times each channel's local noise.

All three spectral brackets are retained: independent pre-continuum channels,
full Hanning and alternate-channel Hanning. Both signal and covariance pass
through the history-bound continuum operator. The measured restoring beam
convolves signal spatially; the Gaussian noise branch retains the earlier
beam-filtered spatial surrogate. Actual backgrounds preserve actual channel
and spatial correlations, contamination and artifacts. Neither branch recovers
the exact instrumental passband, dirty beam or nonlinear CLEAN response.

## Executed results

There are 648 empirical trials,864 Gaussian trials,54 noiseless controls and 48
Gaussian noise realizations. Each group contains 54 parameter/template cases.

| Diagnostic | Actual backgrounds | Gaussian surrogate | Noiseless |
|---|---:|---:|---:|
| Adequate flux recovery |19/54|25/54|27/54|
| Morphology-transfer pairs passing |27/36|29/36|34/36|

Adequate recovery requires mean true retained flux >=90% and mean paired
selected-flux bias within 10%. Transfer requires absolute mean retained-flux
difference from symmetric rotation <=5 percentage points. These thresholds are
diagnostic, not scientific admission thresholds. All failing cases remain in
[case-summary.csv](run001/case-summary.csv) and
[paired-morphology.csv](run001/paired-morphology.csv).

For independent spectral cells and the symmetric template:

| Stored center | Amplitude | True flux retained | Paired selected flux / injected flux |
|---|---:|---:|---:|
|10|5|77.11%|93.95%|
|20|5|45.30%|54.93%|
|30|5|78.83%|95.76%|
|10|10|93.59%|104.42%|
|20|10|90.11%|96.56%|
|30|10|94.86%|109.10%|

The first row illustrates why apparently near-correct recovered total flux
does not establish a complete source mask: noisy selected flux and retained
true emission are different quantities. The parent native-selection analysis
already found elevated channel 20/21 MAD; this test demonstrates its conditional
effect on otherwise identical artificial templates, without assigning a cause.

At channel 10, amplitude 5 and independent spectral response, streaming minus
rotation retained flux is −7.49 percentage points across 12 actual patches
(SD 3.77 points; range −13.12 to −2.22). The Gaussian experiment gives −8.05 points
(SD 4.61 across 16 draws). Empirical ranges/SD are descriptive; overlapping,
correlated locations do not justify iid confidence intervals. Case tables give
Monte Carlo standard errors only for independent Gaussian realizations, which
exclude uncertainty in the assumed noise and instrument model.

## Checks and admission

Inherited independent convolution, beam covariance, spectral quadrature,
continuum least-squares, covariance Monte Carlo and spatial integration controls
passed before patch access. New zero-motion morphology identity and velocity
antisymmetry errors are zero; intrinsic flux invariance error 2.22e-16; the
separately looped run-mask matches exactly. A later same-author separate polar
velocity/CDF implementation agrees with new templates to 2.78e-15. Table replay
reproduces all 162 case gates and 108 paired gates to 4.44e-16.
[algebra-table-review.json](algebra-table-review.json) explicitly distinguishes
these checks from an independent external review or complete operator rewrite.

What is usable now: a reproducible conditional test of mask transfer between
known synthetic velocity patterns on observed backgrounds, with paired bias
and finite Monte Carlo uncertainty. It can falsify the assumption that one
mask recovers different source patterns equally well.

What remains unavailable: observational population completeness; independently
certified emission-free background; exact published channel mask/passband and
native covariance; residual scaling, primary beam, CLEAN and missing-spacing
flux likelihood; realistic galaxy-wide source geometry and nuisance uncertainty.
No universal flux correction, corrected baryonic mass or preference among
gravity laws follows from this experiment. A future observed comparison must
forward-model its selection across source/nuisance hypotheses and validate the
flux likelihood, rather than applying a single completeness multiplier.

## Reproduction

From repository root, with the existing bound private THINGS cube/cache and
Python NumPy/SciPy/Astropy/threadpoolctl environment:

```
python -B scripts/mond_atlas_selection_transfer.py
python -B scripts/mond_atlas_selection_transfer_review.py
```

The runner refuses an existing run001 directory. Preserve the published output;
use a separate checkout/output copy for reproduction. All old packages remain
unchanged. No GPU, new raw files, commits or publication performed by this task.
