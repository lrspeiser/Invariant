# The channel-placement deficit is mainly a detection/background effect

The executed ablation locates the large recovery loss within the conditional
pipeline. In the representative symmetric-rotation, amplitude-five case,
switching only the detection threshold from per-channel noise to a fixed global
noise scale reduces the channel-20 deficit from **32.67 to 15.11 percentage
points**. Removing continuum subtraction from the injected source changes no
true-flux recovery values. With no background and a fixed threshold, the
channel-placement deficit vanishes to numerical precision.

This is a controlled statement about injected signals on these particular
observed backgrounds. It is not a diagnosis of the telescope's underlying
contamination, proof that a constant threshold is preferable, or a gravity result.

## Frozen comparisons

[PREFLIGHT.md](PREFLIGHT.md) was frozen before new array access. All earlier
rotation, kinematic-twist and streaming templates, channel centers 10/20/30,
amplitudes 5/10, and three spectral-response brackets were retained. Each was
tested with source continuum subtraction on/off and western per-channel/global
median MAD. The strict two-sigma, three-consecutive-channel mask rule stayed fixed.

Executed **2,592 actual-background injections and 216 noiseless controls**.
The original **702 empirical/noiseless trials replay exactly**, with zero
difference in the three saved flux-fraction metrics. No source parameter or
threshold was tuned after seeing the eastern background results.

The real input remains NGC2976's public standard THINGS native cube and the
previously fixed western calibration/eastern patches. Exact source and code
hashes are in [pre-access-bindings.json](run001/pre-access-bindings.json).
The native-selection source receipt binds the
[THINGS archive](https://things.www3.mpia.de/Data.html),
[Walter et al. 2008](https://arxiv.org/html/0810.2125), and contemporary instrument
documentation. No new downloads, source velocity observations or reserved
galaxies were accessed.

The continuum-off experiment affects **only injected source emission**.
Actual backgrounds retain their delivered continuum subtraction, covariance,
possible emission and calibration artifacts. Missing parent channels prevent
undoing the original background reduction. Saved A and H matrices are the
history-constrained image-domain continuum surrogate and spectral brackets,
not a recovered exact visibility-domain instrument response.

## What changes recovery

For symmetric rotation with amplitude five and the independent spectral bracket:

| Stored channel center | Actual background, channel-specific threshold | Actual background, constant threshold | No background, channel-specific threshold | No background, constant threshold |
|---|---:|---:|---:|---:|
|10|77.11%|76.92%|81.49%|80.94%|
|20|45.30%|63.15%|65.88%|80.94%|
|30|78.83%|79.59%|80.68%|80.94%|

Entries are retained **true injected flux**, not total noisy flux. Even the
no-background constant-threshold mask retains only 80.94% of this finite signal:
faint line/spatial wings do not all meet the run threshold.

Define the placement gap as channel 20 minus the mean of channels 10 and 30,
paired within the same background position. The original gap is -32.67 points.
Fixing the threshold removes 17.57 points, approximately 53.8% of this particular
gap. The remaining -15.11 points disappear when the delivered background is
replaced with zero. The two contributions interact through a nonlinear mask;
53.8% is a conditional descriptive fraction, not a universal causal allocation.

At channel 20, fixing the threshold improves mean recovery by 17.85 points
(paired SD 10.90 points; range 2.53 to 36.67 across twelve positions). The
remaining fixed-threshold placement gap has SD 21.32 points and ranges from
-52.90 to +23.59. Some individual patches reverse the mean ordering. Positions
overlap, are correlated, and were already development-exposed; these are
descriptive ranges, not iid confidence intervals or population uncertainties.

The same conclusion holds across the retained spectral brackets:

| Spectral bracket | Original placement gap | Constant-threshold placement gap |
|---|---:|---:|
|Independent cells|-32.67 points|-15.11 points|
|Full Hanning|-32.35 points|-14.98 points|
|Alternate-channel Hanning|-32.69 points|-15.17 points|

Each bracket retains its prior symmetric-peak amplitude convention, so the
cross-branch numbers include that normalization and are not a pure comparison
at identical intrinsic amplitude. Within a branch, the factorial source and
background are identical. [paired-effects.csv](run001/paired-effects.csv)
retains every morphology, amplitude, center, continuum and threshold effect.

## Where flux does and does not disappear

[stage-ledger.csv](run001/stage-ledger.csv) separately records the intrinsic
spectral-cell integral, spectral mixing, stored-channel restriction, continuum
subtraction, native restoring beam and detection smoothing. Spectral pre-grid
sums use their actual cell width; later channel sums use stored-channel units.
They are unnormalized brightness sums, not inferred galaxy masses.

- Spectral mixing preserves integrated input within 2.23e-16 relative error.
- Restriction to delivered channels loses at most 1.57e-12 of these templates.
- Signed source-continuum attenuation is at most 4.38e-9 of source flux.
- Native-beam cropped-aperture loss is at most 2.21e-6.
- Additional detector smoothing loses at most 0.000762, or 0.0762%, from the
  finite aperture. This spatial loss is unrelated to spectral placement.

The enormous recovery differences therefore arise at selection rather than
from disappearance of modeled integrated signal at these earlier stages.
This does not establish negligible continuum loss for broader sources, signals
near fitting channels, or different line profiles, none of which were tested.

[channel-ledger.csv](run001/channel-ledger.csv) preserves each channel's stage
flux, detector peak, western MAD and global scale.
[linear-operators.json](run001/linear-operators.json) saves A, S, each H and
their conditional spectral covariance matrices. The hypothetical S covariance
is recorded for interpretation only; it was not substituted into actual noise.

## Validation and remaining limits

Before patches, all inherited independent numerical controls and new exact
channel-selection, direct spectral convolution, full spatial-flux conservation
and operator-commutation checks passed. Maximum new convolution/commutation
errors are below 2.23e-16. A separate same-author direct spectral filter and
polynomial least-squares implementation reproduces the stage ledger; channel
profiles sum back to their totals; all paired-effect tables were replayed.
The receipt [separate-bookkeeping-review.json](separate-bookkeeping-review.json)
distinguishes this from an external review or an independent source template.

The real noise/background is not independently certified line-free. This
experiment cannot attribute the remaining effect to Galactic emission,
instrumental artifacts, nonstationary covariance, or another physical origin.
Nor does it measure false alarms for the diagnostic constant threshold; lowering
thresholds in unusually noisy channels can increase contamination.

**SOURCE_BLOCKED remains for observational gravity likelihoods.** The concrete
advance is an auditable location of the conditional bias: channel-specific
thresholds plus delivered-background interaction, rather than modeled source
continuum attenuation or integrated spectral flux loss. Next source-likelihood
work should constrain those background/instrument components and include
selection in the forward model. No single flux correction is justified here.

## Reproduce

With the existing bound private cube/calibration cache and NumPy, SciPy,
Astropy and threadpoolctl, from repository root:

```
python -B scripts/mond_atlas_selection_ablation.py
python -B scripts/mond_atlas_selection_ablation_review.py
```

The immutable runner refuses an existing run001 directory; reproduce in a
separate checkout/output copy. No new private arrays, GPU use, shared frozen
edits, commits or publication were performed by this task.
