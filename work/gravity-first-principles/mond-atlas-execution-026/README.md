# From spatial predictions toward reliable observed-motion tests

This increment advances the uncertainty and external-field dependencies of the
broader program. The previous turn was concrete progress: execution025 was
published and its 1,560 bound files have been reverified unchanged. This turn
adds new executed experiments; the full research goal remains unfinished.

## Noise: an average passing score can hide a failing model

A positive stationary spatial-kernel mixture predicts all six aperture powers
much better than the previous unrestricted spatial covariance. Nevertheless,
its full joint q/N is 0.401, outside the predeclared 0.8–1.2 descriptive range.
An independent replay confirms this is a model limitation, not a scoring bug.

The next predeclared model allows different channel correlations at different
spatial scales, using an orthonormal cosine basis and four frequency bands.
Western-only selection gives aggregate joint q/N=0.993 and passing aperture
checks. Detailed diagnostics reveal cancellation: the lower half of its highest
frequency band has normalized power 1.960, while the upper half has only 0.00237.
The average conceals large opposing errors. It is not admitted as a calibrated
joint likelihood, and this result does not establish the physical cause of the
noise spectrum. No tuning followed the eastern results.

These tests use the same 29 western and 27 eastern observed background cores,
which are historically exposed development data. Spatial-mode powers describe
mathematical combinations of image pixels, not gravity or stellar speeds.

## Selection: motion changes how much gas the pipeline retains

We injected equal-brightness rotating, kinematically twisted and streaming sources
into the actual background observation, with fixed beam/spectral response
alternatives. This gives 648 real-background trials, 864 correlated-Gaussian
trials and 54 noiseless controls. The twist changes the velocity field while
holding brightness geometry fixed; it is not a self-consistent warped disk.

Streaming reduces peak brightness per channel even when the intrinsic integrated
brightness stays the same. The selection mask then loses up to 7.49 percentage
points more true flux than for symmetric rotation in the empirical trials.
Changing spectral placement also matters: one five-sigma rotation template
retains about 77.11%, 45.30% and 78.83% at the three tested channel centers.
Only 19 of 54 real-background cases pass the frozen recovery criterion.

This is a concrete measurement bias that can imitate or obscure a relation
between gas structure and inferred gravity. It does not show that real streaming
changes gravity. A motion likelihood must forward-model the source, instrument
and selection together; detected gas alone is not automatically an unbiased mass
map. The real patches overlap and were previously screened, so these counts are
conditional recovery tests, not population completeness estimates.

## External fields: separate relative acceleration from bulk motion

Sixteen new full spatial solves apply controlled external boundary fields to the
existing conditional density-response model. After removing the center's common
acceleration, major-axis fields retain RMS differential response about 0.115–0.131
times the applied field and pass the declared mesh/domain checks.

The normal-direction fields still fail domain checks and remain incomplete.
The actual external environment was not measured. An independent sphere solution
shows that a uniform interior response disappears entirely after center subtraction;
it must not be counted as extra internal gravity. Fixed-density linear response
is also not MOND's nonlinear external-field effect.

## Evidence and next work

- [Stationary covariance](../mond-atlas-noise-stationary-001/README.md)
- [Spatial-scale channel covariance](../mond-atlas-noise-scale-channel-001/README.md)
- [Kinematic selection transfer](../mond-atlas-selection-transfer-001/README.md)
- [External-field predictions and limits](../mond-atlas-external-program-001/README.md)

Independent checks reproduce fitted covariance parameters, joint and aperture
scores, detailed mode diagnostics, all 162 selection aggregate cases and 108
morphology pairs, and the external field comparisons. Source and code bindings,
preflights, controls and failures are retained. Raw data stay outside Git.

The next necessary step is a noise/instrument representation that reproduces
resolved mode powers as well as aggregate scores, and a joint selection-aware
source/motion comparison with propagated mass/geometry uncertainty. External
normal-field domain convergence and independently measured environments remain
separate requirements. No new observed-gravity, lensing, cluster or Solar-System
likelihood is admitted here; the full goal is still active.
