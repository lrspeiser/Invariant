# Frozen pipeline attribution experiment, execution 027

This is a post-hoc causal ablation of the conditional injection pipeline prompted
by selection-transfer-001's channel-placement effects. It is not a gravity fit,
independent galaxy holdout or observational source-likelihood admission.

Reuse the exact source/background/calibration bindings from selection-transfer-001:
NGC2976 THINGS standard native cube, 12 eastern patches, western medians/MAD,
measured restoring beam, historical continuum operator A and three declared
spectral brackets H. Source documentation: Walter et al.2008
https://arxiv.org/html/0810.2125 and the native-selection source-evidence receipt.
All sources were previously development-exposed. No reserved velocities read.

Keep all prior templates: rotation, kinematic twist and streaming; centers
10/20/30; amplitudes5/10; same symmetric-peak normalization within branch and
center; native 81x81 patch and 30arcsec detector. Freeze a 2x2 factorial:
(1) source continuum A versus simple stored-channel selection S (no subtraction),
(2) western per-channel MAD versus its global median in every stored channel.
Threshold remains strictly >2sigma for a run of at least3channels, with no
retuning after eastern pixels. The global threshold is a diagnostic perturbation,
not a proposed new mask or validated false-alarm rule.

Actual observed background is identical in all ablations, including its existing
continuum subtraction, correlations and contamination. Disabling A applies ONLY
to injected synthetic emission: unavailable parent-channel data prevent undoing
continuum subtraction on observed noise. Thus the factorial attributes source
attenuation and threshold effects conditional on the delivered background, not
the result of changing the telescope reduction end to end.

Execute all216template/factorial cases on12real patches (2592trials) and on zero
background (216controls). Retain all original-factor cells and require replay
of prior empirical/noiseless trials to max absolute error1e-10 on flux fractions.
No fitted parameters, source changes, new noise-realization model or optimization.

Before arrays, re-run existing independent analytic controls, zero-motion and
flux-invariance checks. Require (i) S exactly selects stored channels;
(ii) direct finite spectral filter agrees with matrix H to1e-12;
(iii) unit-sum full spatial convolution conserves flux to1e-12;
(iv) spectral/continuum and spatial convolution commute to1e-10 on a manufactured
cube. Failures stop observed patch access and remain saved.

Record signed sums and channel profiles at intrinsic pre-H (weighted by input
cell width), full parent post-H, stored pre-continuum, post-continuum, native-beam,
and detection-beam stages. Keep positive reference flux separate from signed
continuum output; report finite spectral/cropped spatial losses, no renormalizing.
Full-convolution conservation is a control, not an assertion that a cropped
patch conserves all flux. Explicit numerical matrices H,A,S are saved.

Report paired deltas for source-continuum removal and fixed-global threshold,
their interaction, and the center20 versus mean(center10,center30) gap for each
branch/template/amplitude. Report empirical SD/range across12paired positions;
no iid SE, population inference or calibrated causal attribution to foregrounds.
Noiseless versus empirical separates conditional background effects, not an
independent instrumental noise estimate. A change is descriptively material if
absolute mean retained-flux difference exceeds .05; no significance claim.

New private output zero; reserve8GB disk. Exact passband, dirty-beam/CLEAN,
primary-beam/residual scaling, line-free support and background uncertainty
remain unresolved. SOURCE_BLOCKED observational admission is unchanged.
