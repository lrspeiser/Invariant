# Frozen covariance-transfer experiment

Use all 48 completed noisy injection trials from execution032 (eight Gaussian and eight previously exposed real-background assignments, across all three instrumental branches). Keep the injected physical signal, noise packet, apertures and three optimizer starts unchanged. For every trial, refit the three nuisance parameters with each of the29 existing western leave-one-core-out means and covariance matrices. These are sensitivity alternatives, not independent data sets and not an uncertainty posterior.

Compare recovered parameters and held discrepancy with the original full-western result. Freeze held predictions before evaluation, retain every optimizer result and failed rank check. Independently check held quadratic scores with a dense solve. Do not calibrate any noise correction or pick a preferred omitted core, response branch or noise assignment. Report distributions and extrema, not a post-selected significance threshold. No source-region observational spectra are accessed and no gravity-law preference follows from this test.

This tests robustness to the available western noise sample. It does not repair spatial covariance between separate apertures or source-region noise mismatch. The original injected signals are conditional valid-emitter predictions with separately retained unknown-emission envelopes.
