# Native NGC2976 background covariance: reviewed delivery

The interrupted task's original run is complete and now independently reviewed.
It fitted eight fixed mean/channel-covariance alternatives to 29 western
background cores, choosing by three geometry-defined western folds before
opening 27 eastern cores. All cores are 24x24 native pixels with a 24-pixel
gap; each spectrum has 42 channels. Both regions were development-exposed
before this experiment. The original files and failures remain unchanged.

Western validation chose a constant channel mean and full covariance shrunk
10% toward its diagonal. Eastern q/N is 1.0122 and all five frozen descriptive
checks pass. Four constant-mean alternatives pass; four affine-sky alternatives
fail the mean-transfer check. An affine fit extrapolated across the galaxy is
not a measured foreground correction.

The chosen model's adjacent horizontal pixel correlation is approximately
0.82 after channel whitening, falling to approximately 0.007 at eight pixels.
This is a measured property of these screened background cores, not an assumed
restoring-beam noise model or a gravity effect. The cross-core warning does not
trigger, but some separation/direction combinations have very few pairs (one
eastern horizontal pair at 96 pixels). Independence is not established.

Predicting even channels conditional on odd channels at the same eastern pixel
gives q/N=1.0098. It is same-observation conditional prediction, not a fresh
noise realization. Pixel-level channel calibration does not certify spatial
averages or a joint cube likelihood. The follow-up aperture experiment explicitly
tests this missing condition.

Review: all 31 distinct manifest/binding paths verified, eight training models
refitted and the same ranking recovered. Sixteen eastern quadratic/log-density
scores agree with explicit inverse and SciPy distribution calculations within
8.9e-16. Twelve original tests pass again. The original figure was visually
reviewed for labels, units and geometry. No original output was overwritten.

Run `python work/gravity-first-principles/mond-atlas-native-covariance-001/verify_package.py`
for a read-only local verification. Raw cube, supports and extracted arrays remain
outside Git. Original cube/source-paper URLs and exact hashes are in the config
and frozen receipts. MOM0 screening reuses the same observation and does not
prove an absence of faint emission or foreground contamination. Source, instrument,
selection and spatial covariance uncertainty still block a full observed cube
likelihood. No gravity or motion score was produced.
