# Fixed-aperture noise weights are ready

The exact42-channel marginal covariance for the frozen central12x12 mean was
exported from the existing western24x24 DCT model. No source-region or eastern
spectra were read. Independent manual-cosine/dense-spatial-matrix contraction
agrees to4.16e-16 relative; white-pixel and common-mode analytic limits pass.
Refitting all29 western cores with the frozen recipe exactly reproduces the
published covariance. All29 leave-one-core-out matrices are positive definite.

The predicted standard deviation per aperture channel is0.102–0.174mJy per
native restoring beam. Broad spatial modes dominate: DC accounts for43.34% of
covariance trace and the low-frequency band53.49%, together96.83%. Therefore
pixel averaging does not yield the improvement expected from144 independent
pixels. This is a property of this fitted background model, not a statement
about how gravity propagates or about physical gas structures.

Leaving out one calibration core changes total covariance trace by-1.19% to
+1.05%, while full matrix relative change reaches3.45%. The western per-channel
mean changes by at most0.00334mJy/beam RMS. These29 explicit parameter variants
are exported for later score sensitivity, not interpreted as confidence bounds.
Omitted western-core standardized mean-square error averages1.210 per channel;
it is a descriptive development result with possible spatial dependence,
estimated-parameter uncertainty and previous selection history.

The exported operation is a mean, not a pixel sum. Values usemJy/beam and
covariance uses(mJy/beam)^2; rawJy/beam data must be multiplied by1000 exactly
once. Channel order is the original42 stored channels. Means are western-only;
no local or held-response mean subtraction is permitted. The DC predictive
variance correction and selected non-DC recipe are preserved.

This completes the working-weight adapter in the first-score protocol. It does
not validate source-region noise or correlations between separate apertures.
Use the matrices for the declared descriptive individual-patch losses and their
mean, not a calibrated joint likelihood or a significance claim.
