# Exact working covariance for the fixed observation apertures

Before execution, freeze the central12x12 mean in the existing24x24 spatial
DCT model and all42 native NGC2976 channels. Export the previously selected
western model without selecting a new covariance. Its channel means are in
mJy/native restoring beam; covariance is in squared units. Mean aperture
weights sum to1, not144. Keep the supplied DC predictive correction exactly.

Calculate C_ap = sum_b sum_k_in_b (A U_k)^2 p_k C_b. This is the exact
marginal under the frozen working model, not a calibrated joint likelihood
of separated apertures or a claim that source-region noise matches blank sky.
No actual source-region spectra or evaluation responses may be read.

Independently form a manual cosine basis and full576x576 spatial covariance
for each band, then contract A K_b A^T C_b across every channel pair.
Relative agreement must be below1e-11, Cholesky positive definite, and mean
weights/orthogonality within1e-12. White-pixel and common-spatial-mode analytic
controls must pass1e-12. Export mean, covariance, Cholesky and units.

Freeze leave-one-western-core refits using the existing selected spatial
pooling/shrinkage parameters and existing chosen DC target/weight. Do not
reselect any recipe after seeing omitted cores. Preserve29 individual means,
covariances, and omitted-core descriptive residual scores; these are sensitivity
variants, not confidence intervals. Require all covariance matrices positive
definite and finite. Compare direct frozen-recipe refit to the published full
western fit with relative tolerance1e-10 before leave-one-out execution.

Use only the existing bound background packet's training array and frozen
geometric aperture metadata. No validation/source spectra accessed, no new
raw downloads. Public matrices/receipts and scripts; private pixel arrays
remain private. This product supplies weights for the first-score protocol,
not gravity-law evidence by itself.
