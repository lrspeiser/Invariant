# Individual channel scatter needs a finite-sample reference

The earlier result that21of42DCchannels lie in the descriptive range[0.8,1.2] is not, by itself, evidence that the variance model is wrong. With27independent Gaussian cores and correctly known mean/variance, the expected number in that range is **22.67**, with count standard deviation3.23 if channels are also independent. The reference probability of21or fewer is **0.357**.

No individual channel is exceptional at family alpha0.05 under a two-sided chi-square reference with Bonferroni adjustment; the smallest adjusted p value is0.543. This conclusion does not require independence between channels, but the marginal chi-square law still requires the assumed independent Gaussian cores and correct reference parameters.

This is a retrospective conditional reference, not a recalibration of the fitted procedure. It omits parameter-estimation and model-selection uncertainty, and does not establish core independence or Gaussianity in the actual sky. The actual pipeline's predictive distribution may differ. Source-region behavior, cross-core dependence and transfer remain unresolved.

All original scores and descriptive pass/fail flags are preserved. No threshold was changed, no model refitted and no channel excluded. The result prevents us from treating ordinary small-sample scatter as proof of an instrumental defect. It does not admit the full source likelihood or establish that the variance model is correct.

The chi-square density normalization and interval probability were independently checked by quadrature. Only previously published scalar channel diagnostics were read; no raw sky or galaxy-motion observations were accessed.
