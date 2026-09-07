# Sparse-DC variance regularization improves the retained channel-group failure

Completed one frozen statistical estimator extension on the same historically exposed29western/27eastern NGC2976 background cores. DC provides only one spectrum per core, while42 channel variances and a channel mean are estimated. The new estimator shrinks diagonal variance estimates toward a shared template and includes a finite-training-mean variance correction. It does not introduce a physical noise component.

Western three-fold selection chooses **50% shrinkage toward the mean channel variance**, followed by the fixed factor30/28 for29fitting cores. This factor combines unbiased sample-variance adjustment and a plug-in correction for estimating the mean. It assumes the nominal number of cores is relevant; core dependence and exact predictive-distribution calibration are not established. All11candidates were frozen before eastern access and their eastern scores retained.

| DC diagnostic | Previous | Selected estimator |
|---|---:|---:|
| Whole DC q/N | 1.166 | 0.973 |
| Original lowest-variance channel quartile | 1.566 | 1.107 |
| Original second quartile | 1.141 | 0.906 |
| Original third quartile | 1.048 | 0.941 |
| Original fourth quartile | 0.871 | 0.931 |

The fixed original channel groups now all pass the descriptive[0.8,1.2] range, so the improvement does not depend on rearranging channels into new groups. Whole-core joint q/N remains0.99947. All six aperture q/N values pass, ranging0.973–1.023; corresponding trace ratios range0.962–1.060. No non-DC model parameter was changed.

Important limits remain: **only21of42 individual DC channels** fall inside the descriptive range, and **575of576 spatial modes** pass as before. These are retained residual discrepancies, not a claim that every individual deviation is statistically significant. Adjacent-mode collapsed residual products remain below0.1, but do not establish full cross-mode independence. Source-region and cross-core behavior remain unvalidated, so source/emission likelihood admission is still blocked.

Two pre-access tests verified variance-factor arithmetic, positivity/trace/unit scaling and explicit Gaussian inverse/log determinant. Inherited DCT/operator controls remain bound. Independent review reconstructs the manual cosine transform, source mode powers, channel covariance and all variance candidates, then replays297candidate DC core scores,27selected joint scores,576mode scores,42channel scores,30diagnostic groups and six aperture projections. Maximum discrepancy4.98e-13; western selection matches.

Actual calculation took about1.2seconds, with no new raw bytes and no observed galaxy velocity/lensing/gravity scores. Every western candidate, mean, variance vector and eastern failure is retained. This is further development on previously exposed background regions, not fresh confirmation or proof of an instrumental root cause. No additional noise branch was run.
