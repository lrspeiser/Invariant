# Larger program: joint field endpoint and local calibration

This round completes four bounded follow-ups. The clearest progress is a passing joint boundary/grid test for the smoother density-response model. The clearest new physical requirement is consistent normalization to locally measured gravity. Neither result is a fit to observed anomalous gravity.

## Findings

**The external-field solver passes the combined test for ell=0.5 kpc.** A new 385-cubed calculation at 24 kpc halfwidth ran in 106.4 seconds with sampled peak working memory 9.12 GB. At fixed fine spacing, increasing the domain from 18 to 24 kpc changes the center-relative field by 0.493% RMS. At fixed 24 kpc domain, halving grid spacing changes it by 1.388% RMS; the worst height group changes 4.784%. All original global and height gates pass. The earlier ell=0.25 kpc failure remains; no additional result for that law is implied. This is an imposed normal field acting on a conditional NGC2976 source, not a measured environment.

**Noise-group calibration improves, but finer failures remain.** Western-only selection chooses a regularized variance estimator that moves the previously failing background-offset channel group from q/N 1.566 to 1.107. All four original channel groups and six aperture checks now pass. Only 21 of 42 individual background-offset channels lie inside the same descriptive range, however, and one spatial mode remains outside it. These exposed-region checks are not independent confirmation or significance tests. A full source/emission likelihood is still unvalidated.

**A common gravity increase cancels against the local measurement of G.** In a locally constant medium our equation implies G_measured=G_bare/epsilon_lab, so another constant environment has a/Newton=epsilon_lab/epsilon_environment. Thirty-six manufactured environment pairs pass the analytic checks. Our large averaging scales also mean the Sun's internal density alone cannot guarantee the high-density limit: surrounding Galactic matter must be modeled. Actual local calibration remains unresolved; this is not a new Solar-System observation or a falsification of every refracted-gravity model.

**Foreground contamination is testable but not diagnosed.** The actual NGC2976 cube has radio velocity coordinates, despite an earlier expectation of optical coordinates. Astropy gives stored zero-based channels 10/20/30 as +54.527/+3.000/−48.527 km/s. The nearby-group foreground paper uses a different observation and does not supply a verified frame/convention match in the reviewed material. No foreground mask or channel exclusion was transferred. Low systemic velocity motivates further checking; it does not identify the cause of the injection-recovery deficit.

## Evidence and limits

- [Joint external-field calculation](../mond-atlas-external-joint-001/REPORT.md)
- [Background-offset estimator and independent replay](../mond-atlas-noise-dc-regularization-001/README.md)
- [Local gravity normalization benchmark](../mond-atlas-normalization-001/README.md)
- [Foreground metadata audit](../mond-atlas-foreground-metadata-001/README.md)
- [Status of every broader mechanism family](MECHANISM_STATUS.md)

The previous completed batch is [round027](../mond-atlas-execution-027/README.md), including 2,592 actual-background injections, 216 noiseless controls, and molecular-gas coverage/clipping tests. All prior published artifacts remain preserved. New calculations are CPU tasks; this round does not claim RTX5090 acceleration or an RL discovery.

No new observed-motion, lensing or cluster response fit was performed. Remaining requirements include transferable source/noise/selection likelihoods, independently constrained source depth and surroundings, local-G calibration, and Solar-System/cluster/lensing predictions from the same physical law. The goal remains active. Improved numerical precision must not be reported as evidence that one of the speculative mechanisms explains the observations.
