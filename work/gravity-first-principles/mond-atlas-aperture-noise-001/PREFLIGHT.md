# Frozen spatial-aggregation test

SOURCE_BLOCKED. Use only the saved, hash-bound NGC2976 external-background
cores from the interrupted covariance task. No new inner-galaxy cube pixels,
motion targets, source fits or gravitational fields are opened.

Before implementation: six fixed aperture sides 1, 2, 4, 8, 12, 24 native
pixels, covering 1.5 to 36 arcseconds. Tiling is aligned and disjoint within
each core. Fit channel means and aperture residual second moments using only
29 western cores. Shrink covariance 10% toward its diagonal at every size.
Compare with the single-pixel covariance divided by area, using all 27 eastern
cores unchanged. No eastern mean fitting, candidate selection, exclusions or
threshold adjustment. Save models before loading the eastern array.

Independent controls before the new run loads values: exact loop-based tile
averages; separable analytic covariance for independent and common-mode pixels;
explicit inverse/logdet score; training-only fitting interface; singular sample
covariance regularization and malformed input checks. Existing 12 covariance
tests must also pass. Report every scale and per-core score, with the fixed
descriptive q/N interval [.8, 1.2]. No pixel-based confidence intervals or
independence claims. Failed large-scale transfer remains a result.

The scientific question is whether aggregate background uncertainty transfers
across this observation, not whether any gravity formula is correct. The prior
background screening reused this observation and does not certify pure noise.
