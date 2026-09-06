# Independent selection-transfer audit

PASS_WITH_SCOPE_LIMITS. All162 aggregate cases and108 paired morphology cases
reproduce exactly from the1,566 saved trials. Every diagnostic gate agrees:

| Background | Trials | Recovery cases passing /54 | Morphology pairs passing /36 |
|---|---:|---:|---:|
| Previously exposed empirical patches | 648 | 19 | 27 |
| Conditional Gaussian surrogate | 864 | 25 | 29 |
| Noiseless synthetic sources | 54 | 27 | 34 |

An independently looped three-consecutive-channel mask matches exactly. A
separately calculated synthetic recovery example reproduces all recovery
quantities within1.74e-18. Independent Gaussian line integrals agree within
3.34e-16. Synthetic intrinsic flux across rotation/twist/streaming differs by
at most1.56e-15 in the tested branch/center combinations. All11 bound input
hashes are verified. No observed velocities or background pixel arrays were
opened for this review; the full expensive production trials were not rerun.

## What the morphology result means

The experiment holds intrinsic brightness fixed, and uses the rotational
template's normalization for each morphology at a given center/response branch.
It does not renormalize streaming back to the same detector peak. This is the
appropriate distinction when asking whether equal-flux but differently moving
sources are selected differently. The saved template table shows streaming
detector peaks only85.4–86.7% of the rotational reference, and twist peaks
97.7–98.1%, while continuum subtraction removes negligible total flux here.

The largest empirical mean retained-flux difference is-7.49 percentage points:
streaming relative to rotation, independent-boxcar branch, center10, amplitude5.
The largest Gaussian difference is-8.62 points; even noiseless streaming loses
5.89 points in its worst case. These are selection effects under the stated
synthetic line fields, not evidence of changed gravitational acceleration.

There is no discrepancy in the table arithmetic or paired-by-draw comparison.
The conditional mask response is not morphology invariant and is not yet an
admitted full observed-motion likelihood. Gaussian recovery outperforming the
empirical cases does not certify the Gaussian surrogate for the actual cube.

## Limits that must remain visible

The 'warp' is a velocity twist with unchanged projected brightness, not a
self-consistent warped density reconstruction. Streaming amplitudes are invented
line-of-sight channel offsets, not measured orbital velocities. The empirical
patches overlap, were screened from the same observation and were already
development-exposed. Their spread must not be treated as independent sampling
uncertainty. The simulated spectral and beam-filtered noise remains conditional
on unresolved instrument history and spatial covariance assumptions.

Template and mask implementation checks pass, but the failure cases remain
scientifically meaningful. These results support improving and validating the
selection model before interpreting motion residuals as a gravity pattern.
