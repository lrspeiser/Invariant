# Molecular source coverage and signed photometry

The inner regions have useful CO coverage, while the much larger source apertures
do not. Native moment0/error maps give the following approximate covered-area
fractions and a diagnostic of replacing negative intensities with zero:

| Galaxy | Aperture radius | Covered area | Increase in CO luminosity under pixelwise clipping |
|---|---:|---:|---:|
| NGC2976 | 1 kpc | approximately 100% | 0% |
| NGC2976 | 3 kpc | 98.17% | 8.22% |
| NGC2976 | 6 kpc | 37.48% | 14.71% |
| NGC3198 | 3 kpc | approximately 100% | 0% |
| NGC3198 | 10 kpc | approximately 100% | 1.68% |
| NGC3198 | 28 kpc | 34.32% | 13.67% |

**Missing coverage is not evidence of a void.** Outer density/coherence claims
must preserve the distinction between unobserved material and a measured low
brightness. These fractions describe the molecular tracer, not all ordinary
matter or a measured three-dimensional gas distribution.

The [HERACLES release guide](https://www.iram.fr/ILPA/LP001/README) describes
finite moment0 measurements throughout the observed footprint and supplies
matching uncertainties. It also describes integration windows partly based on
HI velocities. CO photometry therefore has a shared processing dependency with
HI kinematics; it is not an entirely independent input when predicting those
same kinematics. No HI velocity values were accessed in this audit.

Negative faint measurements are retained in the signed integral. Their physical
source is not negative gas mass; noisy brightness estimates can be negative.
For a zero true signal with Gaussian standard deviation sigma, clipping negatives
to zero produces mean sigma/sqrt(2*pi). Analytic quadrature and seeded simulation
verify this expectation. It is not a correction to subtract everywhere: actual
pixels have different positive signals and correlated errors.

The clipping column is an explicit counterfactual, not a claim that the existing
source inversion clips native pixels or has exactly this bias. It is a change
in the molecular component alone, not a percentage error in total galaxy mass.
No source reconstruction, conversion parameter, observed velocity or gravity
fit was changed. The inherited alphaCO conversion remains an assumed mass proxy.

At R=3 kpc in NGC2976, about 46% of the covered area has signal/error below one;
at R=10 kpc in NGC3198 the corresponding fraction is about 21%. Integrated flux
can still be useful, but independent-pixel uncertainty is not calibrated when
beam and integration-window correlations are unknown. The output retains that
approximation and a fully-correlated upper standard-deviation bound separately.
Neither is presented as an actual posterior interval.

Pixel-area integration uses four-by-four subpixels and a one-point comparison.
The nominal area is an analytic small-angle projected ellipse. Raw coverage
ratios slightly above one (at most about 0.055% with four-by-four subpixels,
0.249% with one-point quadrature) are retained as quadrature/area
approximation effects; the table rounds these to approximately 100%. There is no
silent clipping of the recorded ratios. Projected physical area converts the
brightness to CO luminosity; no extra inclination factor is applied to integrated
luminosity. Exact files, WCS equality, units and hashes were checked.

Source measurements: [Leroy et al.](https://arxiv.org/abs/0905.4742), inherited
registered-source configurations and private moment0/error maps. Protocol,
Gaussian checks and all 12 aperture/quadrature records are in PREFLIGHT.md and
run001/. Independent WCS/photometry review accompanies the result. The next step
is a signed, selection-aware source likelihood with correlated errors and explicit
missing-footprint uncertainty, not filling unobserved regions as known empty gas.
