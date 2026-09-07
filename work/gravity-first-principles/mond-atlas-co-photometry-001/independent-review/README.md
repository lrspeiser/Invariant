# Independent CO photometry review

**The executed native-aperture bookkeeping passes independent replay.**
All twelve aperture/subdivision rows, including each galaxy's complete finite
outer footprint, agree to **4.46e-11 relative error**. The independent reader is
Astropy FITS, the independent sky transform is Astropy celestial WCS followed
by separately written spherical gnomonic equations, and projected area comes
from a numerical coordinate Jacobian. Neither production `source_coordinates`
nor production aperture weights are imported.

[verify.py](verify.py) and [receipt.json](receipt.json) retain the replay.
The legacy headers trigger Astropy's warning about three WCS axes on a two-axis
image; explicitly extracting the celestial WCS gives the intended spatial map.
No velocity image or gravity response was opened.

The source coordinate transform deprojects positions to define an elliptical
aperture, but the luminosity weights retain **projected physical area**, without
an extra inclination factor. This is correct for integrating the supplied
brightness into projected CO luminosity under the declared geometry. The fixed
CO-to-mass conversion is an assumption, not a new mass calibration.

All declared input hashes were reverified before and after replay. The original
prospective binding list does not include two transitive helpers:
`scripts/build_mond_atlas_ngc2903_source.py` and `scripts/mond_atlas_common.py`.
The parent was asked to include them in a supplemental publication dependency
receipt; a later hash must not be described as prospectively frozen. This is a
reproducibility receipt gap, not a detected numerical disagreement.

The Gaussian clipping expression is correct:

\[
E[\max(Y,0)]=\sigma\phi(\mu/\sigma)+\mu\Phi(\mu/\sigma).
\]

Independent integration over the positive observed variable for negative, zero
and positive means agrees to 5.78e-15. For zero signal the expectation reduces
to sigma/sqrt(2pi). This reference cannot be subtracted indiscriminately from
actual positive emission. The measured clipping increment also exactly equals
the magnitude of the discarded signed-negative luminosity, within numerical
tolerance. It is a counterfactual pixel-clipping effect; it does not measure
the bias of the source inversion or the true molecular mass error.

The reported quadrature sigma is valid **only under independent pixel errors**.
The positive-weight sum of pixel sigmas bounds the standard deviation for any
covariance consistent with those marginal sigmas, by the covariance inequality.
Neither is a measured confidence interval. The independent approximation is
not generally a lower bound; anticorrelation can reduce the actual variance.
Calibration, mask/window uncertainty and conversion uncertainty lie outside
these conditional marginal-error calculations.

Coverage fractions above unity remain visible: with four subdivisions,
NGC3198 R=3 kpc gives 1.00054793, R=10 gives 1.00009457, and NGC2976 R=1 gives
1.00008386. They reflect finite aperture quadrature and the approximate analytic
ellipse denominator, not physical coverage exceeding 100%. Do not clip them to
one or imply precision beyond the quadrature. Outer measured fractions are
37.48% for NGC2976 R=6 and 34.32% for NGC3198 R=28; each outer aperture includes
every finite measured native pixel. Missing area is not a nondetection limit.
The actual files contain no finite-intensity pixel excluded for an invalid error.

The [HERACLES release guide](https://www.iram.fr/ILPA/LP001/README), independently
checked during review, describes integrated maps over observed regions and an
integration window combining local HI-centered velocities with bright CO
selection. Its uncertainty maps describe that same window. Therefore the CO
source product shares processing dependence with HI kinematics. Reading this
documented dependency is not accessing new target velocity values, but a future
joint likelihood cannot simply assert statistical independence of CO and HI.

No implementation bug was identified for these inputs. The result remains
source photometry, with conditional geometry, noise and conversion assumptions;
it admits no new gravity likelihood or calibrated mass posterior.
