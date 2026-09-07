# Signed CO source photometry and missing-coverage audit

Freeze before new map-array access. Actual HERACLES moment0 and error maps for
NGC2976/3198, exact hashes and geometry/conversions from existing registered-source
v1 configs. Primary release guide https://www.iram.fr/ILPA/LP001/README and paper
https://arxiv.org/abs/0905.4742. The guide states moment0 has finite measurements
throughout observed regions, including faint/signed values. Missing values must
not be treated as non-detection upper limits. Its integration windows combine
HI-velocity-centered windows and a bright-CO mask; therefore this source product
has a shared processing dependency with HI kinematics, not complete independence.

Source-only photometry; no galaxy velocity values, gravity predictions or lensing
responses accessed. No existing source arrays or reconstruction changed.
For fixed deprojected apertures R=[1,3,6] kpc and [3,10,28] kpc respectively,
measure signed CO luminosity, pixelwise-positive luminosity, their difference,
negative/SNR<1/SNR>=3 area fractions and their luminosity contributions.
Use valid finite moment0 and error>0 support. Four-by-four subpixel area/aperture
integration and one-point comparison; projected physical area converts brightness
to K km/s pc². No extra inclination correction to integrated luminosity.
Report all coverage gaps, including outside native image extent via analytic
projected ellipse area (small-angle approximation explicitly).

Compute sigma quadrature as an independent-pixel approximation and sum(sigma*area)
as the fully-correlated upper standard-deviation bound conditional on reported
per-pixel errors. Do not use either as a calibrated measured likelihood. Native
beam and shared velocity-window correlations are not supplied by the error map.

For comparison only, Gaussian zero-signal clipping gives E[max(Y,0)]=sigma/sqrt(2pi).
Verify by independent quadrature for multiple means/sigmas and deterministic
seeded simulation. This is a reference, NOT a correction to subtract from actual
positive emission everywhere. Pixelwise clipping is a diagnostic counterfactual,
not a claim that the source inversion equals clipping native data. Source conversion
alphaCO=4.35/.65 is inherited/fixed, not an independent mass calibration.
Require exact WCS match, units and hashes, positivity of errors on support,
numerical Gaussian expectation1e-9 and mass/luminosity scaling identity1e-12.
Retain aperture quadrature changes without tuning apertures to the outcomes.
