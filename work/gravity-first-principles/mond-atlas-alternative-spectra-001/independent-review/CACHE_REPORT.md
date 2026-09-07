# Independent alternative source transport audit

All16 caches pass hash, source-packet, HI mass/flux, matched source-force and pressure interpolation, and support-mask replay. All288 model/pressure/instrument envelope cases agree exactly. Atomic helium is removed once for HI emissivity; the gravity tables retain helium. The thick common30 case uses its own stellar-height .4 force rows despite the cache adapter's h0p1 array-slot name. Common30 thin/thick correctly share the same HI geometry and pressure.

The unsupported common30 source is chiefly an inner-region issue after projection. On the finest planar/vertical grid, R<.05kpc contributes .026463Jy km/s globally and at most2.08580e-5Jy km/s to a sampled aperture. R>6.025kpc contributes .067645Jy km/s globally but only5.69503e-11Jy km/s to an aperture. Their maximum weighted contributions differ by about366,000. These are separate maxima across apertures, not necessarily the same aperture.

The two missing-region alternatives have no radii outside their matched table range. Supported negative-v² nodes remain classified as unknown independently of radial support. Their largest weighted flux across all cases is4.89224e-9Jy km/s. The largest complete unknown spectral envelope at allowed amplitude2 is .0472806mJy/beam. This retained uncertainty cannot be replaced with a fictitious circular velocity or with the baseline's smaller tail bound.

Code inspection confirms that the signed continuum/channel operator's absolute row sum bounds unknown positive emission after transport. The replay independently reconstructs invalid masks and sums their sampled aperture flux; it inherits the previously audited spectral operator coefficients and native beam sampler. It therefore audits source matching and envelope accounting, not a new independent projection or full numerical convergence calculation.

No observed source-region spectra were read. A path-separator mismatch in the initial review lookup caused a KeyError before evaluation; normalization fixed that review-only issue. The successful receipt binds the final reviewer and completed cache outputs. No source or parent result was modified.
