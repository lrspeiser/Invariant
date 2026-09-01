# Lane 1 source-only preflight

Status: `SEALED_SOURCE_PREFLIGHT_ALL_EIGHT_RESPONSE_BLOCKED_ONE_PHASE_ALIGNED_PARTIAL`

The public-source search did not produce a scoreable response dataset for any of the eight frozen exploration lenses. Seven are hard source-blocked. SDSS J1515+1511 has a genuinely time-delay-aligned A/B spectral pair in the CDS tables, but the tables contain wavelength and flux only: no per-bin uncertainties, wavelength covariance, calibration residuals, or published differential centroid with uncertainty. It therefore remains source-partial, not scoreable.

Opaque acquisition copied and SHA-256 verified 157 public files (420,288,947 bytes), including 17 ESO science exposures, the complete 135-product FORS1 Raw2Raw union for J1226/J1335, four J1515 spectral tables plus ReadMe, seven primary papers, and seven SMOKA metadata pages. No FITS was decompressed and no spectral row/value was parsed. Subaru raw products are account-gated by SMOKA; J1320 has no exact public spectral product identified. Confirmation systems stayed sealed.

The next honest empirical step is new or recovered image-separated spectroscopy at source phases separated by the measured delay, with wavelength-solution diagnostics, centroid uncertainty/covariance, and a precision lens/path model. Catalog redshifts cannot substitute for that response.
