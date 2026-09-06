# NGC3198 source-only preflight

SOURCE_BLOCKED. This pilot reuses unchanged registered-source equations. No source
packet has been constructed at freeze; no motion, velocity cube, lensing response,
field operator or gravity score is allowed. Numerical CPU thread limit is one.

The largest photometric radius is the P4 375 arcsec outer isophote (25.4291 kpc
at 13.987 Mpc). Round up to a 28 kpc cutoff; linear taper spans 24–28 kpc.
Grid axis centers span +/-32 kpc at 125 pc spacing (513 by 513); cell edges
extend by half a cell. Annuli are 250 pc. These are conditional computational
choices, not new measurements. Header rectangles extend beyond this field,
especially after inclination stretching, and include invalid image areas.
Finite-field loss, taper loss, signed negatives and absent coverage stay explicit.

The twelve geometry/registration/quadrature cases, six conversion/fill alternatives
per case and all thresholds are frozen in the config before reconstruction.
Distance cases scale all physical lengths together to hold angular support fixed.
Both fitted P1 translation receipts are bound; their flux scales/offsets are unused.
P1 core TAN omits inherited SIP under the existing finite-footprint Gaia evidence.
The earlier all-catalog Gaia failure remains historical, not overwritten.

Nine independent analytic/synthetic tests and 30 actual-header checks pass before
packet construction. Actual checks cover 121 pixels per component and every
distinct geometry/registration case using wcslib and finite differences. The runner
repeats nine tests and checks actual supported image pixels before rebinning.
Any failure is preserved; no threshold or aperture may be retuned from responses.
Cell/annular coverage thresholds .5/.2 and the 3% annular refinement flag are
unchanged from generic-source-001. A flag is a numerical limitation, not a pass.

Native beams are retained without matching/deconvolution. P5 flux is not mass;
M/L, CO conversion and excitation are assumed. HI blanked zeros are unobserved;
CO remains signed until conditional nonnegative fill. EMOM0 is a diagnostic,
not propagated covariance. No unique three-dimensional mass is established.
