# Explicit source alternatives before response access

SOURCE_BLOCKED for observed-response scoring. Construct source-only physical
packets and diagnostics; preserve all original packets. Read the HI source
missingness audit: the current positive latent inverse is fitted to trusted
signed source cells and is not the generic annular-fill map.

Branch A: additional observer-plane Gaussian smoothing of existing f4 positive
latent sources to a nominal30arcsec Gaussian target. Native proxies: stellar
circular2.1arcsec IRAC core (non-Gaussian wings not represented); atomic ellipse
7.407x6.42384arcsec BPA71.79 from MOM0 AIPS CLEAN HISTORY; CO circular
13.396779044784arcsec from actual MOM0 BMAJ/BMIN. The HERACLES paper's13arcsec
is approximate; use delivered-map header. Geometry PA144deg, inclination
53.8623309521deg, distance3.611Mpc. In sky east/north coordinates use
Cadd=C30-Cnative; transform to galaxy coordinates with
A=[[sinPA,cosPA],[cosPA/cosi,-sinPA/cosi]]*kpc_per_arcsec.

The original inverse has no native PSF operator or instrumental deconvolution;
it only removes the prescribed vertical Laplace projection and cell integration.
Its latent density is therefore an effective native-beam source. Applying the
mapped extra Gaussian is a conditional beam-model approximation, not recovery
of true density or exact matching of non-Gaussian PSFs. Spatially invariant
Gaussian convolution commutes with the prescribed separable vertical projection;
test this independently, and retain finite-domain edge differences.

Use original source grid.03125 and nested.015625 bilinear interpolation before
convolution. Positive Gaussian kernel truncated at8sigma on each axis, normalized
once as numerical kernel; full convolution mass closure1e-10. Crop to original
±8kpc, record lost tails and boundary-node removal, never renormalize source.
Require lost mass<1e-6 and coarse/fine source projection RMS<.2%; preservefailures.
Primary height.1stars/.2gas and height.4stellar sensitivity are retained.

Branch B: new missing-cell alternatives fitted from zero initialization on f4
latent nodes, at primary heights only. Both use max(signed measured mean,0) in
trusted cells(coverage>=.5); missing cells inside fittedR5 take eitherzero or
the previously frozen genericannular value. This isolates a declared missing
cell assumption on the SAME trusted positive target; it is not identical to
genericarea-dilutedpositive_zero and not the original signed-weighted inverse.
All cellsR<5 get unit imposed-target weight; this is a regularized construction,
not a source-noise likelihood. Same fixedlambda1e-4,12000iterations,tolerance1e-6,
source-node supportR<6. Keep every optimizer failure; no target velocity enters.
Measured negative values projectedpositive are separately counted and retained.

Before arrays: bind code, this protocol, source manifests/audits/configs and
native FITS files. Primary papers: arxiv1410.0009(S4G ICA),0810.2125(THINGS),
0905.4742(HERACLES); S4G P5 README and Salo2015PSFcore approximation. Record
source headers and exact beam assumptions. Controls: covariance transformation,
Gaussian mass/moments, rotation, projection/convolution commutation, constant
coverage and missing-cell distinction, original inverse unit tests. No raw cube
spectra read. Pressure/emission must be recomputed from the SAME alternativeHI
packet; no relabeling baseline pressure/emission. Field replay if time permits
uses existing conservative column operators with unchanged gravity parameters.
