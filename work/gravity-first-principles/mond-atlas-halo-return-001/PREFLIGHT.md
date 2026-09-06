# Published halo to return-field benchmark

THEORY_BENCHMARK_ONLY. Downloaded and hashed McMillan (2017) paper, author's
commit-pinned PJM16_best.Tpot and provenance README, and Li et al. (2020) paper
plus original 12.86 MB Fits.tar.gz. The archive contains per-galaxy numerical
tables. Read their documented parameter values; do not open observed rotation
arrays or the archive's fit plots for scoring. Paper tables viewed to establish
source definitions. These published fits are model targets, not raw 3D truth.

Before implementation: parse all per-galaxy tables and retain NFW-Flat,
NFW-LCDM and Burkert-Flat parameters, uncertainties, fitted M/L, distance,
inclination and fit quality; no quality-based selection. Author Milky Way file
has rho_s=8.53702e6 Msun/kpc^3, rs=19.5725 kpc, slightly more precise than
rounded paper Table 3 / galpy preset. Use author's values and record both.

Implement spherical mass and full vector halo forces; verify independent
density quadrature, galpy NFW/Burkert forces, NFW potential gradient, source
Poisson identity, central/outer limits, units and rotations before target-field
generation. Reject origin for cusp vector evaluation. Do not silently extrapolate
an observed-map claim. A spherical 3D continuation is a model assumption.

Exact effective-return model: H=r^2*g_h, with H'=4pi G rho_h r^2; add the
center-directed field -H r_vec/r^3 to the unchanged baryonic field. Matching
this definition is an inverse reconstruction, not a theory of reflection.
Fit the three fixed approximate return families in the config on inner
dimensionless training radii. Test interleaved and larger radii without refitting.
Report all starts, failures and profile dependence. Fixed shape fits reused
across scaled halos are not new independent confirmations.

Use a separately labeled analytic Miyamoto-Nagai disk as a manufactured geometry
counterexample: a scalar multiple of its field cannot generally reproduce a
spherical center-directed added field above/below the plane. Check its potential
gradient independently. This is not the full McMillan baryonic model. Also report
curl of a locally scalar-boosted field as a conservative-force consistency test.

No claim that gravity is consumed, energy is reflected, exterior spherical
shells attract interior bodies, or an action/relativistic light closure exists.
The next theory step must predict H from ordinary matter and boundary physics
without importing fitted halo parameters. Preserve all original source bytes.
