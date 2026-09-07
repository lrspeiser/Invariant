# Requested joint external-field endpoint completed

The24kpc halfwidth domain was solved at fine spacing(0.125,0.125,0.0625)kpc,
385^3 cells, for the unchanged ell0.5kpc normal-field case. Both domain and
mesh comparisons at this joint endpoint pass the original numerical gates.
This closes the particular missing joint-endpoint check from execution027.

| Comparison | Raw-field RMS change | Center-relative RMS change | Largest center-relative height-group change |
|---|---:|---:|---:|
| Fine18 -> fine24 | 0.603% | 0.493% | 0.574% |
| Coarse24 -> fine24 | 0.268% | 1.388% | 4.784% |

Thresholds remain5% globally and8% for each height group. No source properties,
permittivity parameters, smoothing length, external direction or amplitude were
retuned. The earlier ell0.25kpc height-group failure is preserved and is not
resolved by this ell0.5 calculation.

## Conditional field pattern

The applied acceleration is a unit normal field. The predicted center response
is(0,0,0.424701) to numerical precision. After subtracting that common center
acceleration, the RMS differential field is0.085368 per unit applied field.

| Height above plane | RMS center-relative field per unit applied field |
|---|---:|
| 0 kpc | 0.01911 |
| 0.2 kpc | 0.02373 |
| 0.5 kpc | 0.04960 |
| 1 kpc | 0.16051 |

The conditional spatially varying coefficient produces a differential response
that increases with height in these sampled groups. These values are not a
fraction of an observed gravity anomaly, nor necessarily an extra radial inward
pull. No actual external amplitude, environmental mass distribution, observed
velocity or lensing response was fitted. This remains a linear fixed-density
boundary calculation with observational admission SOURCE_BLOCKED.

## Resources and verification

The initial8GB working-memory estimate was an operational bound, later revised
to16GB by the coordinating agent before any new solve. The original estimate
and resource addendum are preserved. No alternative21kpc calculation was run.
The requested solve completed in106.4seconds on one CPU thread, below the
300second budget. Sampled peak process working memory was9.121GB;0.1second
monitoring is a sampled peak, not a guarantee about every instantaneous allocation.
Full3D fields were held in RAM and were not written as raw outputs.

Analytic controls were rerun; the independent anisotropic sparse reference and
all source/code hashes were verified. The elliptic solve converged in19
iterations, with boundary-forcing-normalized residual6.73e-11. An independently
written saved-vector/resource audit verifies all1,155 vector rows, four comparison
cases and their gates within6.94e-18 arithmetic difference. One new solve and
two exact hash-bound prior solutions supplied these comparisons.

The narrow numerical result is now stronger: this fixed smoother density model's
normal applied-field response survives the tested joint mesh/domain refinement.
It does not establish MOND's nonlinear external-field effect, an equilibrium
galaxy solution, a causal propagation theory or an observed environmental cause.
