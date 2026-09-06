# NGC3198 actual source recovery: numerical blocker fixed

The failed distance-scaling check is resolved. Twelve fresh stellar/HI/CO source packets and72 conditional mass rows were rebuilt from the same real images using the previously frozen correction. Maximum distance-scaling error fell from **0.03584% to8.89e-16 relative**. All saved measured grids and coverage arrays remain byte-for-byte identical as arrays to the original packets. Only annulus and taper indexing was corrected.

This is actual resolved-source progress, not another radial formula fit. It supplies a numerically checked family of registered flat-disk source maps for later source reprojection, height and force calculations. It does **not** establish a unique three-dimensional galaxy or admit observed gravity scoring.

## Correction and validation

Original floating-point `floor(physical_radius/physical_annulus_width)` changed exact ring-boundary membership when distance changed, even though the physical lengths scaled together. The existing correction protocol specified integer cell coordinates and exact squared-radius ring thresholds. The v2 module copies the legacy deposition/registration/conversion code, changing only this classification and the related dimensionless taper/support calculation. No image values, masks, flux conversions, thresholds or apertures were retuned.

- Original nine registered-source tests and four new v2 tests passed before source reconstruction.
- New controls include independent integer/Decimal boundary expectations, one-ULP neighbors, multiple distance scalings, signed images with holes, and bad metadata rejection.
- Thirty actual-header coordinate, projected-area and registration checks passed, plus the legacy supported-pixel WCS checks during construction.
- Independent packet replay checked36 component cases and72 mass rows. Reconstructed annular fills agreed within5.52e-18 relative L1; packet integrals within5.56e-16; mass arithmetic within2.23e-16.
- Original configuration, code, freeze and correction bindings verified unchanged. Existing failed packets and reports are preserved.

One orchestration attempt failed before packet construction because a header-audit function returned four values rather than the three expected by its wrapper. `failure.json` and `failed-attempt-runner.py` preserve that attempt. The corrected attempt is recorded separately under `attempt-002/`; no source algorithm or numerical tolerance changed because of that wrapper error.

## Remaining source limitations

The unchanged3% annular quadrature follow-up flag still triggers for CO at one versus four pixel subdivisions: **3.075%**. Two versus four gives1.051%; stellar and HI comparisons remain below0.31%. These are annular comparisons, not proof that every cell has converged or that telescope beams have been removed.

At nominal conversions and annular fill, the tapered region contains conditional masses of18.234 billion solar masses in stars,9.434 billion in atomic gas plus helium, and0.933 billion in molecular gas plus helium:28.601 billion total. Alternative conversion/fill choices span21.59–35.98 billion in the same nominal geometry. These are assumed-conversion source alternatives, not a posterior interval or whole-galaxy measured mass.

Missing regions remain explicit. Within the cutoff, coverage fractions are90.93% for stellar light,95.22% for HI, and32.94% for CO. Missing CO support is not a measured void. Nonnegative CO projection adds about106 million solar masses under the nominal molecular conversion; signed observed values are retained separately. The photometric taper/finite field removes about31.2% of the released full-image signed HI integral; exterior matter cannot be assumed absent.

Other unresolved issues: nominal inclination71.923 degrees under an assumed intrinsic thickness, warps/bulge/depth, unmatched native beams, calibration/M/L/CO excitation, hot/ionized/CO-dark phases, correlated noise, and unobserved exterior mass. HERACLES moment masking used HI velocity windows upstream, so the source processing is not independent of all kinematic information even though this recovery opens no velocity arrays.

## Files and readiness

- `scripts/mond_atlas_ngc3198_source_v2.py`: separately versioned source implementation.
- `scripts/build_mond_atlas_ngc3198_source_v2.py`: checked immutable construction.
- `scripts/verify_mond_atlas_ngc3198_source_v2.py`: independent source-only replay.
- `tests/test_mond_atlas_ngc3198_source_v2.py`: added numerical controls.
- `run-001/`: source provenance, per-case packet hashes, annular profiles and conditional masses.
- `verification/receipt.json`: before/after distance checks, independent arithmetic and retained convergence flag.

Private packets occupy180,004,633 bytes, below the2GB budget; over26GB was free after construction. They remain outside Git under `work/private/mond-atlas-ngc3198-recovery-001/run-001/`. No downloads, GPU run, velocity likelihood, lensing access or gravity score occurred. Source disposition remains `SOURCE_BLOCKED` for physical3D/response admission. The annulus discretization blocker is fixed; source uncertainty and the next field/observation gates remain to be completed.
