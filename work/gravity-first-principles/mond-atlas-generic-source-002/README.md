# NGC3198 source-only pilot: executed, with numerical limitations

**SOURCE_BLOCKED. Actual annular-fill distance-scaling: BENCHMARK_FAILED.**
Twelve cases and 72 conversion/fill mass rows are saved and replayed. Nine
independent source tests and 30 actual-header checks pass before construction.
No gravity score, observed motion/lensing response, GPU run or new download.

The nominal tapered-aperture annular mass is 28.601 billion solar masses under
fixed conversion assumptions. This is a conditional source integral, not a
whole-galaxy mass, unique 3D reconstruction or observational admission.

The retained limitations matter:

- HI loses 7.715% of its signed input integral outside the grid and 31.215%
  after the photometrically selected 24–28 kpc taper. Exterior gas is not empty.
- The actual annular-fill D² check fails the unchanged 1e-10 tolerance:
  up to 0.03584% total mass error and 0.05312% stellar-component error.
  Dimensionful floating-point annulus indexing changes exact boundary cells.
  Signed measured and zero-fill integrals scale at roundoff. No builder repair
  or threshold change was made; the first failed report is preserved.
- CO's one-versus-four pixel subdivision annular comparison is 3.075%, above
  the frozen 3% flag. Two-versus-four is 1.051%. Corresponding two-versus-four
  cell-map differences are 9.93% stellar, 25.49% HI and 30.81% CO. Annular
  agreement does not certify cell-scale convergence.
- Native beams differ; covariance, flux/mass conversion, flat-disk geometry,
  missing phases and depth remain unresolved. Signed negatives and coverage
  above one from finite pixel quadrature are retained in saved arrays/receipts.

Read [the completed findings](findings-002/README.md), then the bound
[preflight](PREFLIGHT.md) and [input audit](input-audit.json).
[The original failed report](findings-001/failure.json) and
[counterexample diagnosis](findings-002/distance-scaling-counterexample.json)
remain part of this package. `run-001/summary.json` is the unchanged runner's
execution receipt; use the findings and verification for final numerical status.

The grid uses the largest photometric radius, P4 Rmax=375 arcsec (25.429 kpc
at 13.987 Mpc): cutoff rounded to 28 kpc, taper beginning at 24 kpc, grid axis
centers +/-32 kpc at 125 pc spacing, 250 pc annuli. Distance, PA, ellipticity,
q0, transfer partition, pixel quadrature, conversion and fill cases retain
the generic-source-001 sensitivity family. No response selected these values.

Both relative transfer receipts and the later finite-footprint Gaia strict pass
are bound. The earlier all-catalog Gaia failure remains unchanged. Historical
STELLAR_MASS_MAP is cleaned flux; ICA-mask intensity units are inherited labels.
Only the fitted translations are applied, through explicit core TAN without SIP.

From the repository root, with the same scientific Python environment:

```powershell
python -B scripts/build_mond_atlas_ngc3198_source_checked.py verify-freeze
python -B work/gravity-first-principles/mond-atlas-generic-source-002/verify_package.py
```

The existing run is immutable. The checked wrapper accepts a never-used
`--run-id run-NNN` for a reproducibility replay; no follow-on is launched here.
`report_source.py` documents and reproduces the post-run review into fresh
`findings-002`; its initial failed version remains under `findings-001`.
The scoped publication manifest contains only the new config, checked wrapper
and this public package. Twelve raw array packets remain in the assigned private
directory and are represented publicly by hashes. Integration/publication belongs
to the coordinating task; no Git metadata, shared modules or prior packages were
edited by this pilot.
