# NGC3198 executed source-only pilot

Disposition: **SOURCE_BLOCKED**; actual annular-fill distance-scaling gate **BENCHMARK_FAILED**. Twelve registered geometry/quadrature cases, 36 component grids and 72 conversion/fill mass rows executed using the unchanged generic builder. One CPU numerical thread; no GPU, new downloads, motion/lensing response arrays or gravity scores. These are repeated conditional representations of the same observations, not independent measurements.

Nine independent tests pass before source construction and again inside the unchanged runner. Thirty actual-header checks pass before construction; four checks on actual supported image pixels also pass before rebinning. All 12 saved packets and 72 mass rows replay. Signed measured and zero-fill integrals obey D² scaling at roundoff, but annular fill fails the same 1e-10 criterion on actual NGC3198 packets.

The first report stopped on that failure and is preserved in findings-001. Total annular-mass scaling errors reach 0.03584%, and the stellar component reaches 0.05312%. Diagnosis from saved axes: floating point floor(radius/annulus_width) relabels 248/304 exact ring-boundary cells inside the cutoff at lower/higher distance. Observed maps and coverage agree at roundoff and trusted masks are unchanged. No source equation, threshold, aperture or packet was repaired. This extension of the synthetic scaling check is a retained actual-data counterexample; do not promote annular-fill outputs to a numerical pass.

Nominal annular-fill conditional mass: **28.601 billion solar masses**: stars 18.234, atomic gas including helium 9.434, molecular gas including helium 0.933. Fixed assumptions are M/L=.6, alpha_CO10=4.35 and R21=.65. These are aperture integrals, not whole-galaxy masses.

| Source | Covered area inside 28 kpc | Signed loss outside grid | Full-input loss after taper |
|---|---:|---:|---:|
| stellar_luminosity | 90.93% | 0.48% | 2.71% |
| atomic_helium | 95.22% | 7.72% | 31.21% |
| co21 | 32.94% | 0.00% | 0.00% |

Loss fractions use signed integrals. The photometric P4 outer isophote (375 arcsec, 25.429 kpc) sets the 28 kpc cutoff, 24–28 kpc taper and +/-32 kpc axis-center field. No motion extent selected the aperture. The gas loss is material and cannot be treated as empty exterior space.

| Component | One vs four subdivisions | Two vs four subdivisions |
|---|---:|---:|
| stellar_luminosity | 0.302% | 0.069% |
| atomic_helium | 0.304% | 0.107% |
| co21 | 3.075% FLAG | 1.051% |

These compare matched accepted annular signed means, radius weighted, against four subdivisions. The unchanged 3% threshold is a follow-up flag. Cell-scale L1 changes and coverage above one remain in CSV diagnostics; passing an annular comparison does not certify cell-scale convergence or observation resolution. No extra reconstruction is performed to remove flags.

Signed negative stellar and CO measurements are preserved in observed grids. Conditional nonnegative projection adds 0.1058 billion solar masses to the CO-derived zero-fill mass under nominal conversion. Annular versus zero fill is an alternative coverage treatment, not an uncertainty interval. Native CO EMOM0 is an area-weighted diagnostic, not propagated covariance.

Both P5-to-P1 checkerboard transfer receipts pass; only their shifts are applied. Original P1 Gaia all-catalog failure is retained alongside the later finite-footprint strict pass (9 validation stars; median .257 arcsec, p90 .317 arcsec). Core TAN explicitly omits inherited SIP. Registration scale/background do not calibrate stellar mass.

P5 STELLAR_MASS_MAP is cleaned flux in MJy/sr; the ICA mask has inherited but semantically incorrect intensity units. HI uses the original CLEAN beam history (11.43108 × 9.36252 arcsec); CO header beam is 13.396779 arcsec. Native beams differ and are neither matched nor deconvolved. Missing/blanked support is not measured zero. HERACLES masking has prior HI-velocity dependence even though this pilot opens no velocities.

Flat-disk inclination is 71.923 degrees at nominal q0=.13. Warp, bulge, depth, source covariance, calibration, missing baryonic phases and exterior mass remain unresolved. No unique 3D source or observational response admission follows. The generic runner’s prior_initial_control_failure field refers to the preserved historical NGC2976 uniform-box test repair, not a failed NGC3198 case.

Reproduce from the repository root with Python313:

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
python -B scripts/build_mond_atlas_ngc3198_source_checked.py verify-freeze
# Frozen settings; select a never-used run-NNN pair for a replay:
python -B scripts/build_mond_atlas_ngc3198_source_checked.py run --run-id run-002
```

The saved run-001, failed findings-001 and completed findings-002 are immutable. The checked runner verifies all new bindings plus the unchanged legacy generic-source-001 freeze. Initial preflight is already executed and may not overwrite its receipts. report_source.py reproduces findings in a fresh checkout with no findings-002. No further run is launched by this report.
