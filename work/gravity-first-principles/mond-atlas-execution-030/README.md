# Larger program: source geometry and measurement transfer

This increment executes real-source force calculations, second-galaxy noise
and selection experiments, and an explicit route to a first conditional spectrum
comparison. It has no new observed gravity scores and identifies no winning law.
Previous time/oscillator, density/refraction, current and local-calibration
results remain unchanged. They were not all rerun or made observationally valid
by this increment. The full goal remains active.

## Findings

| Work | Executed result | What it establishes |
|---|---|---|
| Logarithmic distributed response on NGC2976 maps | RTX5090, 67.5s, 10,368 component/total records; all16 extra-force refinement gates pass | A stable conditional field with stronger directional variation than the prior finite NFW-like kernel; no measured velocity or lensing agreement. |
| Newton calculation from the same continuous maps | 53.6s, 4,608 vectors; all16 box gates and12/16 grid gates pass | The near-source integration problem is substantially reduced. Only f4/stellar-height0.4kpc passes every component and total gate; other failures remain numerical, not failures of Newton. |
| NGC3198 selection/uncertainty transfer | 9,558 injections and8,262 known-template amplitude trials; independent score replay below7.12e-12 | Threshold detection can lose extended flux and template-specific uncertainties can exceed aggregate forecasts. No gravity-law inference. |
| Source and observation admission audit | 23 hashes/eight source packets verified, 15 fixed apertures, five finite missing products | Existing data suffice for a restricted conditional comparison after adapters; no indefinite demand for perfect3D truth. |
| Force/pressure to emission integration | 20/22 frozen synthetic checks pass; two precision failures isolated to inherited approximateCDF | Ordinary and modified force can feed a common conditional spectrum renderer; real source column forces and native instrument joins remain. |

The logarithmic field's in-plane directional variation decreases from19–29%
at1kpc to1.5–4% at6kpc. The earlier finite response gives10–16% and1–2%.
This supplies a candidate shape difference to test observationally. It is not
evidence for reflection, consumed gravity or time as an energy source. The
static potential can represent several mechanisms; matching it does not identify
one. The chosen eta1,L4kpc,b.05kpc are explicit physical trial parameters, not
calibrated consequences of the dimensionless oscillator.

All original point-quadrature Newton and combined-field convergence failures
remain. The spectral repair preserves the continuous bilinear source and adds
no force softening. Its independent vertical, Poisson and Gaussian/Hankel
controls pass. Three total fields pass, but CO component failures mean only one
source alternative is fully component-validated. No repaired log/Newton ratios
or total-field observational scores have been claimed.

In NGC3198, nominal5/10 injection amplitudes used a native-restored-peak
normalization; actual full-detector peakSNR is1.145–1.487 and2.291–2.974 after
smoothing. Thus the retained all162 recovery-flag failures are not failures
at5/10 detectorSNR. At the stronger strength, threshold masks retain5.6–43.6%
of total signal despite the field capturing over99.996%. Real-background
standardized template amplitude mean-square errors range1.002–3.208 versus
.731–1.607 in32 conditional Gaussian draws. Shared backgrounds and estimated
covariance prevent a naive significance claim.

The full9west/7east163-square patches support the detector's30arcsec smoothing
with actual surrounding background. They overlap and are not independent
galaxies/trials. The original108,130,496-byte private packet exceeds its
104,857,600-byte operational budget; it was restored byte-for-byte after an
interim packaging attempt. The storage failure and2.10e-9 individualGLS-weight
replay flag are visible. All score arithmetic is independently replayed within
7.12e-12. These are qualified completed experiments, not all-green receipts.

The integration's two failed1e-11 cube comparisons measure4.46–4.65e-8 relative
error. A separate exactCDF substitution diagnoses the old approximation and
reduces error below4.62e-16 without changing prior code or outputs. Impossible
orbits are rejected, support dispersion and tracer width are separate, and
cropped flux is never normalized back in.

## Next finite work

1. Supply converged density-weighted radial force over the actual HI emitting
   column, not sparse midplane vectors. Preserve source alternatives and resolve
   thin-structure/CO grid failures in a separately frozen refinement.
2. Build the actual source-only HI pressure/emission packet using existing
   scalar-pressure closure; retain fixed5/10/15km/s support alternatives and
   impossible equilibria. Stars and molecular gas contribute to gravity.
3. Join native elliptical beam, descending radio-velocity channels, continuum
   and spectral-response alternatives, finite apertures and flux accounting.
   Upgrade the CDF in a new version or declare its analytic error explicitly.
4. Export/verify the exact12x12 aperture working covariance from the frozen
   western24x24 noise model, preserving units and sensitivity alternatives.
5. Execute the frozen ten-training/five-evaluation aperture comparison with
   only systemic velocity, tracer line width and global emission as fitted
   nuisances; run warp/streaming adequacy controls. Previously exposed regions
   permit descriptive prediction comparison, not a fresh independent discovery.

See the individual log-source, newton-spectral, selection-second-galaxy,
observation-admission and force-emission packages for equations, input hashes,
all failures and reproduction scripts. No raw/private observations enter Git.
