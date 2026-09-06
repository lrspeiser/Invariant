# Gravity pattern system: executed source, motion and lensing milestone

The seven-task system has now completed its first compute, radial-learning,
source-metadata, gas-selection, stellar-registration, synthetic-motion and
lensing-ingestion milestones. **152 integration tests pass.** This is progress
toward a gravity test; it does not establish a new gravity formula. The larger
goal remains active, with zero admitted observed full-field cube likelihoods.

## What we have learned

| Question | Executed finding | What it means |
|---|---|---|
| Do the tested galaxy structure summaries predict departures from MOND? | In the previous CUDA run on 126 galaxies, the combined nonlinear features improved average squared error by 2.26%, but the result ranged from -5.33% to +10.29% across split seeds and its uncertainty included zero. | No stable new structure correction yet. Preserve the negative and ambiguous results. |
| Can mismatched stellar images imitate structure differences? | Registering five cleaned stellar reconstructions to their parent images reduced validation mismatch from 16–76% to 0.6–4.4% in the first partition. Reversing the calibration blocks retained all five overall relative passes. | Position and processing differences matter before interpreting mass structure. This is image reconstruction, not a measured change in gravity. |
| Can ordinary motion imitate extra rotation? | In one synthetic radial-flow case, a circular-only model fitted 105.56 km/s rotation instead of the injected 100 km/s and changed position angle from 18 to 31.23 degrees. | Streaming and viewing geometry can absorb one another. A good rotation fit alone need not identify the force. |
| Does a richer motion model make better predictions? | Warp, streaming and uneven-emission cases improve held-out image/channel predictions; zero-amplitude freedom slightly worsens them. Face-on velocities remain unidentifiable even when pixel predictions pass. | We have useful positive, null and degeneracy controls for the eventual observed-data comparison. These are six conditional simulations, not galaxy detections. |
| Can we add an independent light-bending test? | Three SLACS systems now have verified redshift/dispersion rows, photometric inputs and native HST SCI/ERR/DQ image arrays. | There is a concrete lensing pilot locally. Its image likelihood and light-propagation model still require validation. |

The existing learning result is in
[pattern-findings-001](../mond-atlas-pattern-findings-001/README.md).
No new stellar-transfer, motion or lensing result is used to retune that run.

## Source registration: four usable absolute-position transfers

The completed [stellar-transfer report](../mond-atlas-stellar-transfer-findings-001/README.md)
preserves both partitions, every quadrant and exact input/output hashes.
NGC2903, NGC2976, NGC3198 and NGC3521 pass relative transfer and the prior
finite-footprint Gaia position check. NGC4214 passes the overall relative test
but lacks sufficient absolute-position evidence and has a 9.11% mismatch in
one validation quadrant. It is not promoted to an absolute pass.

The two partition estimates differ by 0.054–0.152 P1 pixels; this is sensitivity
to an alternate split of the same data, not a posterior or independent epoch.
Flux-scale/background fits are registration nuisance terms. They do not
calibrate stellar mass. Existing NGC2903 fields and their older integer shift
remain unchanged. Unknown depth, absolute photometry, component covariance and
mass-to-light uncertainty remain unresolved.

The original transfer runner is frozen because both runs bind its hash. Review
found it could overwrite private sample packets if reused with a new public
directory but the same private configuration. **No overwrite occurred**: the
two executions used distinct private directories, and all ten packets rehash.
Use the new checked entry point for subsequent executions; it refuses existing
private samples and public outputs and checks their repository locations.

```powershell
python -B scripts/run_mond_atlas_stellar_transfer_checked.py --config <copied-config-with-new-private_directory> --output work/gravity-first-principles/<new-transfer-run>
```

## Motion controls: an executed conditional comparison

[Motion run-002](../mond-atlas-motion-controls-001/run-002/README.md) passes 25
numerical controls before generating its six synthetic responses. The
independent reference uses different quadrature and projection/convolution
implementations. The relative cube discrepancy falls from 0.1142% to 0.02586%
to 0.006590% as quadrature is refined at the declared instrument.

The expanded fit includes inclination/position-angle warps, radial flow and
azimuthal emission asymmetry. Each case has separate training, held-out
channels, held-out pixels and joint holdouts. All six expanded fits satisfy
the descriptive prediction criterion; only the first five satisfy parameter
recovery tolerances. The face-on failure and an unresolved rotation/flow
degeneracy are retained. A circular-only fit also meets the loose prediction
criterion in the warp-only and radial-only cases, so the criterion does not
prove unique identification or significance.

![Synthetic held-out comparison](../mond-atlas-motion-controls-001/run-002/heldout-diagnostics.png)

This package is THEORY_BENCHMARK_ONLY. Its covariance is a supplied diagonal
matrix, and its source shape, center, flux and instrument are known. It does
not yet implement pressure support, thickness, continuity, force balance or
validated observed channel covariance/selection. Six noise realizations are
an illustrative study, not empirical uncertainty coverage. No observed galaxy
is fit by this benchmark. The separate existing regression suite reads
previously exposed SPARC assets for integrity checks, as disclosed in the
motion validation receipt.

## Lensing: real inputs, no admitted gravity score

[Lensing replay-002](../mond-atlas-lensing-pilot-001/README.md) includes
J0216-0813, J0737+3216 and J0912+0029. The parent integration rehashed all 29
principal downloads (650,187,552 bytes), selected legacy arrays and derived
packets. Offline replay exactly preserves the measured table, source tables
and source manifest. Native ACS F814W pixels have matched SCI/ERR/DQ arrays,
units, identities and subpixel WCS round trips. They are not automatically
matched to the twelve nearby HI seeds.

The packet distinguishes photometry from inferred stellar mass. Auger masses
use a velocity-dispersion-conditioned prior and are ancillary, not independent
mass inputs for a dispersion test. Grillo photometric alternatives retain
their IMF/template assumptions. No lens-model, halo or velocity-inferred mass
is a baryonic training label.

Validated PSFs, foreground/arc separation, analysis masks, full image covariance
and an explicit relativistic light-propagation closure remain missing. A
nonrelativistic MOND acceleration prescription does not by itself specify light
deflection. No lensing/gravity fit was performed. Primary-source inspection
incidentally exposed legacy reserved SLACS rows; that old set is not certified
unseen. Only the three exploration systems received exact-row/image ingestion.

## Integration, publication and next executable work

The parent [verification](verification.json) records **152 tests, zero failures,
errors or skips**, 609 prior manifest entries, 26 lensing package entries, 36
motion package entries, all ten private stellar sample packets and twelve
private motion packets. It rehashed 787 unique files across these checks.
The package manifests exclude themselves; the new cumulative publication
manifest adds them explicitly. Raw FITS, NPY and NPZ remain outside Git.

The initial Git whitespace check flags legal trailing spaces inside the generated
Matplotlib SVG path attributes. The SVG parses as XML and retains its bound
artifact bytes. Code/report whitespace checks exclude that one generated SVG;
the exact scope and initial diagnostic are recorded in publication-checks.json.

The audit ran before editing the mutable handoff/task documents. Their exact
prior bytes are archived here; future audit replay checks those copies. The
lensing task's even earlier handoff hash exactly matches the existing
execution-013 archive. No mismatching checksum is waived.

```powershell
python -B scripts/verify_mond_atlas_execution014.py --output work/gravity-first-principles/<new-integration-audit>
```

Publication base is main commit `5b1ef68807417c344df8a923471124b83baa8194`.
The coordinator publishes an ordinary fast-forward update after validating the
staged Git blob bytes against the new manifest. The earlier source/native
milestone is already on main; no permissions blocker remains.

The next useful increment is a second conditional source pilot, starting with
NGC2976 using its independently checked geometry and measured stellar transfer.
It needs a generic typed source adapter; the NGC2903-specific builder must not
be reused with its hardcoded shift. Preserve plausible mass/depth alternatives
and missing-phase uncertainty. In parallel dependency terms, motion fitting
needs a correlated-noise and selection test plus pressure support before
observed scoring. Lensing needs its own image calibration and light model.

Only after those steps should the learner compare measured structure against
motion residuals on whole held-out galaxies, then physical groups and surveys.
The target remains 10–20 development pilots and expansion to eligible larger
samples. These milestones make that work executable; they do not make every
galaxy's three-dimensional ordinary matter directly observed.
