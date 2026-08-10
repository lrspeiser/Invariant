# Observational evidence policy

The compiler tests theories against measurements, not against an inferred invisible component presented as if it were measured.

## Admitted evidence

- Raw detector counts, calibration products, and recorded backgrounds.
- Calibrated spectra and measured wavelength ratios.
- Angular positions, separations, shapes, proper motions, and time delays.
- Doppler velocity ratios with line-identification and instrument provenance.
- Baryonic light and gas tracers with their uncertainties and full transformation chain.

Distance-free comparisons are preferred. Within a single galaxy, for example, angular-radius ratios equal physical-radius ratios, and a finite-source exterior prediction can be tested with

```text
v(theta_2)/v(theta_1) = sqrt(theta_1/theta_2)
```

without assigning a redshift distance or an absolute baryonic mass.

## Excluded by default

- An invisible-halo mass, radius, concentration, or fitted rotation contribution as observational truth.
- Any unobserved component used to rescue a failed baryons-only prediction.
- A redshift converted into distance, physical size, mass, or environment without a separately audited distance protocol.
- Supernova distance moduli or cosmological fits without an explicitly authorized raw-data and calibration audit.

A spectral redshift is still a legitimate measurement **as a wavelength ratio**. The model-dependent conversion from that ratio to distance is a separate claim and is not silently imported.

## Required wording

The relevant null hypothesis is “GR sourced by the measured baryons only.” If its prediction mismatches a rotation curve, the report must say that the baryons-only hypothesis failed. It must not say that GR itself was falsified, and it must not introduce an unmeasured source as a post-hoc rescue.

The machine-readable policy is [`../configs/observational_evidence_policy.json`](../configs/observational_evidence_policy.json).

## Solar-System direct-observable protocol

The Solar gate now has a separate sealed contract at
[`../configs/solar_observable_protocol.json`](../configs/solar_observable_protocol.json). It
distinguishes five quantity classes explicitly: raw, calibrated, derived, model-dependent, and
latent. Future predictions target untouched round-trip light-time, coherent carrier
frequency/phase counts, and relative angular measurements. Fitted PPN parameters, precomputed
Shapiro/perihelion/light-deflection residuals, ephemerides estimated using held-out targets, and
object-specific gravity or screening parameters are forbidden as formula inputs or truth.

Splits occur by whole tracking pass or observing session. Training-only initial conditions,
calibrations, covariance, the action hash, likelihood, and stopping rule must be frozen before test
sessions open. The passing protocol audit is
[`../runs/observation-protocol/solar-observable-audit.json`](../runs/observation-protocol/solar-observable-audit.json);
it records `observational_dataset_opened=false` and `formula_search_authorized=false`.

The first authoritative source registration is NASA PDS dataset
`CO-SS-RSS-1-SCE1-V1.0`, the Cassini first Solar Conjunction Experiment raw radio-science archive.
The registration fingerprints all eight volumes' catalog labels and tables and classifies ATDF/TDF
tracking records and RSR receiver samples separately from compressed ODF navigation products and
SPK ephemerides. It is deliberately metadata-only: primary files are not downloaded, the raw
bit-level parsers and covariance transformations are not yet verified, and the eight-volume split
is not frozen. See
[`../configs/observations/cassini_sce1_source_registration.json`](../configs/observations/cassini_sce1_source_registration.json)
and its
[`audit`](../runs/observation-protocol/cassini-sce1-source-registration-audit.json).
