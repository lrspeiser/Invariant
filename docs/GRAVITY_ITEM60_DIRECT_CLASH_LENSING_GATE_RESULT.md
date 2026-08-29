# Item 60 direct-CLASH lensing gate result

## Decision

`BLOCKED_THEORY_UNDERDEFINED_RETAIN_CANDIDATE_AND_DO_NOT_OPEN_TARGET_ROWS`

The unchanged Item 59 law cannot yet make a unique direct-lensing prediction. It outputs
a radial acceleration for slowly moving matter. It does not supply a spacetime metric,
two gravitational potentials, a photon-coupling rule, gravitational slip, a lensing
Jacobian, or a Fermat potential. Therefore it cannot honestly predict any of the six
frozen channels: image positions, parities, shapes, weak shear, magnification, or time
delays.

This is a blocked theory-readiness gate, not an empirical failure and not evidence against
the Item 59 pressure/temperature result. The acceleration candidate and its broader family
remain in the archive.

## Why an acceleration curve is insufficient

A star or gas parcel responds to the dynamical potential, conventionally written
`Phi`. Light deflection generally responds to a combination of two metric potentials,
often written `Phi+Psi`. Knowing only the radial derivative that accelerated the gas does
not determine `Psi`, the relationship between the two potentials, or how either is
projected through a nonspherical cluster.

Many inequivalent photon-coupled theories can share the same nonrelativistic acceleration
curve. Assuming the general-relativistic light-deflection conversion would select one of
those theories without deriving it. Fitting a new lensing coefficient would instead give
the candidate one law for matter and another for light. Both shortcuts are forbidden by
the frozen roadmap.

The same missing structure blocks the direct observables in a chain:

1. image positions require the gradient of a two-dimensional lensing potential;
2. parity, shape, shear, and magnification require its second derivatives and Jacobian;
3. time delays additionally require the full Fermat potential and a declared time/distance
   normalization.

## Real-data boundary

The public-source inventory was frozen before opening target values. It identifies:

- the official CLASH HST and Subaru catalog releases at MAST;
- Subaru calibration and catalog-schema metadata for MACS J0416 and MACS J1149;
- 158 public strong-lensing position records for MACS J0416 and 154 for MACS J1149 in
  the CDS catalog metadata;
- the primary SN Refsdal time-delay and magnification-ratio measurement source.

Because the candidate could evaluate zero of six channels, opening response rows would
only expose future confirmation data without producing a prediction. The run therefore
opened zero target rows and zero GR/NFW mass rows. The 312 record count is source metadata,
not a fit or score.

## Counterexample policy

There are zero empirical counterexamples because no empirical comparison was performed.
The single recorded theory witness is underdetermination: a massive-tracer acceleration
does not uniquely determine photon propagation. This witness blocks promotion through
Item 60 in its present form, but does not prune the empirical formula or its mechanism
family.

When a completed theory reaches the real CLASH rows, one bad arc, one anomalous shear bin,
or one time delay will not kill it. Such mismatches must first be checked against source
identification, redshift, line-of-sight structure, calibration, PSF, gas/light maps, and
covariance. Counts alone are never terminal.

## Next construction

The discovery engine must now generate action-level or explicit two-potential descendants
that preserve the Item 59 massive-tracer behavior while deriving:

- the baryon-to-field equations in two or three dimensions;
- coupling to massive matter and photons;
- one gravitational-slip relation rather than a cluster-specific lensing fit;
- the lensing potential, Jacobian, magnification, and Fermat potential.

Only after one such completion is frozen should the direct CLASH target rows be opened.
Items 61 and 65 test the cross-scale consistency and lensing-slip aspects of that work;
Items 66–68 then test conservation, stability, and causality.

## Reproduction

```powershell
python -m sigma_theory_compiler.gravity_item60_direct_clash_lensing_gate replay
python -m pytest tests/test_gravity_item60_direct_clash_lensing_gate.py -q
```

Paid model calls: zero. GPU use: none.
