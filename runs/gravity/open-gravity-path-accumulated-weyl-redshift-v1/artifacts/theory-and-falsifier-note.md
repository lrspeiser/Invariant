# Path-aged Weyl-exposure redshift: theory and falsifier note

## Result

This package advances the path-accumulation idea beyond an endpoint lapse rewrite.  The
extra logarithmic redshift is a scalar line integral along the photon path,

`ln[(1+z_obs)/(1+z_GR)] = alpha * integral A(s) q_W d ell`,

where `q_W=(|C_abcd C^abcd|/48)^(1/4)` and `A(s)=1-exp(-s/L_H)`.  The only fitted
quantity is one universal dimensionless coupling `alpha`; `L_H=c/H0` is fixed.

## What is genuinely discriminating

Two images of one delayed quasar have the same emitter and observer but unequal paths.
A static GR lens gives magnification, deflection, and a time delay, but no residual
stationary-lens frequency change after source epochs are aligned.  This law instead
predicts a signed differential narrow-line shift proportional to the unequal integrated
Weyl exposure.  An exact one-form or endpoint lapse predicts zero path difference.

For the eight frozen exploration lenses, the model-lifted point-lens coefficients span
46.96709 to 212.725749 km/s per unit
`alpha`.  A clean 10 km/s null on one object would imply an approximate single-object
bound on `|alpha|` between 0.0470088837 and 0.212915043, before
moving-lens and source-structure nuisances are included.  These are predictions, not
scores; no spectrum or response file was opened.  The bound predictor TSV was parsed in
full: all 12 source rows were read, eight exploration rows entered these predictions,
and four confirmation predictor rows were parsed but not used.  No confirmation response
was opened.

The geometry and exposure now use one model throughout.  The point-mass flux-ratio
inversion determines the exact two image roots and Einstein angle; that same point mass
sets the Schwarzschild Weyl scalar in both path integrals.  No SIS image relation is
mixed with a point-mass curvature exposure.  The prediction ledger propagates angular
positions in arcseconds into angular-diameter distances, physical impacts, and
gravitational radius in Mpc before forming the dimensionless exposures; its mass and
velocity columns are explicitly in solar masses and km/s.

The path age is also the law's stated invariant rather than a distance proxy.  In the
frozen flat-FLRW baryon congruence the code integrates
`d ell=c dt=(c/H0) dz/[(1+z)E(z)]` from lens to source and checks it against an
independent adaptive quadrature.  The older `(chi_s-chi_l)/(1+z_l)` approximation is
retained in the prediction table only as rejected counterevidence.

## Exact limits

- `alpha=0`: exact GR.
- Weyl tensor zero: exact zero extra effect, including homogeneous FLRW.
- identical path exposure or equal lens impact parameters: exact zero differential.
- `B=d phi`: endpoint-only control with zero path holonomy.
- short paths: the activation begins quadratically, suppressing a one-AU solar ray far
  below the extragalactic coefficient without fitting a Solar-System cutoff.

## Conservation and causality boundary

The line integral is coordinate invariant and invariant under ray reparameterization
once the physical matter congruence `u^a` is specified.  It is causal because it uses
only the already-traversed path.  It is not a complete field theory: a nonzero shift in
a stationary spacetime means the photon stress tensor exchanges energy-momentum with
something.  A publication-level physical theory must derive a compensating field from
an action and prove total covariant stress-energy conservation.  It must also derive the
preferred congruence instead of selecting one by coordinates.

## Existing work and claim boundary

Nonintegrable Weyl length transport, curvature/tired-light mechanisms, integrated
Sachs-Wolfe redshift, moving-lens frequency shifts, and multi-image spectroscopy are all
published predecessors.  The exact path-age-gated fourth-root Weyl exposure and this
frozen lens coefficient were not located in the targeted primary map, but that does not
establish historical novelty.  The defensible status is a potentially new testable
synthesis, not a discovered new law.

## Next empirical falsifier

Acquire or release spatially resolved, wavelength-calibrated spectra for the frozen
exploration lenses.  Align the two image epochs with the measured time delay; use narrow
forbidden emission lines or stable narrow absorbers; fit one universal `alpha`; and model
the moving-lens, differential-magnification, microlensing, intrinsic-variability, plasma,
dust, and calibration channels listed in the frozen config.  A null coefficient across
unequal exposures falsifies the nonzero law over the measured range.  Any positive must
then survive the sealed confirmation systems with unchanged `alpha`.

## Package decision

`PASS_KINEMATIC_PATH_LAW_AND_RESPONSE_BLIND_PREFLIGHT__BLOCK_DYNAMICAL_COMPLETION_AND_SPECTRAL_RESPONSE`
