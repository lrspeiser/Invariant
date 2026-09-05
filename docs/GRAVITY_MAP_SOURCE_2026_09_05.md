# Observed galaxy source bridge — NGC 3198 development pilot

The isolated scalar solver now accepts a conditional axisymmetric reconstruction
of observed stellar and gas maps. This is source and numerical progress, not a
rotation-curve result or evidence for a new gravity law. No velocity values were
used by these three runs. The already exposed SPARC response container was
inspected for provenance and membership during this work; it is not untouched
confirmation data.

## Admission and fixed source assumptions

NGC 3198 was selected for its seven verified source files, published photometric
geometry, existing Run A training membership, and prior photometry-exploration
membership. Selection preceded any new velocity comparison. The admission is
conditional on an axisymmetric lift of two-dimensional maps, not an observed
three-dimensional density. The source inventory and every runtime input are
bound by SHA-256, with exact input bytes saved inside each run.

The stellar image is old-stellar **flux**, converted using 704.04 solar
luminosities per square parsec per MJy/sr and the declared mass-to-light ratio
0.6. The source paper describes this conversion and the limitations of a fixed
mass-to-light ratio. [Querejeta et al., S4G mass maps](https://arxiv.org/abs/1410.0009)
The atomic and molecular maps come from
[THINGS](https://arxiv.org/abs/0810.2125) and
[HERACLES](https://arxiv.org/abs/0905.4742). The inherited helium factor is 1.36,
CO conversion is 4.35, and CO(2–1)/(1–0) ratio is 0.65; these are source
assumptions, not constants inferred from gravity residuals.

The existing [S4G photometric geometry](https://arxiv.org/abs/1503.06550) contract
sets distance 13.987 Mpc, position angle 33.6 degrees, and outer ellipticity
0.666. An assumed intrinsic axis ratio 0.13 implies inclination 71.923 degrees.
Azimuthal averaging discards measured asymmetry. Annuli are 0.5 kpc wide;
nonnegative PCHIP interpolation, a flat central core, and a 2 kpc cosine edge
taper define the continuous surface density. The primary aperture is 36 kpc;
24 and 30 kpc are fixed sensitivity cases.

The vertical profile is normalized sech-squared. Stellar height is
R_half/(1.678 × 7.3), giving 0.56024 kpc; each gas height is 0.2 kpc. The primary
integrated masses are 1.90591e10 solar masses in stars, 1.25005e10 in atomic gas
including helium, and 6.30309e8 in molecular gas including helium. These values
are conditional on the conversions, masking, geometry and finite aperture.
The JSON mass arrays follow the runner's in-memory order: stars, atomic gas,
molecular gas. Serialized profile dictionaries are key-sorted; do not infer
array labels from that sorted order.

## Numerical evidence and retained failures

The new adapter reconstructs a Newtonian potential from density, differentiates
its multipole spline through the Hessian, and supplies the physical Poisson
source to the existing scalar QUMOND operator. It sums ordinary source fields
before applying the nonlinear gravity response. Units are kpc, km/s and solar
masses, with G = 4.30091727003628e-6 kpc (km/s)^2 / solar mass.

Independent controls use spherical and flattened Miyamoto–Nagai solutions, the
exact spherical scalar relation, reflection symmetry, a finite-difference
derivative check, vertical mass normalization and
[Freeman's exponential-disk solution](https://articles.adsabs.harvard.edu/pdf/1970ApJ...160..811F).
The latter compares a finite-height disk to a zero-height analytic disk: its
2.985% maximum difference at h/R_d = 0.02 includes physical thickness effects,
not only numerical error. The thinner sequence approaches the analytic force.
For the flattened analytic source, force RMS error is 0.01488%, Hessian RMS
error is 0.00576%, and reconstructed-source versus analytic-source QUMOND force
RMS difference is 0.01420%. The focused suite has 135 passing tests.

| Retained run | Map refinement | Maximum force change | Interpretation |
| --- | --- | ---: | --- |
| map-source-001 | 256 → 512 pixels | 13.3035% | Solver-only pass; map error was not yet gated. Not source convergence. |
| map-source-002 | 512 → 1024 pixels | 4.1298% | Fails the subsequently frozen 3% map target. |
| map-source-003 | 1024 → 2048 pixels | 0.3439% | Passes the same 3% map target. |

The final coarse-to-fine field change is 0.2460%, below its 1% target. The
largest discrepancy between integrated surface mass and far-field monopole
flux is 0.000844%, below its 0.3% target. These statements apply at the six
predeclared radii 2, 4, 8, 12, 16 and 20 kpc and the tested grids, not all space.
All three results, all failed controls and all exact source/code/config
snapshots remain append-only. All 66 input snapshots and seven cached map
payloads were verified against their recorded hashes.

## Source uncertainty exceeds numerical uncertainty

At 20 kpc, reducing the source aperture from 36 to 24 kpc increases the inward
Newtonian force by 10.34%; using 30 kpc increases it by 1.76%. Exterior disk
material pulls outward on an interior point, so cutting it can increase net
inward acceleration. This is a geometric Newtonian effect. It demonstrates why
a spherical enclosed-mass shortcut is not a valid disk replacement.

Finite original-pixel coverage is incomplete. In the annuli centered at
15.75 and 19.75 kpc, CO coverage is 54.1% and 6.49%; it is zero at 23.75 kpc.
Stellar coverage is 58.8% at 29.75 kpc and 36.7% at 35.75 kpc. Atomic-gas coverage
is complete at those sampled annuli. These fractions describe map coverage,
not uncertainty or an observational assertion of zero material. The inherited
builder fills masks and blanks missing samples, clips negative intensities,
uses a CO significance cut and an approximate face-on beam convolution.
The CO map remains coarser than the selected HI beam. Warps, noncircular motion,
stellar population gradients, exact detector convolution, distance/inclination
covariance and missing outer material are not resolved by grid refinement.

## Next gravity test and claim ceiling

Freeze a development response contract and source sensitivities, then evaluate
the same nine scalar shape/acceleration combinations used for clusters and the
Solar System. Include Newtonian baryons and the established RAR comparator,
report every retained radial residual, and avoid per-regime parameter fitting.
SPARC circular velocities use gas kinematics; they test the gravitational field
experienced by orbiting matter, but are not direct measurements of outer-star
velocities. [SPARC measurement context](https://arxiv.org/abs/1606.09251)

Nonlinear-source, outer-boundary and full radial-response convergence must pass
before scoring. One galaxy remains a development pilot. The cluster/local
tension, multifield external-boundary calculation, photon coupling, stability,
and independent population validation remain open. No new formula is promoted.

## Reproduction

Run `scripts/run_gravity_map_source.py` with the corresponding versioned
`configs/gravity_map_axisymmetric_source_v*.json`, an unused output directory,
and `--source-checkout` pointing to the original local checkout containing the
seven inventory-bound maps. The runner verifies each hash before copying into
the ignored research cache. It never accesses other galaxies' map payloads.
The full source adapter and analytic controls are covered by
`tests/test_gravity_reconstructed_axisymmetric.py` and the gravity CI job.

Final result SHA-256:
`042cc43e16ca2ae43e2aa2731a0fe4f97eb36070b80913489784b8df0982153b`.
Predecessor hashes are recorded in their immutable receipts; the unresolved
second run is not overwritten or relabeled as a physical gravity failure.
