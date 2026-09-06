# Distributed secondary-source result

Status: THEORY_BENCHMARK_ONLY. Four pre-evaluation tests passed. All 56 fixed
sample points passed the 1% base-to-fine numerical convergence gate; worst change
was 0.296%. These are manufactured source calculations, not observational fits.

## Formula tested

Each ordinary matter element dm contributes an additional pair potential

`dPhi_extra(s) = -G eta dm log(1+s/L)/s`.

The resulting additional acceleration is

`dg_extra = -G eta dm [log(1+x)-x/(1+x)] s_vector/s^3`, with `x=s/L`.

This corresponds to a positive extended effective-source kernel around every
ordinary matter element. Adding all these contributions is a convolution. It
does not select a special galaxy center and does not absorb or consume gravity.
It is a single-generation response, not an endlessly recursive relay.

A point source exactly reproduces an NFW field when
`eta=4 pi rho_s L^3/M`, here 13.40614208, with L=19.5725 kpc. The parameters
come from the author model associated with [McMillan 2017](https://arxiv.org/html/1608.00971).
The source receipt is at `work/gravity-first-principles/mond-atlas-halo-return-001/source-receipts.json`.
The matched point-source target is fitted-model calibration, not independent
support for this kernel or evidence that dark matter is absent.

## What the extended source changed

The source was a manufactured razor-thin exponential disk, mass 6e10 Msun and
scale length 3 kpc. Keeping the point-calibrated kernel unchanged gave:

| Radius | Extra force / spherical NFW target | Largest deviation from inward radial direction |
|---|---:|---:|
| 8 kpc | 72.95–74.11% | 7.95 degrees |
| 16 kpc | 88.09–92.10% | 3.70 degrees |
| 32 kpc | 95.63–98.74% | 1.32 degrees |
| 64 kpc | 98.64–99.89% | 0.40 degrees |

Ranges cover fixed directions 15, 30, 45, 60, 75 and 90 degrees above the disk
at the same radius. They are not confidence intervals. All angular samples also
passed the resolution threshold. The disk is axisymmetric and even in z;
symmetry supplies the corresponding lower half rather than observational data.

**Promising:** a flat distribution of ordinary sources naturally produces an
extra field that becomes approximately spherical far out, including above and
below the disk. Pairwise reciprocity, translation and rotation covariance,
superposition and potential-gradient checks passed. A compact-source far-field
test passed independently of the Milky Way normalization. This makes the kernel
a useful mathematical candidate for geometrical testing.

**Challenging:** distributing the sources changes the inner extra field by about
26% at 8 kpc. Point-source halo matching therefore cannot be substituted for a
galaxy calculation. A kernel inferred from one fitted halo also does not tell us
why a particular galaxy should have that length or strength. There is no density
suppression, absorption, storage, propagation delay, or independent source-only
rule in this experiment.

## The finite-budget problem

The integrated kernel weight grows logarithmically without bound. Relative to
the ordinary point source, enclosed effective-source weights at r/L=1,10,100,
1000 are 2.589,19.959,48.598,79.227. These are source weights, not energies.
The potential and force approach zero far away even though this integrated
weight diverges; these are distinct mathematical properties.

The predeclared sharp cutoff at 10L keeps the point-source force unchanged
inside that radius and makes the outer force fall as 1/r^2. At 100L it supplies
41.1% of the untruncated target force; at 1000L, 25.2%. A finite response reservoir
therefore cannot reproduce an infinite NFW tail exactly. A physically motivated
finite extent is a required choice, not automatically a rejection: observations
also cover finite regions. No extended-disk cutoff fit was performed.

## Directions to fix next

1. Freeze a rule for eta and L based on observable ordinary matter, then test
   untouched galaxies rather than setting each from its inferred halo.
2. Evaluate real stellar, atomic and molecular source maps, with their geometry
   uncertainties; the present disk is only a manufactured benchmark.
3. Test a finite smooth cutoff and its transition using outer tracers. The present
   sharp cutoff was an analytic diagnostic, not a preferred physical model.
4. If a density-dependent or time-dependent relay is added, recheck reciprocity,
   stability and the existence of a consistent potential or dynamical action.
5. Derive relativistic metric predictions before comparing lensing. This scalar
   force model alone does not specify photon deflection or spacetime dynamics.

No observed rotation, lensing, cluster or Solar System response was opened or
scored in this branch. There are no failed numerical points to hide. The report
does not establish that this response occurs in nature.

Reproduce in a fresh output directory after preserving the original result:
`python scripts/mond_atlas_secondary_experiment.py`. The runner refuses to
overwrite an existing results.json. Units in the CSV are kpc for positions and
(km/s)^2/kpc for accelerations. Tests can always be rerun using
`python -m unittest discover -s tests -p test_mond_atlas_secondary_experiment.py -v`.
