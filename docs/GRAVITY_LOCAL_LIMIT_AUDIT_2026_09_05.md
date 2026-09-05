# Local-limit audit: the shared baseline has an excessive monopole tail

## Finding

Thirteen of the fifteen cards in the newly merged extension inherit a very large
isolated spherical perihelion anomaly. They exceed a published Solar System
development screen at each tested universal scale. The two length-sensitive
cards are **unsupported by this calculation**, not falsified.

This finding changes the next experiment: replace or derive a different shared
high-acceleration response before searching the same auxiliary parameters. It
does not exclude QUMOND or TRIMOND as broad theory families, and it is not a full
ephemeris fit or an independent confirmation test.

## Derivation from the actual action

The merged action uses

\[
Q(x)=x+\frac{4x}{3(x+\epsilon^2)^{1/4}},\qquad
x=(g_N/a_0)^2.
\]

Its derivative gives the isolated, regular spherical response:

\[
\nu-1=Q_x-1=\frac{x+4\epsilon^2/3}{(x+\epsilon^2)^{5/4}}.
\]

At planetary accelerations the regularizer is negligible. With zero regularizer,
the expressions below are exact:

\[
g=\frac{GM}{r^2}+\frac{\sqrt{GM a_0}}{r},\qquad
\Phi=-\frac{GM}{r}+\sqrt{GM a_0}\ln(r/r_0).
\]

The arbitrary reference radius adds a constant potential and has no force effect.
We impose zero anomalous central integration mass. Any constant shift of central
GM is separately invisible to this first-order perihelion statistic, as tested.

The logarithmic term gives a flat asymptotic circular speed, but it persists into
the planetary domain. A limit `Q_x -> 1` does not test how rapidly it approaches
one. In the regular radial TRIMOND branch, `grad chi = s grad psi`, the auxiliary
flux vanishes and the physical flux is exactly `Q_x grad psi`, independent of
mixing, beta or power. Changing those coefficients cannot fix this branch's tail.

With eccentricity e, semimajor axis a, and s=sqrt(1-e^2), the radial Gauss equation
gives the first-order perihelion displacement per orbit:

\[
\Delta\omega=\frac{1}{e}\int_0^{2\pi}(\nu-1)\cos f\,df
=-2\pi a\sqrt{\frac{a_0}{GM}}\frac{s}{1+s}.
\]

The negative sign is a retrograde anomaly. The implementation evaluates the
finite-epsilon derivative numerically and cross-checks the closed logarithmic
formula and a separately coded nonperturbative Binet orbit integration.

## Published comparison and scope

The numerical reference is **INPOP10a Table 5**, whose intervals were determined
using less than 5% degradation of postfit residuals. They are not Gaussian sigma
errors. The paper simultaneously adjusted nuisance parameters when testing
supplementary precessions, but did not fit these new candidate actions. We use
this historical published result as a gross-mismatch screen, not as a current
best ephemeris, likelihood, confidence level or discovery criterion.
[Fienga et al., 2011](https://arxiv.org/abs/1108.5546), printed page 7 and section 4.3.

Geometry uses approximate J2000 elements; units and solar GM use JPL's published
constants. This mixture is adequate for the many-orders-of-magnitude diagnostic,
not for precision ranging predictions. [JPL elements](https://ssd.jpl.nasa.gov/planets/approx_pos.html),
[JPL constants](https://ssd.jpl.nasa.gov/astro_par.html).

No raw observations, response-bearing reserve products or sealed data were opened.
The public summary numbers now count as development information in this search.

## Results at illustrative a0 = 1.2e-10 m/s^2

The scale is a scenario input, not a fit or a measured interval from this work.

| Orbit | Predicted anomaly, arcsec/century | Published interval, mas/century |
|---|---:|---:|
| Mercury | -14,655.460 | [-0.2, 1.0] |
| Venus | -10,838.074 | [-1.3, 1.7] |
| Earth-Moon barycenter | -9,217.139 | [-1.1, 0.7] |
| Mars | -7,451.155 | [-0.19, 0.11] |
| Jupiter | -4,038.781 | [-83, 1] |
| Saturn | -2,982.723 | [-0.5, 0.8] |

One arcsecond equals 1,000 milliarcseconds. The Mercury anomaly is about 14.7
million times the largest absolute endpoint of its published interval. The
independent Binet value is -14,652.984 arcsec/century, differing from first order
by 0.0169%. Across the full six-orbit, three-scale sweep the largest first-order
versus direct integration difference is 0.493%; it cannot erase the discrepancy.
Increasing quadrature nodes from 64 to 128 changes the first-order result by at
most 1.34e-15 fractionally.

All 18 scenarios (a0=5e-11, 1.2e-10, 2e-10 m/s^2 for six orbits) exceed their
comparison intervals. Only in the unregularized first-order monopole comparison,
Mercury would require a0 <= approximately 2.23e-26 m/s^2 to reach the negative
interval endpoint. This is a **conditional diagnostic ceiling**, not a fitted
physical constraint from a full model. Rescaling the same coefficient also
rescales the asymptotic galaxy speed through `v_flat^4 = GM*a0`; it cannot be
treated as a planet-specific nuisance parameter.

## What faster tails establish, and what they do not

As diagnostic controls, we computed `nu-1=(a0/g_N)^p` with coefficient one and
p=0.5, 1, 1.5, 2. At the central a0 value, p=0.5 and p=1 miss all six intervals;
p=1.5 misses Mars and Saturn; p=2 falls inside all six. At p=2, Saturn's predicted
monopole anomaly is about -0.0298 mas/century.

These tails are not complete action functions or new theory claims. They do not
establish a disk fit, a Solar System pass or a cluster solution. A rapidly
Newtonian local interpolation can still produce an external-field quadrupole:
[Milgrom, 2009](https://arxiv.org/abs/0906.4817). Prior joint galaxy/Cassini studies
already constrain transition families: [Hees et al.](https://arxiv.org/abs/1510.01369).

## Evidence, tests and next action

- Module: `src/invariant_gravity_extensions/local_limits.py`.
- Reproduction: `python scripts/run_gravity_local_limits.py --output <new-directory>`.
- Source and parameter manifest: `configs/gravity_local_limit_audit_v1.json`.
- Current receipt: `work/gravity-first-principles/local-limit-002/receipt.json`.
- Result SHA-256: `8488d60b7cdf051511990066da3c097db947a84c50538920a8b71977205c5b1e`.
- Sixteen new tests pass; all 46 predecessor tests still pass. Lint passes.
- Tests include two analytic perturbations, a GM-shift null, the action derivative,
  sign mutation, direct-orbit truncation scaling and refusal to apply the monopole
  shortcut to a higher-derivative action.

Run 001 is retained; run 002 uses a numerically stable hypotenuse evaluation of
the same derivative that also handles extreme acceleration inputs. No predecessor
card, action, source seal or synthetic receipt was altered.

The next experiment must construct a **versioned action-level successor** with
a fast local limit while retaining the low-acceleration scaling, conserved joint
field dynamics, and nonspherical interactions. Derive and test its variations
before comparing galaxies or clusters. Record any free interpolation choice as an
ansatz requiring physical justification; changing an interpolation is not itself
deriving a first principle. The length-sensitive branch separately requires the
spherical double-divergence contribution and a universal dimensionful length.

Remaining completion gaps include external-field and multipole effects, isolated
cluster/member boundaries, root-data forward models, a covariant matter/photon
coupling, stability, and untouched cross-regime predictions. The discovery goal
remains active and unachieved.
