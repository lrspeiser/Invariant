# Galactic external-field audit of the bounded scalar actions

## Result and scope

Twelve of eighteen declared scalar scenarios fall outside a historical Cassini
quadrupole summary screen, although the same kernels' isolated-Sun precession
corrections are tiny. All six scenarios at a0=5e-11 m/s^2 remain inside that
screen. The calculation therefore constrains the shared parameter choices; it
does not reject the entire scalar families or establish their compatibility with
galaxies, clusters, or the full Solar System.

The three scalar QUMOND cards are evaluated. The 36 TRIMOND successor cards are
explicitly **unsupported by this external-field solver**. A Galactic field breaks
the spherical collinear branch, so those cards cannot borrow the scalar result.
The calculation also does not cover the older higher-derivative actions.

The new implementation reproduces twelve rounded published reference magnitudes
and compares two integral representations with different integrands. All 99
focused tests and lint pass locally. Remote CI and a full repository test run are
not claimed.

## Observational summary, fixed scenarios, and interpretation

[Hees et al., 2014](https://arxiv.org/abs/1402.6950), table I and equation 12,
report Q2=(3 +/- 3)e-27 s^-2, with the quoted uncertainty described as one sigma.
This audit declares a two-sigma summary screen of [-3,9]e-27 s^-2. It is a
historical published-summary development comparison, not our own likelihood,
new statistical significance, current-best-constraint claim, or reanalysis of
the Cassini ranging data. The source PDF and table were visually verified; its
hash is pinned in the configuration.

The three a0 values are the same illustrative scale sweep used in the earlier
monopole audit. They have not been fitted to galaxies. The two physical Galactic
accelerations, 1.9e-10 and 2.4e-10 m/s^2, are published scenarios from
[Hees et al., 2016](https://arxiv.org/abs/1510.01369), not a present uncertainty
interval. The conversion from physical to Newtonian external acceleration uses
eta_N*nu(eta_N)=eta, where eta=g_external/a0. This assumes the standard scalar
external-field mapping; it is not a reconstruction of the Galaxy's mass and
potential. Conclusions are conditional on these inputs.

Predicted Q2 in units of 1e-27 s^-2:

| Shape m | a0, m/s^2 | g_external=1.9e-10 | g_external=2.4e-10 | Declared summary screen |
|---|---:|---:|---:|---|
| 1/2 | 5e-11 | 2.20 | 1.80 | Both within |
| 1 | 5e-11 | 1.57 | 0.99 | Both within |
| 2 | 5e-11 | 0.50 | 0.23 | Both within |
| 1/2 | 1.2e-10 | 13.42 | 12.31 | Both outside |
| 1 | 1.2e-10 | 23.01 | 17.17 | Both outside |
| 2 | 1.2e-10 | 25.77 | 13.19 | Both outside |
| 1/2 | 2e-10 | 29.32 | 30.20 | Both outside |
| 1 | 2e-10 | 43.75 | 58.18 | Both outside |
| 2 | 2e-10 | 37.46 | 61.57 | Both outside |

The result illustrates why a fast high-acceleration transition is insufficient:
the distant, nonspherical response to the combined Sun and Galactic field can
produce a tidal anomaly near the Sun. This is an established MOND external-field
effect, not a newly discovered mechanism. Making the local tail steeper does not
guarantee a smaller external quadrupole.

## Equations and independent representation

The candidate response is unchanged from the bounded-action experiment:

\[
\nu_m(y)-1=(y^2+\epsilon^2)^{-1/4}
 [1+(y^2+\epsilon^2)^m]^{-1-3/(4m)},\qquad
 m\in\{1/2,1,2\}.
\]

Here y=|grad psi|/a0. Set R_M=sqrt(GM/a0), v=R_M/r, and
w=sqrt(eta_N^2+v^4+2 eta_N v^2 xi). Equation 12 of Hees2016 gives

\[
q=\frac32\int_0^\infty dv\int_{-1}^1d\xi\,[\nu(w)-1]
 [\eta_N(3\xi-5\xi^3)+v^2(1-3\xi^2)].
\]

The signed convention is

\[
Q_2=-\frac32 q\,\frac{a_0^{3/2}}{\sqrt{GM}},\qquad
\Phi_{\rm an}=-\frac{Q_2}{2}r_i r_j
 (e_i e_j-\delta_{ij}/3).
\]

Thus the anomalous acceleration tensor is Q2*(e e^T-I/3). It is symmetric,
traceless, and unchanged by e to -e. Negative q corresponds to positive Q2.
Table B1 displays positive reference magnitudes; the implementation keeps those
magnitudes separate from this signed convention.

For an independent integrand, write the dimensionless Newtonian potential as
psi=-1/r-eta_N*z, and p=grad psi. Outside the central source the induced Poisson
source is S=nu'(w)*p.H(psi).p/w, with H(psi)=(I-3nn^T)/r^3. Differentiating its
Newtonian Green function at the origin gives

\[
q_{\rm source}=-\frac12\int_0^\infty dv\,v^2
 \int_{-1}^1d\xi\,\frac{\nu'(w)}{w}
 [w^2-3(v^2+\eta_N\xi)^2](3\xi^2-1).
\]

This is an independent analytical representation within the same implementation,
not a separate research group's replication. The angular coordinate is
xi=-cos(theta); the sign conversion is explicit. The saturated derivative is
checked by finite differences, and the known nu_alpha responses with alpha=2,4,8
provide controls distinct from the proposed kernels. The twelve controls span
physical eta=1,1.5,2,3 and match the published rounding intervals.

## Numerical checks and reproducibility

Both integrals use Gauss-Legendre quadrature in angle and radius. The radial
domain is split at v=sqrt(eta_N); the second interval maps to infinity, avoiding
a finite radial cutoff. An angle-independent response term is subtracted from
the first integrand. Its weighted angular bracket integrates to zero, reducing
cancellation error without changing the integral.

Each of the eighteen scenarios uses 128, 256 and 512 nodes in each dimension.
At 512 nodes, epsilon=1e-4 is compared with 1e-5 and 1e-6. Measured maxima are:

- Last refinement change in signed q: 1.92e-10.
- Difference between the two integral representations: 6.68e-7 in q.
- Change from the epsilon sensitivity checks: 3.14e-9 in q.

These are empirical checks, not certified integration error bounds. The
source-Hessian representation converges more slowly near the Newtonian saddle.
All scenarios are farther from the summary-screen edges than the observed
numerical spreads. No extra precision in the table is implied.

Run from the repository root with a fresh output directory:

```text
python scripts/run_gravity_external_quadrupole.py --output <new-directory>
python -m pytest tests/test_gravity_extensions.py tests/test_gravity_local_limits.py tests/test_gravity_saturated_actions.py tests/test_gravity_external_quadrupole.py -q
```

Configuration: `configs/gravity_external_quadrupole_v1.json`. It pins source
URLs/PDF hashes, scenarios, comparison meaning, and numerical tolerances. The
configuration was fixed before the full scan but after examining reference
values and prototype calculations; this is development, not blind confirmation.
The runner downloads nothing and accesses no raw or sealed observations.

Completed evidence: `work/gravity-first-principles/external-quadrupole-002/`.
Result SHA-256:
`e10b1f4e00bad3a932188976f156b144ac2471ba54ba2e0e66375b396a8824c1`.
Code/configuration hashes and runtime versions are retained in the receipt set.
The first attempt, `external-quadrupole-001`, is retained with an execution-failure
record: the report writer requested the wrong action-card identifier key. It
made no physical rejection. The corrected runner uses the content SHA-256.

## Research consequence

The lower-a0 scalar region is a candidate for cross-regime testing, not a selected
winning constant. The same a0 must predict galaxy outskirts and clusters without
object-specific adjustment. An isolated, same-source field solver is still
needed for that transfer; periodic synthetic scenes cannot supply it.

TRIMOND needs its own external auxiliary boundary condition and nonspherical
solution. The already verified quadratic coupling identity may help organize
that calculation only after those conditions are defined consistently. A scalar
quadrupole cancellation must be calculated, not assumed from the spherical
identity or fitted independently of the galaxy/cluster coupling.

There is still no derived relativistic matter/light law, global stability proof,
joint three-regime validation, untouched successful prediction, or established
historical novelty. The discovery goal remains active and unachieved.
