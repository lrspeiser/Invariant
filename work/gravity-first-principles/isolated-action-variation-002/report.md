# Isolated spherical action and force consistency

The filtered-curvature action now has an isolated spherical variation check. With chi=S psi, a0=1, and Hr=chi second derivative, Ht=chi first derivative/r:

    x = (psi first derivative)^2
    h = ell^2 (Hr^2 + 2 Ht^2)
    w = (Ph Hr) first derivative + 2 Ph (Hr-Ht)/r
    J = Px (psi first derivative) - ell^2 S_vector w

Here S_vector is the regular, decaying l=1 Helmholtz inverse verified in the preceding vector control. The nonlinear product remains inside the outer filter. Px denotes the full derivative, including the Newtonian contribution. For a regular spherical gravitational solution, the radial physical potential derivative would equal J; the present test does not establish physical source admission.

Two independent representations are compared with finite differences of the energy integral:

    delta A = 2 integral r^2 [Px psi' delta_psi' + ell^2 Ph (Hr delta_Hr + 2 Ht delta_Ht)] dr
            = 2 integral r^2 J delta_psi' dr.

The common angular factor 4pi is omitted. Smooth Gaussian combinations define chi and delta_chi; constructing psi=(1-L^2 Laplacian)chi gives the exact isolated filtered potential. The Gaussian boundary terms vanish at infinity and regularity removes the origin terms. These manufactured potentials can imply signed source densities: they test the variation, not an admissible matter model.

The campaign covers three kernel shapes, three filter lengths, two curvature lengths, and two integration extents. Three finite-difference steps test the energy derivative. The initial run failed its predeclared 1e-4 relative force tolerance. Its results remain intact. The error decreases approximately fourfold when the quadrature order doubles, consistent with the Green kernel's derivative discontinuity at equal radii. A successor at 2048 nodes passes the unchanged tolerance in all 36 cases.

| Outer radius | Integration nodes | Worst relative force/variation error |
|---:|---:|---:|
| 20 | 256 | 0.0028054913 |
| 20 | 512 | 0.00070274855 |
| 20 | 1024 | 0.00017585904 |
| 20 | 2048 | 4.3986246e-05 |
| 40 | 256 | 0.0057605281 |
| 40 | 512 | 0.0014429253 |
| 40 | 1024 | 0.00036108203 |
| 40 | 2048 | 9.0314482e-05 |

The successor's worst relative force discrepancy is 9.0314482e-05; the smallest-step finite energy variation differs from the weak derivative by at most 6.9861797e-11. Omitting the outer vector filter produces a maximum relative discrepancy of 0.28424722. The two extents are a truncation diagnostic, not an independent physical boundary prior; at fixed node count a larger interval also reduces spatial resolution.

This establishes numerical agreement for the registered smooth fields. It does not prove general nonlinear stability, positive-source behavior, observational agreement, relativistic light bending, or a first-principles origin of the filter length. Next, apply the verified spherical response to finite positive matter distributions and test its force and short-wavelength behavior before using it for astronomical ranking.

Evidence hashes:
- radial-filter-002: 8fa829a30ff067aa94ecde7bf74f2a88e175c00797503765a42d83de4a8bb0ab
- radial-vector-filter-001: 04772fef046103402b1910508102fbaf8efad3a3e1f92b6bcc980320c5524773
- isolated-action-variation-001: 2ce0784888360865a12078b325244ee07a6aca44d54f542d6d7359ec40a524e0
- isolated-action-variation-002: fddd8637ec1d53d74509165839a140dc27e7d7ac70b9e8301da0f32057fa2e4d
