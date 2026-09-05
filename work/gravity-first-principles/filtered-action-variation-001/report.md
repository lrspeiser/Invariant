# A nonlocal action alternative with a bounded curvature contribution

Defined a self-adjoint spatial filter S=(1-L²*Laplacian)^(-1) and placed the filtered Hessian inside the action:

    x = |grad psi|²/a0²,
    h = ell² |Hessian S psi|²/a0²,
    P = x + x K(x+h).

Varying this action gives

    J = Px grad psi - ell² S div(Ph Hessian S psi),
    Laplacian Phi = div J.

The nonlinear product must be inside the outer filter. Filtering a local force afterward is not the same theory.

In a frozen-coefficient analysis, the curvature term in the transfer changes from ell²*B*k² to

    ell² B k² / (1+L² k²)².

For L>0, k²/(1+L²k²)² is at most 1/(4L²) and tends to zero at large k. This removes the specific unbounded k² transfer term previously identified. It does not prove full nonlinear stability or eliminate ordinary gravitational growth. The unfiltered local theory is recovered at L=0.

Eighteen periodic one-dimensional action-variation checks span three shapes, three filter lengths and two curvature lengths. The worst absolute discrepancy against central finite differences at the smallest step was 2.35e-10. Discrete translation-variation residuals were below 7e-18. Omitting the required outer filter produces discrepancies up to 4.37e-6. These checks support the implemented variation, not a general conservation theorem for every discretization or boundary treatment.

This is an exploratory static nonlocal ansatz, not a first-principles derivation of L, a novelty claim, a relativistic matter/light theory, or an observational success. L and ell would have to be global parameters or derived scales. Boundary conditions must preserve the adjoint relation; the periodic implementation does not establish isolated-source behavior.

Next: derive and verify isolated-boundary behavior and curved-background transfer for this action, then assess dynamics, source partition invariance and cross-regime predictions. Retain all failures of the local predecessor. No observations were scored and no candidate was admitted.

Evidence: `filtered-action-variation-001`, including the explicit filter and action, all 18 variation controls, translation residuals and the analytical high-wave-number bound.
