# Compact perturbations of positive finite sources

The filtered action has now been tested on 72 compact, mass-conserving source perturbations. Six have a negative projected static response; all occur at the two lower tested oscillation frequencies. This confirms that a sign reversal can occur beyond the frozen-background approximation. It is not a finding that the entire background force becomes outward, or a proof of dynamical instability.

The filtered-potential perturbation is (1-(r-2)^2)^6 cos(k(r-2)) on 1<r<3 and zero outside. Its derivatives through order five join continuously at the support boundary. The Newtonian-potential perturbation is delta_psi=(1-L^2 Laplacian)delta_chi. It too is compact, and its Laplacian integrates to zero total mass. A conservative bound on the polynomial and trigonometric derivatives sets each amplitude so that both signs of the perturbed density stay at least 90% (or 99% for the second amplitude) of the positive background density everywhere. The actual sampled perturbations are smaller than these conservative ceilings.

The response is the spatial integral of delta_grad_psi dot delta_J, divided by the integral of |delta_grad_psi|^2. It is evaluated through the exact action Hessian, retaining the mixed gradient/curvature term and both radial and tangential Hessian components. The verified isolated variational identity moves the outer filter onto the perturbation, whose filtered form is known exactly. Thus this is a full-space integrated response, not a local plane-wave substitution. It is not a dynamical eigenvalue.

| Frequency k | Minimum response | Maximum response | Negative cases |
|---:|---:|---:|---:|
| 2 | -1.402518 | 4.536254 | 3 |
| 8 | -2.482007 | 7.771646 | 3 |
| 32 | 0.2286922 | 3.640214 | 0 |
| 128 | 1.22087 | 2.327943 | 0 |

The strongest negative response is -2.4820 for L=0.1, ell=1, shape=0.5, k=8. Its first-gradient contribution is +1.4780, mixed contribution -0.2233, and curvature contribution -3.7367. The regular positive background has G=M=a=a0=1. Parameter values are diagnostic, not fitted astronomical constants.

Doubling integration nodes from 512 to 1024 changes the response by at most 4.32987e-15 on a max(1,|response|) scale. Central differences of the first variation at both positivity-bounded amplitudes agree with the analytic Hessian within 1.71711e-09, passing the predeclared 1e-5 tolerance. This check differentiates the first variation rather than subtracting nearly equal total energies.

At the two highest tested frequencies all projected responses are positive; the previous analytic high-frequency limit remains a separate result. This finite set cannot prove bounded behavior for every perturbation or stability of a physical equilibrium. Negative static response can be restoring under some matter closures. The next step needs a specified equilibrium and matter dynamics, or a direct force-profile response test, before assigning a physical stability verdict. No observations were scored and no candidate was admitted.

Evidence: positive-compact-perturbations-001/result.json. SHA-256 be76597ea5177a7d1823964b2fe8bc709f3864d4d4209c57f21c71ee02c240b3.
