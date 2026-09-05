# Direct force confirmation of the compact response

The strongest negative projected response from the positive-source perturbation scan is reproduced by directly evaluating the nonlinear spherical force for both perturbation signs. The force projection is -2.48200675221, compared with -2.4820067521423117 from the action Hessian. Across both source integration orders, both projection orders, and both positivity-bounded amplitudes, the worst relative discrepancy is 2.59842e-10, below the predeclared 1e-5 tolerance.

This confirms the full force calculation, including the outer vector filter, against the energy-based result. The case was selected from an exposed diagnostic scan; it is a numerical cross-check, not independent observational validation.

The perturbation redistributes matter only within 1<r<3, conserves total mass, and preserves positive total density. Its Newtonian potential and force changes vanish outside that region. The nonlocal gravitational response does not vanish there: it extends both inward and outward with a decaying tail.

| Radius | Newtonian force derivative | Model force derivative |
|---:|---:|---:|
| 0.5 | 0 | -0.0008783368 |
| 0.9 | 0 | -0.02960018 |
| 1 | 0 | -0.07332067 |
| 1.5 | -3.148603 | 8.682946 |
| 2 | 0.76 | -2.204855 |
| 2.5 | 3.438803 | -16.38151 |
| 3 | 0 | 0.04102502 |
| 3.1 | 0 | 0.01459022 |
| 3.5 | 0 | 0.0002358434 |
| 5 | 0 | 5.008068e-11 |

These are derivatives with respect to the perturbation amplitude, not total forces. Multiply by the very small registered amplitude to obtain an actual change. A negative entry means a decrease of the inward force under the chosen perturbation sign; it does not mean that the total force points outward. The last, tiny tail value should not be interpreted as an independently precision-validated measurement.

The model therefore responds to a spherical internal redistribution even where Newtonian gravity predicts no force change. This is a useful discriminating property of the nonlocal action. It has not yet been tested against measured objects or incorporated into a dynamical equilibrium calculation. The static action still lacks a derived photon sector and complete cross-regime validation.

Next: determine whether physically supported equilibria exist for this action and whether their perturbations grow, while retaining the observed sign reversal and nonlocal tails as model predictions. No candidate was admitted and no astronomical score was produced.

Evidence: compact-direct-force-001/result.json; SHA-256 a70aab919707c83b80932b3f8787413f6d39854e2d5b1629a2ba2516f4e793a3.
