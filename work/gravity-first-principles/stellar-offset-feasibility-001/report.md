# Small projected offsets change source feasibility

Tested the same five stellar profiles with every projected mass bracket retained. Added positive Plummer components with offsets 0, 0.005, 0.01 and 0.02 times the inherited R500 and scales 0.001–0.02R500 to the previous positive shell dictionary. Positions are source hypotheses, not measured galaxy locations. No gravity residuals entered the calculation.

| Cluster | Shell-only extra allowance | Shells plus offset components |
| --- | ---: | ---: |
| A1795 | 5.78% | 0% |
| A2142 | 0% | 0% |
| A2319 | 31.82% | 5.74% |
| A85 | 0% | 0% |
| ZW1215 | 0% | 0% |

Allowance means additional fractional mass outside the existing bounds. These values are optimization diagnostics, not statistical errors or significances. Baseline shell results reproduce the preceding campaign. Every new predicted profile was checked directly against its constraints.

The exact aperture mass fraction for a Plummer component of scale a, offset d, and projected aperture R is

    F = 0.5 * [1 + (R²-a²-d²)/sqrt(((R-d)²+a²)*((R+d)²+a²))].

A cancellation-resistant evaluation passed 36 independent quadrature controls, with maximum relative discrepancy 3.95e-16. This verifies the tested projection formula, not an inference of three-dimensional galaxy positions.

A1795's central conflict can therefore arise from enforcing a common spherical center. A2319 remains unresolved in this finite dictionary; it is not excluded as a physical source. Multiple offset components are allowed, and no claim is made that their fitted positions correspond to observed members. Annular masses alone do not constrain their position angles or line-of-sight locations. Such choices matter for the nonlinear full-field calculation.

Next source work must constrain centering and stellar substructure with independent source information, refine representation sensitivity, and replace singular shells with differentiable source models before using density gradients. None of these feasible mixtures is admitted as a complete gravitational source. No candidate law was scored or excluded.

Saved evidence: `stellar-offset-feasibility-001`, including the registered dictionary, hashes of both inputs, all ten source solutions, positive component masses, directly verified bracket violations and the 36 numerical controls.
