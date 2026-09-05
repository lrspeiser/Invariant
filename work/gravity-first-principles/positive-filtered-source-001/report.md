# First finite positive-source test of the filtered action

All 81 registered source/parameter combinations produced inward force at all 41 sampled radii. This is a limited positive result: the force calculation behaves sensibly on this source family, but it has not passed a stability test or astronomical comparison.

We choose an exactly filtered Plummer potential chi=-GM/sqrt(r²+a²), and construct the actual Newtonian potential psi=chi-L² Laplacian chi. Its source density is

    rho = rho_Plummer * [1 - 5 L² (4r²-3a²)/(r²+a²)²].

The bracket is bounded below by 1-20L²/(7a²), so the source is positive everywhere when L/a<sqrt(7/20). It is smooth, has finite total mass M, and has an isolated potential. For L/a=0.1, 0.3, 0.5 the lower bounds are respectively 0.9714, 0.7429, and 0.2857. These are exact positivity guarantees, unlike the sampled force check.

The scan uses G=a=a0=1, masses 0.01, 1, 100, three kernel shapes, curvature lengths ell=0.1, 1, 10, and the three filter lengths above. Source construction depends on L. Consequently this is a diagnostic family, not a comparison of different L values against the same observed object. The exact construction avoids numerical uncertainty in the inner scalar filter; the nonlinear outer vector filter is integrated with the previously checked l=1 kernel.

Force was sampled at 41 logarithmically spaced radii from 0.01a to 100a. The smallest force/Newtonian-force ratio was 0.59691612. No sampled force pointed outward. Doubling the quadrature order gave a worst difference of 7.0934264e-12, scaled by the larger of the final force magnitude and Newtonian force. Doubling the upper integration distance from r+40L to r+80L changed the result by at most 3.9477959e-16 on the same scale. These convergence tests do not establish a rigorous error bound or behavior at unsampled radii.

Next: test perturbations and the high-frequency response around these positive sources. An inward background force alone does not establish stable dynamics. The global-parameter galaxy/cluster/Solar System and light-bending requirements remain open; no observations were scored and no candidate was admitted.

Evidence: positive-filtered-source-001. Result SHA-256 821685c6d9bde9a262496d89e597aafd67e288037b381399f4a3312685483527; tail-check SHA-256 35fdea6ce5846b040ff73ef49d3de9ecd2ff92b4407ad58a5d8c5d0a51b5043b.
