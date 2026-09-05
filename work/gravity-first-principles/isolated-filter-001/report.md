# Isolated Helmholtz filter checks

Implemented the one-dimensional isolated kernel exp(-|x-y|/L)/(2L) using exact kernel weights per cell, zero source extension and linear FFT convolution. This avoids the periodic wraparound used in the first action checks.

Eighteen cases span three filter lengths, three grid spacings and two box sizes. At spacing 0.01, the worst relative discrepancy against independent whole-line integration at the specified probes is 2.18e-5. Errors and the interior derivative-commutation residual decrease approximately fourfold when spacing is halved, consistent with second-order convergence. Doubling the box gives negligible change at the tested probes.

The maximum relative adjoint residual, comparing the discrete integrals of f*Sg and Sf*g, is 3.82e-16. This matters because the force derivation requires the filter's adjoint. It does not establish every boundary implementation as valid.

The exact positive kernel preserves positivity. FFT evaluation produced tiny negative values in remote tails, down to -6.85e-17, which are retained rather than clipped. No source positivity or gravity-force admission is based on those tails.

These are one-dimensional filter controls for decaying test functions, not a full isolated action calculation, a three-dimensional field solution, or a precision claim beyond the sampled grids. Next checks must establish the radial/tensor operator and the nonlinear product variation with the actual isolated boundary conditions.

Evidence: `isolated-filter-001`, including the kernel, all numerical settings, independent integrals, adjoint residuals, box comparisons and derivative checks. No observations were scored and no gravity theory was admitted or excluded.
