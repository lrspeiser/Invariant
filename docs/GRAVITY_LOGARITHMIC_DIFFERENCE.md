# Direct logarithmic calculation of the length-dependent flux

An exact rearrangement of the existing action avoids subtracting nearly equal
flux coefficients and avoids integral quadrature. It passes all 972 retained
actual-source sample checks, including 252 exact-zero controls and 720 nonzero
comparisons against 80-digit arithmetic. The worst nonzero relative error is
1.263e-12. Lint and 278 focused tests pass, including 32 new high-precision,
rotation, batching, zero-limit and physical-unit controls.

This adds a numerical implementation of J(ell)-J(0); it introduces no new
physical parameter and leaves the original full-action solver available. The
earlier failed fixed-quadrature prototype and successful sampled hybrid replay
remain preserved. No new galaxy force run or observational score is claimed.

## Exact rearrangement

With u=x+h and k1=u K'(u), the action coefficient obeys

    P_x - 1 = E'(u) - (h/u) k1,
    delta P_x = E'(x+h)-E'(x) - (h/u) k1.

Let v=x+epsilon^2, m be the existing shape, and c=1+3/(4m). Then

    E'(x) = v^(-1/4) (1+v^m)^(-c).

The implementation calculates the logarithm of E'(x+h)/E'(x) using log1p,
then evaluates the difference with expm1. The softplus difference is computed
through an equivalent logarithmic identity for small shifts, with an overflow-
avoiding expression for large shifts. No physical law is switched at that
floating-point implementation boundary. The existing full second-gradient
reaction term is included, with zero-length and stationary-point limits
handled without division by zero.

The independent checks differentiate the closed-form kernel in high precision;
they do not use this rearrangement as their reference. Synthetic tests include
both ordinary and very small gradients, lengths spanning ten orders of
magnitude, all three shapes, rotations and non-unit acceleration scaling.

## Next calculation and limits

Use this flux difference as the source of its own Poisson solve. That will
avoid another subtraction of nearly equal final fields. Compare the result
against the saved full-field subtraction, then refine angular and radial
resolution relative to the small signal itself. Retain all parameter cards,
both thicknesses, distance assumptions and negative numerical results.

These sample and code checks are not a uniform-domain error bound, a proof of
observational detectability or a new first-principles law. Physical source
uncertainty, other registered source scenarios, cluster dynamics and lensing,
Solar System predictions, direct outer-star measurements, stability and
independent validation remain open.

## Evidence

- `source-cancellation-003`: `49d7cd269833880f4694ae831c153f952a8c467cb0b21ebf2c766d2387e49d2f` (46 verified input snapshots)
- `tensor-controls-010`: `722bf8471012c072ba3d43965c57d0831eec304a0739ece716ce8da1681e9d63` (137 verified input snapshots)
