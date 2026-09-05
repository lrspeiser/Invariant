# Cancellation control on actual source samples

The fixed-quadrature flux-difference prototype passes 918 of 972 sampled cases
and fails 54. A combined method passes all 972 with the same 1e-9 target. Both
runs are retained; the production action and ongoing source-response run are
unchanged.

The samples cover all 54 existing cards, both thicknesses and nine positions:
the centre, axis, innermost radius, inner disk, two dynamical radii, taper region,
join region and exterior. There are 720 nonzero comparisons against 80-digit
kernel differentiation and 252 exact-zero symmetry or identical-law controls.
The zero controls do not require division by a zero reference norm.

All 54 fixed-quadrature failures occur at R=0.0001, z=0 kpc. At large
h/(x+epsilon^2), the integral's endpoint structure is not resolved by the fixed
16- or 32-point quadrature. The worst relative error is 0.792. Direct
subtraction is accurate in that exposed case because the change is substantial.

The combined method uses direct subtraction when h/(x+epsilon^2) exceeds 0.01,
and the integral identity for smaller shifts. This threshold is a numerical
conditioning choice, not a new parameter in the physical law. It uses the same
action on both branches and removes no samples. Its worst relative error is
1.70e-12; 180 cases use the direct branch. The two quadrature-order records on
the direct branch are identical and are not independent quadrature checks.

This is a sampled diagnostic, not proof of accuracy throughout the domain.
The branch transition, general small-gradient configurations and complete
solver-grid response still require validation. A direct Poisson solve of the
flux difference is also needed to avoid subtracting two nearly equal final
fields. Angular-resolution uncertainty remains separate. No observational
score, physical exclusion or new gravity law is established here.

Evidence: `source-cancellation-001` retains the failed fixed-quadrature
prototype; `source-cancellation-002` retains the combined-method replay. Each
contains all cases, exact input snapshots and the implementation used at run
time.
