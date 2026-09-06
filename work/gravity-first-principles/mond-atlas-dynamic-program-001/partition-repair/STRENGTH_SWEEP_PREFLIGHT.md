# Bounded post-repair strength sweep

THEORY_BENCHMARK_ONLY, frozen before new integrations. Keep fixed Mref=1, L=1,
softening and all original initial states. Compare epsilon0.005,0.05,0.5 for
nested co-rotation, nested counter-rotation and flyby. Existing epsilon0.5 results
are reused explicitly with hashes, not counted as new integrations. The first
two strengths receive both original base/fine resolutions and durations. No
observational fitting or further strength search after outcomes.

The epsilon0 limit is an analytic Newton control. At each new strength test
co-moving half-mass subdivision and the Newton limit before integration.
Equations follow the same Lagrangian; massmatrix, force, extra kinetic energy
are affine in epsilon. The new wrapper interpolates these three quantities,
then solves acceleration, and does not scale an already computed acceleration.
Original conservation1e-5 and trajectory1e-3 thresholds apply.

Report initial modified energy, source-relative radius range, source-pass
closest approach, normalized acceleration at identical initial state and
subsequent mean inward acceleration separately. Positive total energy and
maximum radius beyond10 times its initial maximum are descriptive equilibrium/
dispersal flags, not proof of escape or a new detection. Circular Newton initial
conditions are not silently reinitialized as modified equilibria.
