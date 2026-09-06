# Fixed-normalization strength sweep

The requested bounded sweep is complete: epsilon0.005 and0.05 received12 new
integrations across co-rotation, counter-rotation and flyby, at both resolutions.
Six epsilon0.5 integrations were explicitly reused from the prior repaired run;
their source file is hash-bound. Mref=1, geometry, velocities, softening, duration
and thresholds were unchanged. No observed data were fitted. The epsilon0
control returns Newton exactly; subdivision checks pass at all four strengths.

| Strength | Co-rotation behavior by T40 | Counter-rotation behavior by T40 |
|---|---|---|
| 0.005 | Converged, radii0.969–2.019 | Converged, radii0.999–2.001 |
| 0.05 | Long trajectory fails resolution gate | Converged, radii1.000–2.202 |
| 0.5 | Converged dispersal, maximum radius178.549 | Converged dispersal, maximum radius177.158 |

The intermediate co-rotation run has base/fine final-position difference0.0141,
above the frozen0.001 limit. Its apparent maximum radius5.175 is preserved in
the result, but is not admitted as a precise physical prediction. All invariant
checks still pass. No extra retries or hidden strength choices were used to
replace this failure.

**The repair and the strength effect are separable.** The partition defect is
fixed at every tested strength. Reducing strength to0.005 avoids the large
dispersal over this interval. That does not establish useful extra gravity:
at the same initial state, mean inward acceleration is about0.990 of Newton
for0.005,0.919–0.920 for0.05, and0.776–0.781 for0.5. These kinetic modifications
initially weaken that inward response; altered later trajectories must not be
misdescribed as a universal stronger halo pull.

Initial nested total energy is-0.740 at0.005,-0.588 at0.05 and+0.929 at0.5.
The large-strength positive-energy configurations and dispersal reflect changed
mechanics at fixed velocities, not energy nonconservation. Positive energy is
not itself proof that every particle escapes. No modified equilibrium was
constructed or silently substituted.

Flyby trajectories converge at every strength. Their sampled closest source
separations are0.3170,0.3031 and0.2174, respectively, versus Newton0.3182. The
initial force at the large initial separation is almost Newtonian, while the
close passage is affected more. This establishes a scale- and motion-dependent
toy deflection, not an observed lensing or stellar-passage result.

All new fine energy drifts are below2.55e-9, and all conservation checks pass.
The weak-coupling regular-orbit case is a useful mechanical candidate for further
equilibrium/refinement work. The intermediate co-rotation trajectory remains
unresolved. Neither this bounded sweep nor the partition repair supplies a
galactic normalization, a continuum proof, relativistic causality, or a measured
gravity enhancement. The next step should define those missing physical inputs,
not continue selecting strengths until an illustrative orbit looks favorable.
