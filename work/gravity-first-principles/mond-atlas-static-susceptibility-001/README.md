# One conservative modification produces static inward response

**Removing the old oscillator's cancelling counterterm gives a finite inward
force at equilibrium.** The modified equations passed the frozen mechanical
tests and sixteen two-body integrations. At the inherited fast oscillator
setting, the added force reaches about **3.70% of softened Newtonian gravity**
near the coupling length; at the slow setting it reaches about **0.037%**.
No strength, length or initial condition was fitted to observations.

The result is **THEORY_BENCHMARK_ONLY**. It demonstrates how an internal degree
of freedom can encode an attractive static potential with a complete energy
account. It neither makes time an energy source nor derives a gravity law from
first principles. The candidate still fails important gravity requirements.

## What the previous program actually tested

The audit read the original equations and all saved dynamic/strength results:

- Original program: 32 integrations; energy/invariant gates passed, but all
  four long clustered trajectories failed positional convergence. Four paired
  short-time tests subsequently passed without repairing the long trajectories.
- Corrected kinetic coupling: 20 integrations. Co-moving source partition was
  repaired, but the nominal strong coupling dispersed the nested configurations.
  Its long clustered Newton comparison still failed positional convergence.
- Strength sweep: 12 new integrations and six explicitly reused strong-coupling
  integrations. Weak kinetic coupling avoided large dispersal over the tested
  interval, but initially weakened inward acceleration; intermediate co-rotation
  still failed trajectory convergence.
- Original oscillator memory: its potential was a positive square whose minimum
  did not depend on separation. Therefore its equilibrated extra force was zero.
  Dynamic lag could exchange energy and change force sign, but could not supply
  a static extra potential in that formulation.

These old files are hash-bound in [run001/bindings.json](run001/bindings.json)
and remain unchanged. None is an observed-history validation.

## Minimal change and eliminated equilibrium

Keep the old Gaussian target and constants:

\[
h_{ij}(r)=\eta\sqrt{m_i m_j}\exp[-r^2/(2L^2)],\qquad
\eta=0.15,\quad L=1,\quad\omega\in\{0.2,2\}.
\]

The old internal potential was \(U_{old}=\omega^2(q-h)^2/2\). Change only

\[
U=\tfrac12\omega^2q^2-\omega^2qh
 =\tfrac12\omega^2(q-h)^2-\tfrac12\omega^2h^2.
\]

Retain ordinary particle kinetic energy, internal kinetic energy
\(\dot q^2/2\), and softened Newton potential with \(G=1,b=0.05\).
The oscillator equation remains \(\ddot q=-\omega^2(q-h)\). The particle
reaction force changes to \(\omega^2q\nabla h\), added to Newton's force.

At a fixed separation, \(q_{eq}=h\), and

\[
U_{eff}(r)=-\frac{\omega^2\eta^2m_i m_j}{2}e^{-r^2/L^2},\qquad
\mathbf F_{extra,i}=-\frac{\omega^2\eta^2m_i m_j}{L^2}
 e^{-r^2/L^2}(\mathbf x_i-\mathbf x_j).
\]

Writing the oscillator drive as \(f=\omega^2h\), the static susceptibility is
\(dq_{eq}/df=1/\omega^2\). The force is inward at this equilibrium.
Out-of-equilibrium states can still alter its sign and exchange energy.

This is the same static-potential change obtained by completing the square in
a linearly coupled harmonic coordinate. [Caldeira and Leggett (1983), §3,
equations 3.2–3.4](https://lab.semi.ac.cn/download/0.8104329777920175.pdf) explain
the eliminated potential and the counterterm that cancels it. That paper is
the mechanical reference; it supplies no support for this Gaussian gravitational
coupling. Removing the counterterm is a new physical assumption, effectively
adding the displayed attractive pair potential, not a consequence of conservation.

All calculations use the old dimensionless units. Restoring dimensions, q has
units square-root-mass times length, and eta has units length divided by
square-root-mass, so \(\dot q^2\) and \(\omega^2qh\) have energy units.
No galactic value for these units or oscillator frequencies has been established.

## Boundedness, splitting and Newton limits

Completing the square proves, for fixed masses,

\[
E\geq-\left(\frac{G}{b}+\frac{\omega^2\eta^2}{2}\right)
\sum_{i<j}m_i m_j.
\]

The Gaussian addition has no short-distance singularity. At zero separation,
its equilibrium potential curvature is \(\omega^2\eta^2m_i m_j/L^2>0\),
and the oscillator curvature is \(\omega^2>0\). Coincidence is a finite
potential minimum, not a model of a supported galaxy. Existing Newtonian
softening is essential to the total lower bound; removing it restores Newton's
own point-particle singularity, though the Gaussian addition stays finite.

Static mass dependence is bilinear. Splitting each source into coincident,
co-moving 0.3/0.7 children preserves child accelerations when cross-pair q and
qdot are scaled by the square root of the mass fractions. New internal child
pairs are initialized at equilibrium. Their constant self-energies are explicitly
subtracted in energy comparisons. This tests representation of the same internal
state; arbitrary independent excitations of new child modes are different physics.
Finite spatial cells and a continuum state measure remain unvalidated.

For ordinary unsoftened Newton gravity, the fractional extra force is

\[
\frac{F_{extra}}{F_N}=\frac{\omega^2\eta^2}{G L^2}
r^3e^{-r^2/L^2}.
\]

It vanishes at small r and at large r. With the actual fixed softening, the
ratio instead contains \((r^2+b^2)^{3/2}\) and approaches a small nonzero
constant as r tends to zero; both softened forces themselves vanish there.
Neither limit is an executed Solar-System test.

**A universal high-acceleration Newton limit fails.** At fixed separation,
increasing source mass from 1 to 1,000 to 1,000,000 increases both accelerations
in direct proportion and leaves the fractional correction unchanged. Large
acceleration alone therefore does not suppress the new effect.

**An extended flat-halo limit also fails.** The Gaussian force falls exponentially
beyond L; it does not maintain the approximately inverse-radius force needed
for a long flat circular-speed interval. Memory notation does not change that
static asymptote.

The fixed-total-mass bound is not thermodynamic extensivity. Coincident particles
of fixed individual mass contribute an energy proportional to N(N−1), not N.
Gravity itself is nonextensive, so this is not by itself a disproof, but it rules
out claiming ordinary bulk stability from the finite-particle bound.

Two naive shape changes were retained as analytic negative controls: h proportional
to 1/r makes the eliminated attraction diverge as −1/r² at the origin; h
proportional to r makes energy unbounded below at large distance and produces
an outward force. A mass-independent h also changes its total cross-coupling
by a factor four when two sources are each split in half. These are counterexamples,
not additional fitted candidates.

## Executed checks and trajectories

Before integrations, independent scalar minimization, potential-force finite
differences, energy-rate, reaction-force, covariance, source-splitting and
zero-coupling controls passed. Maximum force finite-difference discrepancy was
5.19e-11; source-split acceleration discrepancy was 2.23e-16, and corrected
energy discrepancy was 3.11e-15. A fixed-radius analytic oscillator agrees to
1.83e-11. Without damping, its motion **does not settle**: starting from q=0
gives \(q=h(1-\cos\omega t)\). An equilibrium preparation needs physical
history or a specified dissipation channel, neither supplied here.

Sixteen integrations used two masses, radii 1 and 2, both inherited frequencies,
and two numerical resolutions over T=40. One case starts on the exact modified
circular solution; another starts 3% slower. These are explicitly derived new
equilibria, not reruns of old Newtonian initial conditions or evidence of a
galaxy lifetime. All conservation and trajectory gates passed. Maximum
normalized energy drift across all runs was 7.36e-14; the exact circular
position error was below 1.94e-12. Signed internal energy and q versus h are
retained in [timeseries.csv](run001/timeseries.csv).

The coordinating agent independently replayed all eight fine cases using
reduced radial equations and conserved angular momentum with the implicit
Radau integrator, rather than the production Cartesian DOP853 equations.
Across 3,208 matched samples, maximum radius/q/energy/internal-energy discrepancy
was 2.06e-11. The separate [review receipt](independent-review/receipt.json)
records the successful comparison and preserves its own scope and implementation.

The next useful formula must address the failed radial/high-acceleration behavior
with an independently motivated interaction and then repeat these checks.
Extending a kernel until it looks like a halo would introduce new physics;
this result does not authorize treating such an extension as already validated.

Implementation: `scripts/mond_atlas_static_susceptibility.py`.
[PREFLIGHT.md](PREFLIGHT.md), [controls](run001/pre-integration-controls.json),
[static tests](run001/static-susceptibility.json), and
[integration results](run001/results.json) preserve all definitions and failures.
The runner refuses existing run001 output; reproduce in a separate checkout.
No observations, new raw data, fitted targets, shared edits or commits.
