# Evolving currents and explicit memory: first dynamic program

This is a new THEORY_BENCHMARK_ONLY dynamics run, beyond earlier instantaneous
snapshots. We integrated all particles and their reactions, with an explicitly
defined energy account. No observed velocities were fitted. Length, time, mass
and oscillator parameters are dimensionless; none have been converted into
galactic scales or ages.

## Executed mechanisms

**Motion-dependent mechanical response:** an extra positive kinetic term depends
on relative particle velocities and separations. It changes both acceleration
and canonical momentum. The interaction matrix remains positive, and total
momentum and canonical angular momentum are conserved. This is a precise toy
for motion-dependent gravity-like behavior, not an established spacetime current.

**Memory with a real energy account:** each pair has a harmonic internal state q
whose equilibrium h changes as the particles separate. The particles can transfer
orbital energy into that state and receive it back. Two response frequencies,
0.2 and2, were fixed before testing. The equilibrium state q=h gives no additional
force: this particular memory model cannot simply supply a static extra halo.
Its force can change sign with phase. It is causal as an ordinary differential
equation but instantaneous across space, with no relativistic propagation law.

**Static control:** softened Newtonian pair gravity with exactly the same initial
particle positions and velocities. This helps distinguish new interaction
effects from ordinary changes of geometry. Modified and Newtonian initial
energies need not be equal; none of these comparisons claims otherwise.

**Reflection:** SOURCE_BLOCKED. No carrier, reflecting surface, force direction
law or reflecting medium has been defined. We did not pretend an optical analogy
supplies an implemented inward gravitational force or an energy budget.

## What happened over evolving orbits

Thirty-two integrations compared four laws, four source/motion configurations
and two numerical resolutions. Nested-orbit cases ran for40 time units, about20
nominal inner orbits. A passage past a moving source ran for12. This is a bounded
mechanics experiment, not a claim of stability over a galaxy lifetime.

| Configuration and law | Satellite distance range from central source | Largest oscillator energy |
|---|---:|---:|
| Nested co-rotation, Newton | 0.976–2.008 | 0 |
| Nested co-rotation, kinetic | 0.472–4.867 | 0 |
| Nested counter-rotation, Newton | 0.999–2.000 | 0 |
| Nested counter-rotation, kinetic | 0.689–2.055 | 0 |
| Nested co-rotation, fast memory | 0.954–2.054 | 0.01181 |
| Nested counter-rotation, fast memory | 0.999–2.000 | 0.000143 |

**A useful discriminator emerged:** the same nested geometry responds differently
to co-rotation and counter-rotation when the interaction depends on motion or
internal state. The kinetic example can strongly disturb the co-rotating orbits,
rather than providing a tidy universally stronger inward pull. Slow memory stays
very close to Newton for these prescribed cases. Fast memory transfers far more
energy in co-rotation than counter-rotation. That is a property of the chosen
coupling/frequency, not a prediction that co-rotation always strengthens gravity.

In the flyby, closest source separation is0.318 for Newton,0.223 for the kinetic
law and0.320 for fast memory. Fast memory reaches internal energy0.00756 while
the total initial system energy is0.03848. Orbital and stored energy can therefore
change appreciably without creating energy. These are dimensionless toy results;
they do not establish an actual gravitational absorber or relay.

## Numerical challenges retained

All fine integrations conserve energy, total momentum and canonical angular
momentum within the frozen1e-5 threshold; the worst normalized drift is9.16e-9.
Nested and flyby trajectories also pass the1e-3 final-position resolution gate.

**All four clustered-source trajectories fail long-time positional convergence.**
The base/fine differences atT40 range0.714–3.356 relative. This includes Newtonian
gravity. The precise final clustered positions and their apparent law rankings
are therefore not admitted quantitative predictions, despite small energy drift.

A separately frozen short-time diagnostic reran those same initial conditions
and equations toT2. All four pass, with positional differences2.90e-9–1.91e-7
and conserved invariants. This is consistent with accumulated trajectory
sensitivity rather than an immediate force error, but does not prove mathematical
chaos or rescue the unconverged long-time trajectories.

Pre-integration controls checked energy derivatives, boost/rotation covariance,
momentum, positive inertia and the analytic fixed-radius memory oscillator.
The independent finite-difference Euler-Lagrange force audit was completed after
the integrations, not before; its timing is explicitly recorded in the receipt.
All four equations pass with residual at most1.34e-10. No observational response
was scored before or after these checks.

## A concrete limitation before using simulation cells

The kinetic coefficient uses pair reduced mass m_i*m_j/(m_i+m_j). Consider two
equal co-moving source groups, each initially represented by one massm. Splitting
each into two coincident particles of massm/2 leaves the physical ordinary mass
and velocity fields unchanged, but doubles the sum of cross-group kinetic
couplings: one pair has reduced massm/2; four pairs each havem/4. Internal
co-moving relative-velocity terms vanish. Thus this finite-particle example is
not invariant under arbitrary source-cell subdivision. It cannot yet be applied
as a converged continuum galaxy prescription. Positive energy alone is insufficient.

This analytically identified problem points to a needed next repair: a source
measure with a partition-independent interaction, then repeated orbit and
refinement tests. We did not silently change the law after finding it.

## What real observations are already available

The metadata-only inventory confirms local NGC2976 andNGC3198 THINGS HI cubes,
stellar maps/masks/colors and CO(2-1) integrated-intensity/error maps. Receipt
hashes, exact paths and byte checks are in audit-and-inventory.json. Cubes contain
line-of-sight Doppler information, not full three-dimensional particle velocities
or source histories. The CO products inventoried here are integrated maps, not
an independent full velocity cube.

An existing MaNGA product manifest lists645 processed and902 all-row internal
kinematic summaries, including stellar/gas position angles, gas rotation,
stellar velocity dispersion and asymmetry. This branch inspected the manifest,
not those velocity values. A spatially separated or independent tracer test may
eventually be possible, but the present scalar summaries do not supply all source
velocities or the history needed by this mechanical model. The same measured
velocity cannot serve as the source-motion input and the quantity being predicted.

Therefore observational motion and memory fitting remain SOURCE_BLOCKED pending
an independent source/response split, geometry and selection/noise admission.
No suitable gravitational reflecting medium is identified by these inventories.

## Promising directions and fixes

1. Use co-/counter-rotation and independent tracer passages as discriminators;
   motion need not reduce to one scalar spin or density multiplier.
2. Keep explicit stored energy and reaction forces. Memory can be tested without
   invoking an unaccounted source of extra energy, but a static halo needs an
   additional mechanism if equilibrium memory gives zero force.
3. Repair source-partition dependence before the kinetic law reaches real cubes.
4. Use ensemble or short-interval observables for the sensitive clustered case;
   converged energy does not guarantee a converged individual trajectory.
5. Specify a spatial propagation law and independent histories before interpreting
   internal oscillator frequency as actual gravitational memory or galaxy age.
