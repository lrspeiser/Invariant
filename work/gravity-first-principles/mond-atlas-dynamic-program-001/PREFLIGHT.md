# New orbit and storage mechanics program

THEORY_BENCHMARK_ONLY. No observed response fitting. Extends the previous
structure-environment-motion snapshot experiment to evolving coupled orbits,
with a different smooth positive kinetic coupling and new explicit internal
memory states. All constants dimensionless; G=1, softening b=0.05, L=1.

Velocity model Lagrangian:
L=T0 + 1/2 sum k_ij(r)|v_i-v_j|^2 - V,
k_ij=epsilon*(m_i*m_j/(m_i+m_j))*exp(-r^2/(2L^2)),
V=-sum m_i*m_j/sqrt(r^2+b^2).
The mass matrix is diag(m)+positive weighted graph Laplacian. epsilon=0.5.
Euler-Lagrange acceleration includes both gradient-k and time-derivative-k
terms. Canonical energy and angular momentum, not ordinary kinetic terms alone,
must be monitored. Relative velocities ensure Galilean covariance.

Separate memory model:
L=T0 + sum qdot_ij^2/2 - V - sum omega^2*(q_ij-h_ij(r))^2/2,
h_ij=0.15*sqrt(m_i*m_j)*exp(-r^2/(2L^2)).
qddot=-omega^2(q-h); forces include +omega^2(q-h)*grad(h).
omega=0.2 and2. Initial q=h and qdot=0. Finite pair-state dynamics is causal
in time but instantaneous across distance, not relativistic propagation. Its
energy reservoir is explicit. Static equilibrated q=h gives no extra force.
No damping or external driver. No arbitrary infinite-energy memory multiplier.

Four fixed configurations: nested co-rotation, nested counter-rotation,
clustered satellites, and a source flyby. Nested: central mass10 plus four
mass0.05 satellites at (1,0),(-1,0),(0,2),(0,-2), softened circular speeds.
Counter-rotation flips outer pair velocities. Clustered positions (1,+/-0.15),
(-1,+/-0.15), same central mass, local circular tangential velocities. Flyby:
m=[1,0.1], positions(-3,0),(3,0.8), velocities(0.1,0),(-1,0), COM removed.
Different configurations do not share the same total modified energy.

Integrate Newton control, kinetic model, memory slow, memory fast. Orbit cases
T=40 (about20 nominal inner periods); flyby T=12. DOP853 at maxstep0.1/0.05,
rtol1e-8/1e-10, atol1e-10/1e-12; 401 common samples. Gate max normalized energy,
momentum and angular-momentum drift<1e-5; normalized final-position difference
between resolutions<1e-3. Keep failures; chaotic paths may fail trajectory
convergence without violating conservation. Report radius range, closest
encounter, mean source-relative inward acceleration, oscillator energy exchange.
No late threshold tuning or case removal. Qualitative numerical stability is
limited to this interval, not a theorem for arbitrary systems.

Before integration: asymmetric manufactured state energy directional derivative,
canonical momentum conservation, boost/rotation invariance, positive massmatrix,
finite-difference potential force and Newtonian zero-coupling limit, tolerances
1e-6 relative or normalized absolute. A fixed-radius unforced memory oscillator
has analytic cosine solution and serves as an independent temporal control.

Reflection branch SOURCE_BLOCKED: no explicit carrier/surface interaction law
or independent reflecting medium. No optical reflection force inferred by
analogy. Real-data inventory is separated from the simulations; HI Doppler and
stellar velocity responses cannot be reused as predictor features for themselves.
