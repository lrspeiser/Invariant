# Partition repair frozen before calculations

THEORY_BENCHMARK_ONLY. Preserve original reduced-mass kinetic-law failures.
Replace k_ij=epsilon*mu_ij*exp(-r^2/2) by
k_ij=epsilon*m_i*m_j/Mref*exp(-r^2/2), Mref=1 fixed external mass normalization,
epsilon=0.5, same G=1, L=1 and softening b=0.05. Mref is not a pair mass or
a newly computed total mass when subdividing a cell. No target fitting.

Same variational equations, positive graph-Laplacian inertia, canonical energy
and angular momentum. Co-moving subdivision must preserve kinetic Lagrangian,
canonical momentum and aggregate acceleration. Softened Newton potential has
an internal constant self-potential for coincident split particles: compare
external interaction energy or subtract this explicitly identified constant;
self-force is zero, and cross kinetic terms vanish for identical velocities.
Do not claim equality of the arbitrary self-energy reference without adjustment.

Before integrations: compare original four-particle system to all particles
split into two coincident half masses and to unequal0.3/0.7 splits. Test each
child acceleration matches its original, weighted momentum and kinetic energy
agree within1e-10; positive inertia; finite-difference Euler-Lagrange residual
<1e-6; boost/rotation covariance and energy derivative<1e-6. Independent original
reduced-mass coupling is a known negative subdivision control, not repaired data.

Repeat the original four configurations (nested co/counter, clustered, flyby)
for corrected kinetic and Newton laws at same DOP853 maxstep0.1/0.05,
rtol1e-8/1e-10, atol1e-10/1e-12. OrbitT40 and flybyT12; additionally clusteredT2.
401 samples, original invariant threshold1e-5 and final-position difference1e-3.
All failures retained. No altered initial conditions or extra strengths after
seeing results. Compare old kinetic outcomes descriptively; long clustered
comparisons require convergence and cannot be inferred from conserved energy.

Coincident co-moving partition invariance is necessary but not sufficient for
continuum convergence: finite cell geometry, smoothing dependence, velocity
dispersion, fixed normalization, relativistic propagation and observational
source/response separation remain separate requirements.
