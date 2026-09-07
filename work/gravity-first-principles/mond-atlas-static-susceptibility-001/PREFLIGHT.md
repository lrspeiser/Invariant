# Frozen minimal static-susceptibility test

THEORY_BENCHMARK_ONLY. No source/history dataset supplies the oscillator states;
no observed motion, lensing or new raw source arrays will be opened. This is
one new conservative mechanism, not a sweep selected to fit galaxies.

Primary mechanical reference: Caldeira and Leggett, Quantum Tunnelling in a
Dissipative System, Annals of Physics149,374–456(1983), §3 equations3.2–3.4,
doi10.1016/0003-4916(83)90202-6; accessible paper
https://lab.semi.ac.cn/download/0.8104329777920175.pdf. Completing the square in
linearly coupled harmonic coordinates gives a negative eliminated potential;
a counterterm cancels it when the original potential is to be held fixed.
The paper is a mechanical-method reference, not evidence for this gravity law.

Keep dimensionless G=1,b=.05,L=1,eta=.15,omega=[.2,2] from dynamic-program-001.
Keep h_ij=eta sqrt(mi mj)exp(−r²/(2L²)), oscillator kinetic energy qdot²/2,
and softened Newton V_N=−sum G mi mj/sqrt(r²+b²).
Original Uold=omega²(q−h)²/2.
New U=omega²q²/2−omega² qh=Uold−omega²h²/2.
Particle pair force F_i=F_N+omega²q grad_i h; qddot=−omega²(q−h).
The entire change is removal of the explicit counterterm. No damping, driver,
time-energy creation or new length/strength is added.

At fixed separation qeq=h, Ueff=−omega²eta² mi mj exp(−r²/L²)/2,
and Fextra_i=−omega²eta² mi mj exp(−r²/L²)(xi−xj)/L².
Static susceptibility dqeq/d(omega²h)=1/omega².
No claim that a conservative oscillator automatically settles to qeq.

Pre-integration gates: independent scalar minimization1e−10; finite-difference
potential force and energy directional derivative1e−6; pair-force reaction,
translation/rotation/boost covariance1e−10; positive oscillator curvature;
coincident 0.3/0.7 source split acceleration and energy agreement1e−10 after
explicitly subtracting internal softened Newton/Gaussian constant energies.
Cross-pair q and qdot scale as sqrt(childmassproduct/parentmassproduct), so the
same physical internal state, not arbitrary child excitations, is compared.

Fixed-separation exact oscillator test q=h+(q0−h)cos(omega t), q0=0,qdot0=0,
T40,DOP853 requires max coordinate error1e−8 and energy error1e−8. Independent
complete-square lower bound checked on random manufactured states.

Two-body tests use masses1,.1 at r1 and2, exact modified circular speed derived
from Ueff, q=h,qdot=0; a separate perturbed case reduces that speed by3% without
changing q. This constructs declared equilibria rather than silently replacing
old initial conditions. Run T40 with steps.1/.05, rtol1e−8/1e−10,
atol1e−10/1e−12;401samples. Sixteen total integrations. Conservation gate1e−5;
base/fine final-position relative difference1e−3; exact circular solution
error1e−6. Keep every failure; no late refinement or parameter retuning.

Record physical limitations as tests, not admission successes: extra/Newton
ratio in unsoftened small-r limit scales r³; at fixed r it is independent of
source mass, so arbitrarily large acceleration by mass scaling does not restore
Newton. At large r the Gaussian force decays exponentially, not as1/r.
Fixed-total-mass potential is bounded below by
−(G/b+omega²eta²/2)sum(mi mj), but fixed-particle-mass dense N growth is quadratic,
not thermodynamically extensive. No galaxy-support/continuum theorem follows.
Retain naive inverse-radius h collapse and linear-radius h unbounded/outward
force as analytic negative controls; these are not retuned candidate models.
