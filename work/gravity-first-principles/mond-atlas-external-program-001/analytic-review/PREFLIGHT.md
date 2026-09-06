# Independent interface reference

THEORY_BENCHMARK_ONLY. No astronomical source forces or response measurements.
Solve div(epsilon grad Phi)=0 for a sphere radius a=1, epsilon_in=1 and
epsilon_out=0.2, applied uniform field E zhat. Potential:
inside Phi=-A E z, A=3epsilon_out/(epsilon_in+2epsilon_out);
outside Phi=-E z+B E a^3 z/r^3,
B=(epsilon_in-epsilon_out)/(epsilon_in+2epsilon_out).

Freeze samples: theta=[0,0.1,0.4,0.8,1.2,pi/2,2,2.6,pi], phi=[0,0.7,2.1],
interface radius1, E=[-2,0,0.3,1,5]. Potential, tangential gravitational field
and epsilon-normal field continuous within1e-12 absolute; finite-difference
potential gradients and Laplacian at separated interior/exterior points agree
within1e-6. Exact E linearity and equal-epsilon uniform-field limit. Interface
normal itself is continuous and material coefficient supplies the jump law.

Derivation is an independent analytic PDE reference, not a proposed new
gravitational mechanism. Parent source model may use nonuniform smooth epsilon,
so this discontinuous sphere tests its equation/interface limit only. Fixed
epsilon makes the PDE linear; uniform applied-field response and internal
source solution superpose. Do not call this MOND external-field effect.
