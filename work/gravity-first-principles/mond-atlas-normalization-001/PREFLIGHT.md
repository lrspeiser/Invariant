# Frozen normalization benchmark

Status: THEORY_BENCHMARK_ONLY. No observed response or source array will be accessed. This algebraic test examines our Gaussian-averaged density prescription, not a claim about the laboratory environment or a test of published refracted gravity in all regimes.

The parent equation is div(epsilon grad Phi)=4 pi G_bare rho, as in the classical permittivity formulation documented by [Matsakos and Diaferio (2016)](https://arxiv.org/abs/1603.04943). Our smoothing and rational response differ from their model and are defined in execution025. Here epsilon=0.2+0.8*u/(1+u), u=rho_bar/(1e7 Msun/kpc^3). Fixed smoothing lengths are 0.25 and 0.5 kpc.

For locally uniform epsilon_lab and a perturbing test mass too small to change it, G_measured=G_bare/epsilon_lab. A spherical uniform-epsilon test environment therefore gives a/a_Newton=epsilon_lab/epsilon_env. Verify that identity against an independent spherical flux calculation, with positive and inward force, inverse-square ratios, dimensions and epsilon=1/identical-environment limits. Relative numerical tolerance 1e-12. Fixed u values: 0,0.01,0.1,1,10,100. Radii: 1,2,8 in arbitrary consistent length units. These are manufactured environments, not Milky Way estimates.

For a pointlike one-solar-mass source, independently integrate a normalized 3D Gaussian and calculate its maximum contribution to u=M/((2*pi)^1.5*ell^3*rho_c). This bounds how much that isolated mass can change epsilon under this smoothing, but does not model its surrounding Galaxy. Numerical normalization tolerance 1e-10.

No physical parameter is selected, no force data are fitted, no Solar-System test is claimed, and no prior published result is overwritten. Variable environments, spatial gradients, field boundaries and the actual laboratory source/environment remain untested. A full multiscale calibration is required before extrapolating the galaxy prescription.
