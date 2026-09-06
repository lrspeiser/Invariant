# Frozen radial comparison before response scoring

2026-09-06. DATA_AND_PAPER_ADMITTED for radial empirical comparisons only.
SPARC Lelli et al. 2016, arXiv1606.09251, documented public Rotmod_LTG.zip and
Table1 supply radius, baryonic force templates, photometry, and rotation speeds.
Configuration fixes source definitions, 139 previously exposed identities,
selection, grids, exclusions, folds and equal-galaxy error metrics. No reserved
archive member bodies will be opened. Historical exposure remains disclosed.

This adapts prior theory ideas to the actual available source information; it
does not upgrade point-source approximations or scalar radial laws to 3D gravity.
No fitted halo mass or parameter is imported. Vobs and its published error are
response/eligibility columns, never source features. Every formula parameter is
global within its training galaxies. We do not fit a separate halo to each galaxy.

Let g_b be the published baryonic radial acceleration, M the declared photometric
plus HI proxy, L=lambda Rd, x=r/L and a0=1.2e-10 m/s^2. Baselines are Newton and
the simple algebraic MOND relation. Extra accelerations added to Newton are:

- Finite p2: eta GM/L^2 /(1+x)^2.
- Finite p3: eta GM/L^2 x/(1+x)^3.
- Finite mixture: the convex combination of p2 and p3 using frozen weights.
- Point kernel: eta GM/r^2 [ln(1+u)-u/(1+u)], u=min(x,10).
- Clock potential: Phi_chi=-beta Psi0 ln[1+GM/(Psi0(r+Rd))],
  Psi0=lambda a0 Rd; its inward acceleration is
  beta GM/[(r+Rd)(r+Rd+GM/Psi0)]. Clock lapse exp(Phi/c^2) is a
  representation of a potential, not a measured clock map or conserved energy.

Absorption uses g=g_b exp(-kappa Sigma_star/100). Surface relay uses
g=g_b[1+beta/(1+Sigma_star/Sigma0)]. Sigma_star is a local stellar surface
proxy, not line-of-sight gas opacity, volume density or a clock measurement.
These two radial empirical laws have no admitted nonspherical force extension.

Independent pre-response tests: dimensional scaling, zero-strength recovery,
absorption bound, central/outer limits, positive/saturating effective source,
independently differentiated clock/p2/p3 potentials, NFW mass quadrature,
NumPy/CuPy agreement on planted manufactured sources, exact grid recovery of a
planted signal, and changing held labels cannot change training selections.
Clock derivative relative gate1e-6, analytic formula1e-10, GPUlogspeed1e-10.
Resolution is analytic radial evaluation; the kernel integral is checked by
independent quadrature. Radius origin excluded; finite kernel cutoff is fixed10L.

Save all attempts, including failures. No post-score grid expansion or response
dependent cohort cuts. Uncertainties in mass-to-light ratio are explored only
through the frozen global factors; full distance/inclination/molecular/geometry
uncertainties remain unresolved. Bootstrap intervals describe this exposed sample,
not survey independence or calibrated discovery significance. Report every family
and strata with actual counts. Freeze before any run loads response arrays.
