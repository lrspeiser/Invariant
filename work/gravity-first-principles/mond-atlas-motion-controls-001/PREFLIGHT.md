# Motion forward benchmark preflight

Disposition frozen before implementation: **THEORY_BENCHMARK_ONLY**. No real source
is claimed suitable for this package, and observational response scoring is forbidden.
This implements prescribed optically thin emitting rings and their projection, with
no mass conversion, gravity inference or pressure-support closure.

The primary projection identity is Schoenmakers, Franx & de Zeeuw (1997),
[arXiv:astro-ph/9707332v1, equation A11](https://arxiv.org/pdf/astro-ph/9707332).
The emitting-ring / Gaussian profile / beam construction is grounded in Di Teodoro
& Fraternali (2015), [arXiv:1505.07834v1, sections 2.1-2.2 and equation 1](https://arxiv.org/pdf/1505.07834).
These equations admit mechanics controls only; their observational applications
are not ingested. The OUP full-page fetch failed; the primary arXiv PDFs were read.

Observer coordinates are right-handed (x,y,z), with +z receding, +x right and +y
up; PA is counterclockwise from +x, deliberately not astronomical PA from north.
Azimuth phi=0 is the receding major axis. Inclination 0 is face-on, 90 is edge-on.
The ring basis is e1=(cos PA,sin PA,0), e2=(-cos i sin PA,cos i cos PA,sin i).
Positions are R(cos phi e1+sin phi e2); velocities are
V_R(cos phi e1+sin phi e2)+V_phi(-sin phi e1+cos phi e2).
Thus v_los=v_sys+sin i (V_phi cos phi+V_R sin phi).

Weights integrate exp(-R/h)[1+a cos(phi-phi_a)] R dR dphi over 0<R<Rmax,
analytically normalized to unit intrinsic flux (arbitrary integrated-flux units).
This is emission per reference annular area, not a continuous warped-surface
mass density or a reconstructed three-dimensional source. Ring orientations vary
as i=i0+warp_i (R/Rmax)^2 and PA=PA0+warp_PA (R/Rmax)^2.
V_phi=V0[1-exp(-R/Rturn)] and V_R=U0[1-exp(-R/Rturn)] are prescribed kinematics.
No continuity, equilibrium, torque or gravity equation is solved.

The instrument uses finite Gaussian-integrated velocity channels, a separable
linear tent spatial assignment (a declared pixel response), then a normalized
finite-support sampled Gaussian beam. A halo as wide as the beam support retains
in-scatter from outside the science field. Spectral wings, halo losses, and final
field losses are recorded, never renormalized away. The finite beam is the exact
declared instrument; its missing infinite Gaussian tail is separately quantified.

The JSON config freezes all dimensions, quadrature levels, numerical tolerances,
synthetic injections, training/held-out partitions, known diagonal noise covariance,
fixed optimizer starts and descriptive recovery thresholds. Independent references
use rotation matrices, Gaussian CDF from SciPy, direct convolution, and quadrature
nodes different from the production midpoint grid. Controls must pass before the
synthetic study is generated; failures are retained and stop that run. Recovery is
an outcome, never a gate adjusted to obtain a desired result.

Both fits use the same training cells. Circular-only frees speed, global geometry,
systemic velocity and line width. Expanded adds inclination/PA warp amplitudes,
radial streaming and azimuthal emission amplitude. Fixed emissivity shape, center,
flux and asymmetry phase make this a conditional benchmark. Held-out channel,
pixel, and joint subsets are evaluated separately under supplied independent noise.
One noise draw per case is not a coverage or selection-validation experiment.

Retain zero-amplitude behavior and non-identifiability: a flat face-on disk cannot
reveal planar velocities, and an unresolved axisymmetric spectrum cannot distinguish
V_phi and V_R when they share a radial profile and have the same quadrature sum.
Pressure support, finite thickness, self absorption, dynamics and observed covariance
remain missing. No galaxy motion, gravity score or speed-derived mass will be reported.
