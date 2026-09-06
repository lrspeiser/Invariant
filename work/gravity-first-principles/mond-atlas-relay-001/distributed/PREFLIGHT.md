# Distributed secondary response: frozen preflight

Admission: THEORY_BENCHMARK_ONLY. No observed response data will be opened or scored.
The public McMillan (2017) model https://arxiv.org/html/1608.00971 supplies a
calibration normalization, not a measured 3D halo map. Its author parameter file
is bound in ../.. /mond-atlas-halo-return-001/source-receipts.json (without the space).
Use rho_s=8.53702e6 Msun/kpc^3, L=19.5725 kpc, G=4.30091727003628e-6
kpc (km/s)^2/Msun. Manufactured razor-thin exponential disk: M=6e10 Msun,
a=3 kpc; this is not a reconstructed Milky Way disk.

The effective source kernel is q(s)=eta/[4 pi L^3 x(1+x)^2], x=s/L.
Single-generation convolution only; eta=4 pi rho_s L^3/M fixes the point-source
response to the published NFW calibration. The associated pair potential is
-G eta dm log(1+s/L)/s. No attenuation, recursive relay, or temporal storage.
The kernel is positive and spherical about each source, with no preferred center.

Before evaluation: unit tests require exact mass and R^2=6a^2 disk moments (1e-10),
point-source analytic identity and translation/rotation/reciprocity (1e-10),
potential-gradient agreement (1e-6 relative), and compact far-field limit (1e-4).
Resolution uses Laguerre radial/uniform azimuthal quadratures 32x64,64x128,128x256.
Fixed points: R=[0.5,1,2,4,8,16,32,64] kpc and z=[0.5,2,8,32] kpc.
Maximum base-to-fine vector relative difference must be below 1%; retain every
failed point. This numerical convergence gate cannot be repaired by target fitting.
Compare fine disk response against point-source NFW, retaining strength and
direction differences; no retuning. Angular roundness is sampled at fixed radii
[8,16,32,64] and angles [15,30,45,60,75,90] degrees above the plane, with the
same convergence gate. Far-field points are not independent galaxy observations.

Also analytically assess sharp kernel truncation at s=10L: enclosed response
uses min(x,10); unchanged below cutoff for a point source, Keplerian outside.
Report effective-source budget eta*m(x) at x=1,10,100,1000; this is NOT energy.
Untruncated NFW has logarithmically infinite total source weight. No recursive
stability, photon deflection, energy conservation, or physical absorption claim.
