# Frozen spectral Newton repair

SOURCE_BLOCKED for observed response scoring. Conditional source/numerical tests
only; prior logarithmic package's Newton integration failures remain unchanged.
Reuse its four NGC2976 alternatives and all 72 points, exact component conversions
and exponential heights. Source maps, packets, inherited primary references and
the original spatial manifest bind before field calculation. No velocities fit.

Represent each original planar bilinear tent exactly in Fourier space: nodal
DFT multiplied by h² sinc²(kx h/2) sinc²(ky h/2), where sinc here is sin(x)/x.
Deposit original nodes onto a common 0.03125 kpc lattice, not new point sources.
The tent transform removes that deposit's point-source interpretation. Coarser
field grids truncate Fourier modes, not physically smooth or retune the source.
This preserves mass exactly and uses the unchanged continuous bilinear source.

For normalized vertical exp(-|z|/h)/(2h), V is its convolution with exp(-k|z-z'|):
V=[exp(-k|z|)-kh exp(-|z|/h)]/[1-(kh)^2]. At kh=1 use
0.5(1+|z|/h)exp(-|z|/h); derivative is
-sign(z)|z| exp(-|z|/h)/(2h²). For k>0, Phi=-2piG Sigma_k V/k,
g_horizontal=-i k_horizontal Phi and gz=2piG Sigma_k V'/k.
The k=0 vertical term is -2piG Sigma_mean sign(z)(1-exp(-|z|/h)).
Potential zero-mode convention is irrelevant to reported accelerations.

Fix G=4.30091727003628e-6 kpc (km/s)^2/Msun. Periodic boxes have full widths
32,64,96 kpc at field spacing0.0625; a fourth jointly largest/fine grid has
width96 and spacing0.03125. Evaluate at original positions with periodic cubic
interpolation. Require middle-to-largest box and largest coarse-to-fine vector
RMS<1%, every point<3%, independently for each component and total. Preserve
early box changes and all failures. This is a periodic-domain approximation to
isolated fields, not an exact isolated boundary. Symmetric z=0 vertical field
must vanish. Reflection/rotation and exact source mass checks precede scoring.

Before source force fields: independent adaptive vertical convolution (including
kh=1), single-mode Poisson residual/analytic gradients, and a Gaussian planar
source with exponential vertical profile checked against independent Hankel-
Bessel integrals. Gaussian benchmark width128, spacing0.125, sigma1 kpc,
R=0.5,2,6 kpc, z=0,0.4; vector agreement<0.2%, retain failures. Check all
units and symmetries; mass normalization1e-10; analytic mode residual1e-5.
Bind implementation/preflight before numerical controls. Single-thread FFT CPU
budget300 seconds source fields, estimated working RAM<8GB (<16GB limit).
No further resolutions, source changes or adaptive physical tuning in this run.
