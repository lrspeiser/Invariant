# HI-weighted radial column forces, frozen before force evaluation

SOURCE_BLOCKED for observed spectra. This supplies conditional source-only forces
for first-score-protocol v2; no spectra, fit or likelihood is opened here.
Primary f4 stars h=.1, sensitivity h=.4; atomic and molecular h=.2 kpc.
Use full original bilinear source maps and fixed conversions, no spherical,
midplane or sparse72point substitution. HI tracer weight is the f4 atomic map
divided by1.36; that constant cancels from the weighted force but is retained
in reported pure-HI columns. Source and code hashes bind before calculations.

Compute gbar(R)= integral dphi dz rho_HI(R,phi,z)*[-g dot e_R] /
integral dphi dz rho_HI. Keep signed forces. Because the target HI vertical
profile is separable, integrate both vertical exponentials before sampling phi.
The exact Newton Fourier multiplier is W(k)=(1+k hs ht/(hs+ht))/
[(1+k hs)(1+k ht)], ht=.2. This is the expectation exp(-k|Zs-Zt|).
For hs!=ht, the difference-height distribution on positive z has weights
[hs exp(-z/hs)-ht exp(-z/ht)]/(hs²-ht²); for equal h it is
(1+z/h)exp(-z/h)/(2h). Use this normalized positive distribution for the log
kernel, whose horizontal acceleration is -G*d/[s²(s+4)],
s=sqrt(R²+z²+.05²), eta=1. No force parameters adjusted.

Newton uses exact original bilinear tent Fourier coefficients, explicit lattice
alignment and real-force Hermitian convention. Settings fullwidth64/dx.0625,
96/.0625,96/.03125. Log uses real-space column kernel FFT convolution:
width32/dx.0625/order32,32/.03125/order64,32/.03125/order128,
32/.015625/order128,48/.03125/order128. Source extent ±8 and requested
locations within6 ensure relative Cartesian separation<14, belowhalfwidth16;
padding prevents source wrap at evaluation locations. Boundary/grid tests still
check spectral interpolation/truncation. No physical source smoothing.

Generate radii .025..6kpc step.025, with .05step output subset; primary requested
support .05..6. Angular rules512/1024/2048, exact bilinear HI weight inside the
integral. Source-only R.025 is a center interpolation extension. Report radial
and tangential weighted RMS deviations, not an unqualified axisymmetric force.
Tables preserve each stellar/HI/CO contribution and total so one-at-a-time mass
factors can be applied linearly later. No posthoc force-strength fitting.

Freeze gates: separate Newton box/grid, log box/grid/vertical and angular
1024→2048. For signed radial columns require RMS change<1% and pointwise<3%
(denominator abs(fine), zero requires absolute agreement1e-10 field units).
Report gates over full .05..6 and aperture-radius .75..2.5 separately, never
discard outer/inner failures. Radial interpolation coarse.05→fine.025 evaluated
at intervening radii must obey same gates. Keep all component and total failures.

Before actual fields: adaptive independent difference-height normalization and
Newton multiplier, log radial quadrature vs adaptive integral includingcenter,
manufactured Gaussian planar Newton column vs independent Hankel integral,
manufactured compact source direct log sum vs FFT (reference quadrature), source
mass1e-10, rotation/translation/superposition controls. Reference force relative
agreement<.2%; quadrature convergence failures retained. Single CPU thread for
FFT; CUDA may independently check actual pair sums. Budget300seconds source
calculation per bounded pass, working RAM16GB, public output<20MB. No adaptive
physical changes. Missing-map assumptions are inherited, not repaired here.
