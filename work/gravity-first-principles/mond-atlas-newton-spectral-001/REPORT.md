# The continuous-source Newton repair partly succeeds

We calculated Newtonian gravity from the same continuous bilinear planar maps
and exponential vertical profiles, without concentrating source cells into
singular point masses or adding force softening. All four conditional NGC2976
source alternatives and all 72 positions were evaluated at four frozen box/grid
settings. The single-thread source calculation took 53.6 seconds and saved
4,608 component/total vectors. No observed response was opened or scored.

The numerical repair substantially improves the previous Newton calculation,
but does not finish it for every source alternative. All sixteen comparisons
between the 64 and 96 kpc boxes pass. Twelve of sixteen coarse-to-fine component
and total comparisons pass. The same frozen thresholds apply: RMS change below
1%, every sampled point below 3%.

| Source alternative | Total box RMS / worst point | Total grid RMS / worst point | Total passes both? |
|---|---:|---:|---|
| f1, stellar height0.1 kpc | 0.021% / 0.248% | 0.340% / 1.192% | Yes |
| f1, stellar height0.4 kpc | 0.022% / 0.257% | 0.099% / 0.386% | Yes |
| f4, stellar height0.1 kpc | 0.020% / 0.249% | 2.032% / 16.597% | No |
| f4, stellar height0.4 kpc | 0.021% / 0.256% | 0.283% / 1.361% | Yes |

The failed component comparisons remain in the receipt: the f1 CO component
has 0.265% RMS but 5.839% worst-point grid change, duplicated across its two
stellar-height cases because the gas packet is shared. The f4/height0.1 stellar
component has 2.193% RMS and 18.936% worst-point change. Thus only f4/height0.4
passes every component-plus-total gate; the three successful total fields must
not be presented as three fully component-validated cases. The old point-source
quadrature failures are unchanged and remain visible in the logarithmic package.

The finer inverse reconstruction's thin stellar source retains enough spatial
detail that Fourier truncation still matters at sampled locations. That is a
numerical/source-representation issue, not evidence against Newtonian gravity.
No source broadening or physical softening was introduced to make it pass.
A further grid refinement, if authorized and frozen separately, would test
whether these remaining discrepancies are resolved. The current run makes no
claim that its finest failed field is converged.

The method integrates each original bilinear tent analytically in Fourier
space. Its coefficient is the original nodal transform multiplied by
h² sinc²(kx h/2) sinc²(ky h/2); h is the original source lattice spacing.
The nodes are deposited on a common computational lattice only to evaluate
their transform. The tent factor reinstates the continuous source; these are
not point masses. Reducing the retained Fourier modes changes numerical
resolution, not the physical source. Mass is preserved exactly to the frozen
1e-10 check. Original packet heights and conversions are unchanged.

The vertical response was independently derived by convolving exp(-k|z-z'|)
with exp(-|z'|/h)/(2h). For k>0 it is
V=[exp(-k|z|)-kh exp(-|z|/h)]/[1-(kh)^2]. Its removable kh=1 singularity has
limit0.5(1+|z|/h)exp(-|z|/h). The corresponding derivative and the exact k=0
vertical sheet term are implemented separately. Horizontal gradients follow
from -i k Phi; vertical acceleration follows the derivative of V. G uses
kpc (km/s)^2 per solar mass, and reported acceleration uses (km/s)^2/kpc.

The boundary is horizontally periodic. Increasing the box tests the unwanted
image fields; it does not turn the largest box into a mathematically exact
isolated boundary. Both box checks use the coarse spacing; the largest box
also has its own coarse/fine comparison. A separate fine-grid box sequence
was not run, so coupled box/grid errors remain a limitation. Cubic periodic
interpolation to the 72 original positions is part of the measured grid error.

Before source fields, independent vertical integrals agreed within 7.55e-15,
the single-mode Poisson identity within 1.97e-9, and six independent Gaussian
disk/Hankel-Bessel force benchmarks within 0.0455%. Reflection/rotation and
midplane symmetry checks passed. Direct summation of the actual continuous
tent transforms independently reproduced 48 low and high Fourier modes across
all eight unique source packets within 3.27e-16 relative to the mass mode.
The source alternatives sharing gas packets use identical cached results, not new
independent evidence. All component sums replay to roundoff below9.1e-13.

The inherited tracer-to-mass conversions, missing material/exterior, unmatched
beams, assumed vertical structure and CO inversion floor remain. Admission
remains SOURCE_BLOCKED for actual motion/lensing scoring. This is a numerical
repair of ordinary gravity, not a new physical mechanism or an observational
test of the logarithmic response.

Files: scripts/mond_atlas_newton_spectral.py, its _review.py and _coefficients.py
companions; run001 binds the inputs and retains every field, control and gate.
