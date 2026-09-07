# Source alignment and Nyquist audit

No original source or saved force output was changed. The audit re-evaluated
source Fourier coefficients and boundary-strip contributions, not the full
source-field experiment. It checks both 96 kpc grids and all eight unique packets.

Every source node lies exactly on the 0.03125 kpc deposition lattice: the maximum
absolute residual of node/0.03125 minus its nearest integer is zero for every
packet. The audit explicitly asserts this. Thus rounding moved no source mass
in this run. The original general-purpose function lacks that assertion; reuse
with differently aligned nodes must add an explicit rejection rather than assume
alignment. No modification to its frozen implementation was made here.

The even-grid crop retains a half-open frequency interval, including negative
Nyquist but not its distinct positive-frequency counterpart. Horizontal
differentiation also has the usual even-grid Nyquist ambiguity. Consequently
the complex force spectrum is not exactly Hermitian at those strips. Taking
real(IFFT) is an explicit, mathematically defined Hermitian projection:
H(S)[j] = (S[j] + conjugate(S[-j]))/2. Applying this projection to EACH FORCE
spectrum before IFFT produces exactly the same real grid, and therefore the
same interpolated real samples. An independent non-Hermitian manufactured
spectrum verified the identity to 4.17e-17. Projecting a source spectrum before
differentiating is a different operation at Nyquist and must not be substituted
without checking its effect.

The discarded imaginary field is not everywhere roundoff. Its largest
whole-grid RMS relative to the real component is 0.2466%, in the f4 thin stellar
coarse-grid calculation. On fine grids the largest such ratio is 0.0116%.
These values were calculated by Parseval from Hermitian/anti-Hermitian spectra,
so they measure the actual discarded imaginary IFFT component. They do not
directly measure an error in the retained real field.

To measure a real-field sensitivity, we separately removed both Nyquist strips
from each force spectrum and evaluated their real contribution with the same
IFFT/cubic interpolation. This is a diagnostic alternative cutoff convention,
not an adopted correction. The f1 source tent transforms have zeros at these
frequencies, making their effects below 3e-36 relative at sampled points.

| Total source | Coarse strip contribution RMS / worst point | Fine strip contribution RMS / worst point |
|---|---:|---:|
| f4, stellar height0.1 kpc | 0.06997% / 2.18780% | 0.0000827% / 0.001518% |
| f4, stellar height0.4 kpc | 0.005289% / 0.133214% | 0.0000310% / 0.000430% |

The thin f4 stellar component alone has coarse-strip RMS0.07540% and worst
point2.5643%. Its fine-strip RMS is0.0000895%, worst point0.001762%. These
effects are much smaller than its retained coarse-to-fine failure of2.193% RMS
and18.936% worst point. The CO f1 failure also cannot be caused by these strips,
whose contribution there vanishes. The audit therefore does not explain away
or remove any failed convergence gate.

Interpretation: original source coefficients are evaluated without moving the
source nodes, but a finite frequency cutoff and interpolation still approximate
the continuous field. The real-part convention is a numerical cutoff choice,
not a newly inferred mass distribution or physical smoothing law. A future
implementation should declare its Nyquist convention and use explicit checks;
force-spectrum Hermitian projection alone would only make the existing real
operation explicit. A change to cutoff weighting should retain separate outputs
and repeat its affected convergence checks. The prior numerical and observational
limitations remain; no new source or motion admission is claimed.
