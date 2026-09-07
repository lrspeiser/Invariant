# Source resolution matters more here than the two missing-cell choices

We constructed14new conditional source packets and evaluated their effects on
Newton and the fixed logarithmic response. The clearest result is that the
nominal30arcsec Gaussian source-resolution alternative changes the primary
HI-weighted Newton column force by9.35%RMS overR=.75..2.5kpc, and the log extra
by5.81%RMS. The corresponding thick-star differences are11.90% and7.07%.
No observed spectra or response fits were used.

| Alternative, versus original same-height source | Newton force difference RMS | Log-extra difference RMS |
|---|---:|---:|
| Nominal30arcsec, primary stellarheight.1 | 9.35% | 5.81% |
| Nominal30arcsec, stellarheight.4 | 11.90% | 7.07% |
| New missing-zero inverse, primary | 0.0693% | 0.0458% |
| New missing-annular inverse, primary | 0.0191% | 0.0108% |

These are conditional model differences across the aperture-radius interval,
not measured gravity discrepancies or uncertainty confidence intervals. The
missing-cell choices mainly affect outer gas in this construction; their small
inner total-force effects do not establish that all missing material is harmless.
The source geometry, conversions, vertical profiles and model family remain
assumptions. These differences cannot identify an oscillator or reflection
mechanism from a static potential.

## Explicit source construction

The original positive f4 inverse was fitted to trusted signed source-cell
means. It was not fitted to the generic annular-fill map. We retained that fact
and created genuinely new source packets for both missing-cell choices. In
trusted cells both new targets use the same nonnegative projection of the
signed mean. Missing cells insideR5 are assigned eitherzero or the existing
generic annular value. Unit target weights throughoutR5 make these imposed
regularized source constructions, not a source-noise likelihood. They are not
identical to the generic area-diluted positive-zero image. All six inversions
started fromzero and converged in750–1000iterations under the unchanged
regularization and stationarity gate.

The imposed-image residuals remain substantial: approximately15.5%stellar,
4.8%atomic and11.2%CO. Optimizer convergence does not imply observational
closure. The resulting zero/annular masses are2108.36/2110.72million solar
masses in stars,181.66/188.45million in atomic-plus-helium, and108.19/111.40million
in molecular-plus-helium under the inherited conversions. These new packets
preserve masks, original signed means and imposed-target provenance.

The common-resolution branch convolves each existing effective f4 source with
an additional Gaussian. Its covariance is first defined in observer east/north
coordinates, Cadd=C30-Cnative, then transformed through the frozen galaxy PA,
inclination and distance into major/deprojected-minor coordinates. The original
inverse's forward operator contains vertical projection and pixel integration
but no nativePSFdeconvolution. It therefore retains an effective native beam.
For the assumed spatially invariant separable-height model, mapped Gaussian
convolution commutes with vertical projection; a separate real-space control
verifies this. This is not applying an arbitrary circular disk-plane blur.

Native Gaussian proxies are the actual HI restoring ellipse7.407x6.42384arcsec,
PA71.79 from AIPS CLEAN HISTORY; the delivered CO header's13.396779arcsec
circular beam; and a2.1arcsec stellar Gaussian-core approximation. The nominal
30arcsec target is **not an exact matched non-Gaussian PSF**. Salo's composite
IRAC core description motivates2.1arcsec, whereas Querejeta's P5 text quotes
typical1.7arcsec at3.6microns and PSF-matching processing. If1.7were the true
Gaussian width, our frozen additional kernel would yield29.9747arcsec. No
retuning was made. Stellar wings, native-pixel response, registration and the
inherited source-cell approximations remain limitations.

Smoothing changes the fine latent source structure by71–78%RMS while preserving
mass. The .03125/.015625kpc nested-source projected differences are3.1–8.4e-7
relative. Cropping at the original±8kpc boundary loses at most2.15e-12 of mass;
no radius6 retaper or source renormalization was applied. This strong reduction
in fine structure with modest total-force changes is physically understandable:
the force averages over neighboring source elements, but it is still sensitive
to their distribution.

## Matched force and pressure outputs

The28.6second field replay uses each alternative's actual HI map inside the
annular/vertical force weight, retaining stellar/HI/CO components separately.
Of128mesh/angular checks,124pass; all aperture-interval checks pass. Four
full-range individual gas-component Newton point checks fail near weak signed
forces and are retained. This radial replay does not inherit full source-to-
spectra admission: its radial interpolation, expanded emission support and
the q2source-to-gravity comparison still need downstream validation.

Pressure profiles were also recomputed from each alternativeHI packet. They
use the frozen closure Pi=sigma_reference² times Gaussian.25kpc(rawHIcolumn),
with the RAWHIcolumn in the Euler denominator. The exported sigma10 column
scales by.25 or2.25 for sigma5or15. It is dPi/dR divided by rawHI, the signed
pressure contribution in vphi²=R*gbar+R*dPi/dR/rawHI; it is not an observed
linewidth or a separately established physical stress measurement. All aperture
derivative-grid checks pass. A missing-annular full-range near-zero-gradient
point check remains failed; no denominator or sign was clamped to hide it.

The common30 HI radial mass fraction beyondR6 is approximately0.175%, compared
with0.000514% and0.000988% for the missing-zero and annular constructions.
These are numerical source radial integrals, not delivered-aperture flux bounds.
The earlier baseline tiny-tail bound cannot be reused for the broadened source.

**Remaining work:** rebuild each alternative's projected emission cells/centroids,
native restoring-beam transport, source-domain tail bounds and channel profiles;
propagate its own pressure table and force uncertainty; complete full emitting-
support and spectral convergence checks. The alternative pressure/emission
contract explicitly forbids relabeling baseline outputs. The cube instrument
beam remains a separate operator: these effective-source experiments do not
claim native beam deconvolution or unique3D matter recovery. No observed-score
admission is granted by this package.

## Reproducibility and review

Private physical packets are listed by exact path/hash inrun001/summary.json.
The independent audit verifies all14packets, finite positivity/boundaries,
mass and forward projections, every missing-fill projected gradient, and the
observer-plane target covariance. Eight existing independent inverse tests were
replayed successfully after packet construction and before field replay; the
inherited inverse had already been benchmarked by source-resolution-001.
New Gaussian/mass/commutation controls ran before the source arrays. All failed
cases and approximations remain explicit.

Public results: fields001/radial-fields.csv.gz (30,720derived rows),
fields001/hi-column-profiles.csv, hi001/pressure-profiles.csv.gz (4,308derived
rows), and associated reviews/contracts. `derived-artifacts.json` records gzip
and decodedCSV hashes, schemas and finite-value counts; these are source-derived
diagnostics, not raw observational spectra. Source NPZ arrays remain private.

Primary references: [Querejeta et al., S4G P5](https://arxiv.org/html/1410.0009v1),
[S4G P5 archive/mask definitions](https://irsa.ipac.caltech.edu/data/SPITZER/S4G/docs/P5_README.html),
[Salo et al., composite PSF approximation](https://doi.org/10.1088/0067-0049/219/1/4),
[Walter et al., THINGS](https://arxiv.org/abs/0810.2125), and
[Leroy et al., HERACLES](https://arxiv.org/abs/0905.4742).
