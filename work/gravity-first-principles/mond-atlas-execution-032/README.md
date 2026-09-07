# Execution032: larger actual-source numerical and injection program

The broader gravity goal remains active. This milestone tests the actual reconstructed NGC2976 source model, source alternatives, spectral numerics and fitting transfer. It does not rank gravity laws against observed source-region spectra.

## Findings

- Four HI source quadratures, 72 physical cases and 192 nuisance cases produced 14,400 aperture refinement comparisons. All profile-L1 checks passed (<0.483%); 42 centroid checks failed and remain recorded. The prespecified primary Newton injection family passed all1,350 checks (maximum0.418% profileL1 and0.188km/s centroid difference).
- The RTX5090 callback agrees with the CPU at tested nuisance corners to7.29e-14mJy/beam; median prediction time fell from0.360s to0.00326s. Broader numerical execution finished in53.4s. This is arithmetic validation, not agreement with observations.
- All51 template experiments were identifiable: three noiseless recoveries and48 noisy fits. Noiseless recovery passed. With real background, median errors were0.177km/s in systemic velocity,0.126km/s in width and1.57% in amplitude. Corresponding Gaussian errors were0.122km/s,0.0945km/s and0.714%. Held discrepancy q/channel medians were1.233(real) and0.987(Gaussian). Backgrounds overlap and were previously exposed; these are descriptive diagnostics, not independent significance estimates.
- All51 fitted injection parameter sets also pass all3825 source-refinement comparisons at their actual fitted values: maximum0.280% profile difference and0.015km/s centroid difference. This closes fitted-nuisance convergence for these injection trials, not for unseen future fits.
- Spreading mapped matter to common30arcsec resolution changed inner total Newton force by9.35% RMS for thin stars and11.90% for thick stars. Log-extra changes were5.81%/7.07%. Missingzero/annular alternatives changed inner total forces much less (Newton<=0.0693%, log<=0.0458%), despite several-percent differences in gas mass. These conditional findings concern this galaxy and radial region.
- Alternative source packets, force tables and matched pressure profiles are available. Common30 places about0.175% of HI mass beyondR6. Its emission uncertainty cannot inherit the baseline edge bound.
- The observed-data driver enforces separate training and evaluation phases, both spins, frozen predictions, exact144pixel means and bound evidence. Manufactured isolation/tamper controls pass. Actual source spectra remain unopened.

## Interpretation and next work

Spatial arrangement and resolution matter enough to compete with proposed extra-gravity effects. Background structure also affects fitted amplitude more than the ideal Gaussian trials suggest. Neither result identifies a new gravity law or a root cause of the observed galaxy's motion.

Propagate all source alternatives through aperture/beam/channel emission, explicitly bounding their extended tails; resolve or bound retained centroid failures under the frozen metric; test nuisance-fit spectrum convergence where parameters actually land; then admit the complete frozen actual-observation comparison. Carry covariance leave-one-core-out and material uncertainties into interpretation. Distributed response, refraction, currents and time/memory still require distinct mechanism tests; this source/measurement milestone must not be presented as their completion.

Prior research integrity:2067 prior-manifest file hashes verified before work. The abandoned first injection report retained its TypeError receipt; the corrected fit002 completed. Raw source/noise/prediction arrays remain private.
