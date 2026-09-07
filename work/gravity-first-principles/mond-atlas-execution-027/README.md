# Larger program: measurement and numerical validation, round 027

Four completed branches strengthen the prerequisites for comparing gravity laws. None supplies a new observational gravity fit, a measured three-dimensional mass distribution, or a validated replacement for dark matter. The broader goal remains active.

| Branch | Executed result | Remaining limitation |
|---|---|---|
| Noise by spatial mode | 575 of 576 modes pass the frozen descriptive range; joint q/N is 0.9998. The earlier cancellation between spatial scales is resolved. | One spatial mode and the lowest background-offset channel quartile (q/N 1.566) fail. Limited adjacency checks do not establish full independence. |
| Detection and background ablation | 2,592 injections into actual backgrounds plus 216 noiseless controls. In the representative case, changing only the threshold reduces the middle-channel recovery deficit from 32.67 to 15.11 percentage points. | Removing continuum subtraction from injected source emission changes no true-flux recovery. Delivered-background processing remains present; its physical cause is unidentified. |
| External field and domain size | Six new conditional field solves. Moving the outer boundary from 18 to 24 kpc changes center-relative fields by 0.44–0.49% RMS. | The shorter smoothing scale still fails a height-specific mesh check (11.21%). The largest domain and finest grid have not yet been tested together. |
| Molecular-gas source audit | Twelve aperture/quadrature measurements independently reproduced. Inner useful apertures have about 98–100% coverage. | Outer apertures have only 37.48% coverage in NGC2976 and 34.32% in NGC3198. Missing area is not evidence of absent gas. |

The noise and injection comparisons use previously exposed development regions. They are not fresh confirmation on independent galaxies. No thresholds or physical parameters were changed after eastern results to manufacture a passing result.

The molecular-gas diagnostic also quantifies a possible processing effect: clipping all negative pixels to zero would raise signed aperture luminosity by 8.22% at NGC2976 radius 3 kpc and 1.68% at NGC3198 radius 10 kpc. This is a counterfactual pixel operation, not a measured bias in our source inversion or a correction to the true mass. The CO-to-mass conversion is still assumed. Published CO integration windows also depend partly on HI velocities, which must be represented in a future joint analysis.

## Reproducible evidence

- [Noise experiment](../mond-atlas-noise-mode-power-001/README.md): independent manual cosine/covariance replay, maximum discrepancy 4.98e-13.
- [Selection experiment](../mond-atlas-selection-ablation-001/README.md): all 702 earlier empirical/noiseless trials replay exactly; independent ledger and aggregate verification retained.
- [External-field experiment](../mond-atlas-external-boundary-001/REPORT.md): independent replay of 3,080 vectors and 12 comparisons, maximum metric discrepancy 1.39e-17.
- [CO photometry](../mond-atlas-co-photometry-001/README.md): independent FITS/WCS/area implementation agrees to 4.46e-11 relative error. A supplemental post-access dependency receipt records two helpers omitted from the original prospective list; it is not a retroactive preflight freeze.

## Next work

The next bounded branches address the background-offset covariance discrepancy, the joint finest-grid/largest-domain external-field endpoint, and whether primary-source metadata makes Galactic foreground contamination a testable explanation for the channel-dependent anomaly. Foreground contamination is a hypothesis, not a diagnosis or permission to remove channels.

Distributed response, density/refraction, motion/current, and time/memory mechanisms retain their previous results and limitations. Improving measurement calibration is necessary before using these data to adjust gravity formulas. Full source-and-selection-aware motion predictions, independent transfer, a lensing metric, and cluster/Solar-System tests remain outstanding.
