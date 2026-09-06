# Smoothed-density epsilon cross-review

The new law is implemented as declared: only epsilon sees Gaussian-smoothed density; the physical source RHS remains 4 pi G times the original rho. Isotropic physical lengths 0.25 and 0.5 kpc are converted to each axis's voxel sigma correctly. The kernel is truncated at four sigma, normalized on the discrete infinite lattice and uses zero-density exterior padding. No finite-box renormalization is applied.

Independent explicit Gaussian-kernel convolution on actual base and enlarged-box source arrays agrees with the production filter below 1e-12 relative. Source rho checksums are unchanged, and boundary smoothing loss stays below its frozen 0.1% gate. This checks implementation and bookkeeping; it does not establish that the chosen smoothing length represents measured coherence or energy exchange.

All four force-comparison gates replay successfully from saved samples:

| New response length | Fine-to-finer vector RMS | Base-to-larger-box RMS |
|---|---:|---:|
| 0.25 kpc | 3.10% | 1.06% |
| 0.50 kpc | 3.40% | 1.10% |

Each height group also passes its frozen 8% gate; overall tolerance is 5%. All eight reported PDE residual/flux checks pass. Full per-height results and the maximum smoothing mass-loss fraction are retained in smoothed-field-receipt.json.

This is a meaningful numerical result: both explicitly modified, spatially averaged density-response laws resolve under the tested meshes and boxes, whereas the original point-density epsilon law still fails its 13.17% refinement comparison. The old failure is not superseded or reclassified. Numerical success of these two new laws does not validate their physical scale, observational predictions or three-dimensional source truth. No measured velocities or lensing responses were opened.

The cross-review independently verifies Gaussian source processing and saved force-comparison arithmetic, not every large-grid solve from scratch. The separately assembled sparse manufactured operator and previously retained face-flux reviews support the solver implementation. Original source geometry, conversions, conditional monopole boundary and unobserved-source limitations remain unchanged.
