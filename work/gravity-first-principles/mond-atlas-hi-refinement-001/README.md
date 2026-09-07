# Finer actual HI emitter refinement

The new0.03125kpc planar grid passes all source-location, cell-moment and mass/flux checks. Its68,497 positive mass centroids replace18,804 at0.0625kpc while preserving the exact same positive bilinear HI source. No observed spectra or velocities were opened.

Both grids contain133,998,525.857313Msun of pureHI and43.61264677418Jy km/s under the inherited helium/conversion assumptions. Independent cell quadrature agrees with analytic centroids within1.78e-15kpc and mass within2.73e-16 relative. Every centroid lies inside its cell and on positive HI density. All finer radii lie between.07365696 and6.01348492kpc, covered by the existing.05–6.025kpc force table; no endpoint force is invented.

Two-sided exponential vertical quadrature at24 and48nodes per side is supplied. All four combinations were streamed through the native3Dprojection and conserve total flux. The largest contains6,575,712 expanded nodes, but none of those full arrays is saved. The private factorized packets are small and retain the complete source. The old0.0625planar packet replays within2.67e-15; the first128 streamed native positions and fluxes match the old full3Dpacket exactly.

## Consumer interface

`load_planar(path)` and `iter_native_batches(packet, vertical_order=24 or48, max_nodes=65536)` are in `scripts/mond_atlas_hi_refinement.py`. A batch supplies `xy_pixel`, `flux_jy_km_s` and global integer `planar_index`. The planar packet supplies radius and azimuth, allowing native aperture weights to be summed by planar emitter without repeatedly constructing spectral profiles for each vertical node.

- `work/private/mond-atlas-hi-refinement-001/run001/HI-planar-0.0625.npz`
- `work/private/mond-atlas-hi-refinement-001/run001/HI-planar-0.03125.npz`

Native positions are zero-based FITS pixels. Flux is integrated Jy km/s per pureHI node, before native beam or channel response, with no extra primary-beam correction. Azimuth is atan2(minor,major). A later height-independent column model uses `vLOS=v_sys+spin*sin(inclination)*vphi*cos(phi)`; no spin or velocity is assigned here. Supporting pressure remains separate from spectral line width.

The frozen four-case native-spectrum comparison is owned by the clock agent. This source package does not claim spectral convergence from successful mass conservation. Earlier coarse-grid centroid failures remain unchanged, as do unresolved source filling, mass-conversion, instrument and observational-admission limitations. Current disposition remains SOURCE_BLOCKED for observed response scoring. The source conversion follows the inherited [THINGS equations and measurements](https://arxiv.org/html/0810.2125), not a new mass observation.
