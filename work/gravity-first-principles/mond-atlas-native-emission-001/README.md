# Native supplied-emitter spectrum adapter

The usable adapter is **`scripts/mond_atlas_native_emission_sampled.py`**. It turns supplied HI emissivity positions, integrated flux and LOS velocities into the fifteen fixed12x12 aperture spectra, retaining all42 native channels. It does not derive source density or velocities, read observed source-region spectra, fit nuisances or score a gravity model.

All180 final synthetic profile comparisons pass. With rotation, a prescribed inclination/PA warp,20km/s radial streaming and0.4 asymmetric brightness, final256x512→512x1024 emitter refinement gives maximum relative profile L1 **0.001638%** and centroid difference **0.0000277km/s**. Each synthetic source has exactly1Jy km/s supplied flux. Actual source-node quadrature still requires separate refinement; these results do not admit a real galaxy source or likelihood.

Independent scalar pixel summation, numerical Gaussian line-bin integration and separately fitted continuum regression reproduce the point-emitter spectra within **1.34e-15mJy/beam**. Every360 saved refinement row is independently replayed. Expanded vertical nodes and exact grouped-flux rendering agree within the saved controls' numerical tolerance. The group reduction requires the same LOS velocity and intrinsic line sigma for every node in a group; it combines their projected beam-weighted flux, not their positions or velocities.

## Units and operators

Inputs are zero-based native FITS x/y pixel coordinates; native-frame LOS velocity and intrinsic Gaussian sigma in km/s; and integrated **Jy km/s** per emitter. The caller supplies helium/distance/mass conversion and sky projection. The restoring response evaluates the elliptical Gaussian at the144 native pixel centers and averages. Flux-density response is Jy/native-beam; output is exactly1000 times that in **mJy/native-beam**. No extra pixel top-hat, primary-beam correction, moment-map rescaling, opacity or dirty-beam model is applied.

Finite6-sigma square support is normalized using its continuous Gaussian probability. The independently measured normalization is0.9999999987422569. Enclosing sampled-image mass closure, fractional point offsets, aperture tiling and6/7-sigma sensitivity pass their controls. Emitters outside an aperture still contribute through the finite beam halo. Exact Gaussian CDF differences integrate each spectral bin and divide by its velocity width before the declared Hanning/boxcar and continuum operators; no line-center sampling replaces integration. Positive parent/stored spectra and signed continuum-subtracted spectra are available separately.

The native history contains **63 parent channels**, with stored parent indices11–52 giving42 outputs. The earlier request/protocol wording64 is retained as a documented discrepancy; no extra channel was invented. The original header says `VELO-HEL`, `VELREF=258`. Astropy/WCSLIB interprets it as `VRAD`, `BARYCENT`, m/s and agrees with every native linear velocity coordinate. The native sequence decreases by5.152666992km/s per channel and is never reordered. There is no silently applied reference-frame offset. [Astropy AIPS-keyword documentation](https://docs.astropy.org/en/stable/wcs/relax.html) and [spectral interpretation history](https://docs.astropy.org/en/stable/wcs/history.html) document that translation. [Walter et al. 2008](https://arxiv.org/abs/0810.2125) supplies the THINGS measurement context.

The three instrument alternatives are independent boxcar, full Hanning and decimated Hanning, each constructed on this galaxy's own parent grid. The actual archived correlator response, visibility-dependent continuum fitting, dirty/CLEAN residual response and primary beam remain unresolved. These alternatives are conditional operators; delivered observed data must never be processed through them again.

## Callable interface

```python
from mond_atlas_native_emission_sampled import (
    load_instrument, aperture_weights, render_spectra,
    grouped_aperture_flux, render_grouped_spectra,
)
instrument = load_instrument()  # Header/history/aperture geometry only.
prediction = render_spectra(
    xy_pixel, v_los_km_s, flux_jy_km_s, line_sigma_km_s,
    instrument=instrument, branch="boxcar_independent",
    systemic_km_s=0.0, emission_multiplier=1.0,
)
spectra = prediction["spectra_mjy_beam"]  # Shape15x42, aperture CSV order.
```

For many vertical nodes, `grouped_aperture_flux(xy_pixel, flux_jy_km_s, group_ids, instrument, group_count=...)` accumulates aperture-by-planar-group weighted flux one aperture at a time. Then `render_grouped_spectra(grouped_flux, v_los_by_group, sigma_by_group, instrument, ...)` evaluates spectral CDFs once per group. Geometry and weighted flux can be cached while fitting global systemic velocity, intrinsic width and emission multiplier. Groups with identically zero finite-halo weights are omitted from spectral computation, which is exact for the declared beam support and is not observed-data masking.

## Preserved earlier failures

The first implementation, `mond_atlas_native_emission.py`, integrated the Gaussian across continuous aperture area, effectively adding an unestablished pixel top-hat. Its code, protocol and run001 remain preserved and are not the native sampled-pixel API. Its initial source quadrature also failed15 of180 final profiles, maximum3.36% L1. The sampled version uses separately frozen higher emitter quadrature; both new comparisons pass without changing thresholds. On the synthetic sources, the old versus sampled response differs by at most0.000774 per point-emitter weight, while the source-flux-weighted aperture difference relative to peak aperture flux is at most0.00658%. Small differences do not make the conventions interchangeable.

Synthetic spectrum arrays are private, with relocation and hash receipts; no `.npz` is needed in publication. Original spectra/receipts remain unchanged. Run `python scripts/mond_atlas_native_emission_sampled_controls.py` in a fresh output directory for full controls, and `python work/gravity-first-principles/mond-atlas-native-emission-001/independent_review.py` to replay the saved sampled run. No observed source-region spectra have been read by this adapter work.
