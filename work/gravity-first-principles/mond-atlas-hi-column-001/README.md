# Actual HI column, pressure and three-dimensional emission packet

The primary NGC2976 packet is built from the existing positive bilinear HI source, with checked pressure gradients, exact integrated cell masses and full three-dimensional native-pixel positions. No observed cube spectrum, velocity or gravity response was opened. **SOURCE_BLOCKED remains for observed scoring:** this completes source inputs, not the combined source/force/instrument comparison.

Use these private artifacts, whose hashes are in the public receipts:

- `work/private/mond-atlas-hi-column-001/closure001/f4-pressure-closure.npz`: raw HI column, pressure estimate and its derivative, radius and support flags.
- `work/private/mond-atlas-hi-column-001/centroid-projection001/f4-emission-0.0625.npz`:902,592 three-dimensional emitter nodes with `xy_pixel`, `flux_jy_km_s`, `radius_kpc`, `phi_rad`, `z_kpc`, and `force_table_supported`. No LOS velocity is invented.
- `work/private/mond-atlas-hi-column-001/centroid002/f4-emission-0.0625.npz`: factorized18,804 planar mass centroids and48 vertical nodes for efficient grouped rendering.

`xy_pixel` uses zero-based native FITS coordinates. Each flux is integrated Jy km/s for pure neutral hydrogen, before beam, channel or primary-beam operations. The full column has assumed exponential height0.2kpc; the projected packet is not a thin-sheet replacement. Height-independent circular velocity remains an explicit later column-closure assumption.

## Actual checks and numerical results

Primary f4 source:133,998,525.86Msun of pureHI, or182,237,995.17Msun including the inherited1.36 helium factor. Its conditional intrinsic integrated flux is43.61264677Jy km/s. These are reconstructions under the old map, masking and conversion assumptions, not a new independent mass measurement. The f1 source comparison has134,933,701.14Msun ofHI and43.91701930Jy km/s.

The absolute coefficient derived by inverting the exact inherited column-density conversion is235,631.09407Msun/(Mpc² Jy km/s). It differs by+0.01320% from235,600 and-0.15632% from the rounded236,000 in [Walter et al.2008, equation3](https://arxiv.org/html/0810.2125). The difference is retained; no coefficient was adjusted to match a desired flux. The independent beam-area/intensity calculation reproduces the same coefficient to numerical precision. The source excludes helium from HI line emission and never applies another beam-area or primary-beam factor upstream. The original flux-rescaled, primary-beam-corrected MOM0 is still different from the standard CLEAN cube.

Inside0.75–2.5kpc, the f4 pressure-column refinement changes by0.02384% RMS and its derivative by0.06341% RMS. The largest normalized local derivative change is1.1488%, below the frozen10% gate. An independent finite difference of the smoothed column agrees with the analytic convolution derivative to1.92e-8 relative RMS. Gaussian analytic/direct-quadrature controls pass before source arrays. A separate interpolation implementation agrees with the actual bilinear source to1.53e-13 absolute. Polar integration agrees with exact source mass to0.00305%; emitted cell masses conserve total mass to machine precision without renormalization.

Exact mass centroids were independently checked by Gaussian quadrature split at every bilinear knot: maximum position disagreement2.67e-15kpc, relative cell-mass disagreement2.91e-16. Independent native SIN projection agrees with Astropy within9.0e-11pixels. Three-dimensional flux sums retain the planar flux within3.4e-16 relative. The two-sided exponential quadrature passes its normalization and first/second moments; full Laplace tails are represented, not truncated and renormalized empirically.

## Explicit pressure closure

The current contract is

`Pi(R)=sigma_reference² * Sigma_smoothed(R)`,

`vphi²=R*gbar + R*Pi'(R)/Sigma_raw(R)`.

The denominator and the force's HI angular/vertical weights use the raw source column. Only the pressure estimate is smoothed with a0.25kpc radial Gaussian, using even extension at the center and zero source continuation beyond8kpc. The5/10/15km/s values are **pressure-normalization labels**, not constant local dispersions and not fitted spectral widths. In the primary inner region, local `Pi/Sigma_raw` divided by that normalization ranges0.80467–1.38485. A consumer must not substitute the old ambiguously named smoothed column as the Euler denominator.

The f4 raw HI column is zero atR=.025kpc, although its smoothed pressure column is positive. That row is explicitly undefined for a raw HI-weighted force and must not acquire an invented force. The requested force grid begins at.05kpc. One-dimensional radial pressure smoothing changes a hypothetical2piR-weighted integral by+0.5191%; this smoothed profile is never used as the emission mass. The genuine unsmoothed mass is preserved.

## Failures, tails and remaining limits

The first emitter version assigned exact cell mass to geometric cell centers. That put tiny amounts atR=.044194 where the actual fine bilinear source is zero. The original packets and nonzero aperture-influence audit remain. A versioned exact-mass-centroid correction fixes the primary fine grid, whose radii are.07365696–6.01215854kpc and all sampled underlying HI values are positive.

The f4 coarse0.125kpc centroid case still fails:16 centroids land in subcell holes, containing0.02193% of mass. This is retained as a failed placement test; it is not presented as an admitted coarse comparison. Both f1 centroid cases pass. The final fine packet therefore supplies usable source nodes, but a fully admitted spectral quadrature convergence claim still needs a valid finer comparator or another independently verified emission integration. No actual-spectrum refinement was performed here.

The final f4 packet has2.2081e-6 of its flux beyond the oldR=6 force-table boundary. Full three-dimensional projection gives a nonzero six-sigma native-aperture influence: the largest integrated weighted flux is2.42666e-14Jy km/s/beam, at most9.663e-13 of an aperture's flux. All contributions remain in the packet. The force operator can extend over positive source support, or the downstream prediction can retain an explicit velocity-independent error bound; there is no silent zero-velocity assignment or deletion. The old integrated-pixel versus corrected native-sampled aperture convention changes actual source-weighted flux by up to0.07509%; this is a source-only comparison, not observed motion evidence.

Missingness remains material: the original generic HI map has approximately21.50% unobserved/partial area inside6kpc; its positive-clipping addition, zero-fill and annular-fill integrals are preserved in [source-alternatives-and-missingness.json](source-alternatives-and-missingness.json). The positive latent fit used trusted measured cells, not the generic annular-filled image. Zero/annular maps are not interchangeable completed3D inverse alternatives. Common30arcsec source smoothing and corresponding force/source replays are also not completed by this pressure smoothing step. Stellar/CO alternatives affect gravity while leaving the HI tracer unchanged. HI mass/flux scale interpretations must remain explicit; mass-conversion uncertainty need not imply a changed measured line flux.

No self-absorption correction, unique gas depth, source noise posterior, pressure tensor measurement, general streaming equilibrium or complete instrument likelihood is claimed. New private storage is91,243,307bytes before any later downstream products; no new raw downloads. Public receipts preserve the original run, closure addendum, failed centroid assertion, corrected primary packet and all measured limitations. Parent owns force/scorer integration and publication.

The photometric PA has a180degree rotational-sign ambiguity. A prospective training-only spin+/- rule must be declared before any observed comparison; no spin or actual-source LOSvelocity was chosen in this packet.
