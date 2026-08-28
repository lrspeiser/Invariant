# Gravity roadmap Item 2: W1 multipole second-attempt result

## Outcome

The second Item 2 attempt is **inconclusive**. It successfully builds and replays a
target-blind two-dimensional morphology pipeline on real public images, but the resulting
common multipole grammar does not predict the sealed galaxy/cluster response and fails an
independent morphology check.

The immutable evidence is
`runs/gravity/roadmap/item-02-wise-multipoles-v2.json`, with decision
`INCONCLUSIVE_ITEM2_WISE_MULTIPOLES`. This does not establish a shape cause, a new gravity
law, historical novelty, or an alternative to GR. Item 2 remains open.

## What was added

The experiment downloaded unWISE NEO11 W1 cutouts for all 83 preregistered SPARC
exploration galaxies with published inclination at or below 65 degrees. Before joining any
gravity target, it froze a target-blind image-quality rule and extracted:

- central concentration and centroid shift;
- a deprojected complex quadrupole amplitude;
- third- and fourth-order aperture multipoles;
- a combined multipole-energy statistic and fixed transforms;
- an external match to published S4G bar family and bar ellipticity.

All 83 cutouts, URLs, image hashes, and quality flags are sealed in
`runs/gravity/roadmap/item-02-wise-multipoles-v2-source/unwise-neo11-w1-manifest.json`.
Seventy-nine images produced finite measurements, and 68 passed the frozen quality gate,
exceeding the minimum of 40. No velocity, Item 1 coefficient, SPARC confirmation target, or
direct lensing likelihood was available to the image feature computation.

The experiment joined those 68 galaxies to the same 20 CLASH cluster objects used in Item
1. Cluster concentration, centroid shift, axis-ratio quadrupole, and published X-ray power
ratios were converted to the fixed common feature grammar. The final test covers 88 real
objects, 1,395 galaxy rotation-curve points, and 84 cluster diagnostic points.

## Fixed real-data test

The unchanged development parent remained

```text
g = g_bar + beta (g_bar chi + g_dagger psi).
```

Fourteen fixed model classes and two nested selectors produced 16 tested formula classes.
Five outer folds held out whole objects and were stratified by galaxy versus cluster. The
universal selector could use a constant or one of eleven qualifying morphology models, but
could not use the binary support/population proxy. A second diagnostic selector could see
all models.

The preregistered gates required the universal morphology selector to:

1. choose a qualifying morphology model in every fold;
2. improve observational score and coefficient prediction inside both populations;
3. beat the binary support proxy inside both populations and their shared multipole-energy
   interval;
4. have positive coefficient R² in 33 high-quadrupole galaxies;
5. reproduce the positive sign of an independent published S4G bar statistic.

No gate was adjusted after the sealed Item 1 coefficient labels were joined.

## Measured result

The universal selector chose concentration, centroid-shift, or
concentration–multipole-energy models across the five folds. It selected a qualifying
morphology model in every fold, but its predictions were worse than the baselines:

| Predictor | Galaxy beta MSE / R² | Cluster beta MSE / R² | Galaxy / cluster chi-square |
|---|---:|---:|---:|
| Global constant | 0.647 / -7.209 | 0.751 / -3.491 | 1,220,059 / 578.331 |
| Nested W1/X-ray morphology | 0.501 / -5.358 | 0.919 / -4.497 | 1,253,072 / 749.161 |
| Binary support/population proxy | 0.080 / -0.022 | 0.208 / -0.245 | 96,885 / 156.743 |
| Support proxy plus all multipoles | 0.103 / -0.308 | 0.264 / -0.579 | 100,235 / 169.483 |

When the diagnostic selector was allowed to use every model, it chose the binary support
proxy in all five folds. Adding all multipoles to that proxy made its result worse rather
than better.

The shared multipole-energy interval is `0.0292–0.2837`, containing 48 galaxies and all 20
clusters. Within that overlap, morphology coefficient MSE is `0.556` for galaxies and
`0.919` for clusters, versus `0.090` and `0.208` for the support proxy. Among the 33
high-quadrupole galaxies, morphology R² is `-3.319`.

The target-blind external check also fails its primary sign gate. For 22 quality-passing
galaxies, W1 quadrupole versus the published S4G visual bar-family scale has Spearman
`rho=-0.296`. The secondary bar-ellipticity match is positive (`rho=0.274`) but has only ten
objects, below the frozen minimum of 20, so it cannot replace the failed primary check.

## What this proves and does not prove

This run materially improves the system: it can now acquire public images, measure
two-dimensional multipoles without target leakage, reject unusable images by a frozen
quality contract, validate them against an independent morphology catalog, run nested
whole-object tests, and retain object-level counterexamples and replay hashes.

It also excludes a concrete equation family: global linear, quadratic, logarithmic, and
fixed interaction models built from the current normalized W1/X-ray concentration,
centroid, and low-order multipoles do not generate the required cross-scale coefficient.
The failure holds within galaxies, within clusters, in their feature-overlap region, and in
the high-quadrupole galaxy subset.

It does **not** exclude all anisotropic gravity. W1 light is not total baryonic mass, X-ray
emissivity is a different tracer, foreground contamination can survive the quality gate,
and bars are not a genuinely filamentary third population. The current result therefore
narrows the search rather than closing Item 2.

## Required next test

Remain on Item 2. A third attempt must improve the physical comparability of the shape
inputs or add a genuinely intermediate/filamentary population. Good candidates are
resolved stellar-plus-gas mass maps for galaxies, gas-density rather than X-ray-brightness
cluster morphology, or public filament/lens systems with a preregistered mapping into the
same tensor invariants. It must retain every current failed region and beat the support
proxy inside each population and overlap before any confirmation sample is opened.
