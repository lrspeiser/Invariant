# MOND predictions and observations: dataset build design

Date: 2026-09-05

Status: source-registry and dataset-design milestone. The accompanying audit checks existing local files. It does not generate new MOND predictions, download new survey observations, or establish complete spatial 3D mass maps.

The objective is the largest defensible comparison of ordinary-matter MOND predictions with observed galaxy motion, with a smaller subset that can discriminate hypotheses about 3D structure. No dark-matter term is included in the proposed prediction branches. Being the world's largest dataset is not an established claim or a substitute for measurable coverage.

## The observational limit

For an external galaxy, an HI or CO cube ordinarily measures two sky coordinates and line frequency/line-of-sight velocity. It does not measure the physical depth of each emitting parcel. Stellar images are also projected. A tilted-ring or volumetric fit constrains possible 3D structures but does not turn unmeasured depth into an observation. See [3D-Barolo](https://arxiv.org/abs/1505.07834).

At the typical 100-500 pc resolution of [THINGS](https://arxiv.org/abs/0810.2125), and approximately 100 pc resolution of [PHANGS-ALMA](https://arxiv.org/abs/2104.07739), individual stars and much gas substructure remain unresolved. The atlas can test resolved cloud/arm organization and gas-versus-stellar fractions; it cannot promise a census of every star's position or all sub-resolution density variations.

Store an ensemble of observationally allowed 3D mass models for each galaxy. Report which conclusions survive that ensemble. If allowed depths or conversion factors reverse a claimed effect, mark the object non-discriminating for that particular hypothesis. Do not discard it from the master catalog.

## Current source inventory and scale

Counts below refer to different products and overlap. They are not additive and are not counts of complete 3D baryon models. The machine-readable source registry carries exact roles and access qualifications.

| Source | Documented scale | Role |
|---|---|---|
| [SPARC](https://astroweb.case.edu/SPARC/) | 175 galaxies | Reproduce the familiar radial benchmark and track development exposure. |
| [THINGS](https://arxiv.org/abs/0810.2125) / [LITTLE THINGS](https://arxiv.org/abs/1208.5834) | 34 / 41 survey targets | Nearby HI morphology and spectra; detailed and dwarf controls. |
| [WALLABY Pilot DR1](https://wallaby-survey.org/data/data-pilot-survey-dr1/) / [DR2](https://wallaby-survey.org/data/data-pilot-survey-dr2/) | 109 / 126 released kinematic models | Scale resolved comparisons. DR2 has 1,760 phase-2 detections and repeats the phase-1 catalog; 80 higher-resolution detections also overlap. |
| [PHANGS-ALMA](https://almascience.nrao.edu/alma-data/lp/PHANGS/) | 90 survey galaxies; 74 in the documented initial delivery | Molecular clouds and CO spectra; verify delivered coverage per object. |
| [PHANGS-MUSE](https://www.eso.org/sci/publications/announcements/sciann17705.html) | 19 galaxies in the nebular release | Independent-tracer motion and gas-state information. |
| [S4G plus extension](https://irsa.ipac.caltech.edu/data/SPITZER/S4G/overview.html) | 2,817 galaxies | Stellar light and structural fits; cleaned stellar maps exist only for qualifying targets. |
| [MaNGA DR17](https://www.sdss4.org/dr17/manga/) | 10,010 unique galaxies in the high-quality sample | Broad population, inner stellar and ionized-gas dynamics; not outer HI coverage. |
| [HI-MaNGA DR3](https://www.sdss.org/dr20/data_access/value-added-catalogs/?vac_id=90) | 6,358 unique MaNGA IDs with observations | Integrated HI detections or limits, not resolved gas maps. |
| [Apertif DR1](https://arxiv.org/abs/2208.05348) | 160 independent target fields | Archive expansion requiring source discovery and processing; field/continuum counts are not galaxy counts. |
| [MIGHTEE-HI / LADUMA study](https://arxiv.org/html/2608.03576v1) | 130 studied galaxies | External-survey extension. MIGHTEE cubes are public; some derived products are available only on author request. Complete access to all inputs was not established here. |
| [BIG-SPARC](https://arxiv.org/abs/2411.13329) | Approximately 4,000 intended galaxies | Reuse the project when a verified release is available. The checked description is work in progress, not proof of a completed downloadable catalog. |

Acquire catalogs and per-object metadata first. Join by stable identifiers, sky position, redshift/distance and angular footprint; retain ambiguous joins for review. Count unique galaxies after crossmatching, and count usable independent resolution elements separately. Include eligible failed fits and nondetections. Do not select galaxies because MOND already fits or fails.

Additional archive expansion can include WHISP, LVHIS, HALOGAS and VLA-ANGST after their product and overlap audits. Gas-poor dwarf and elliptical samples require separate stellar dynamical likelihoods and environment data; selecting only HI detections would not cover all galaxy types. Local Group distance information can help test 3D methods, but does not automatically provide complete gas tomography.

## What is already on this machine

The seed audit rereads and hashes files referenced by:
- `things-cube-acquisition-001/receipt.json`.
- `stellar-co-acquisition-001/receipt.json`.

These are the existing 12-galaxy HI pilot and its stellar/CO assets. Exact current file counts and byte totals are in `local-asset-audit.json`. File identity is distinct from astrophysical validity.

The [latest constrained-matter report](../constrained-matter-findings-001/report.md) has already established why a new atlas needs better source modeling: the pipeline remains a projected conditional cube model, some warped projections have multiple possible intersections, and the checked NGC2903 stellar+HI+CO region is a limited projected footprint. Its HI-plus-helium fraction is only a median 6.5% of nominal modeled matter in 27 selected positions. That fraction is not a whole-galaxy measurement. No complete spatial-3D total-matter reconstruction is admitted by those results.

All these seed galaxies have been used in development. Reusing their pixels cannot create an untouched confirmation sample. The [stellar-mass audit](../stellar-mass-audit-002/summary.json) supplies an additional source-calibration diagnostic, not direct whole-galaxy stellar masses.

## One galaxy record

Use linked tables rather than one enormous flat table:

1. **Galaxy identity:** stable ID, aliases, coordinates, distance posterior, redshift, selection history, survey memberships, target class, development exposure and split/group IDs.
2. **Observation assets:** source URL/DOI, observation ID/date, download hash, units, astrometry, native PSF/beam, spectral response, footprint, masks, noise products, calibration history and repeat-observation relationships.
3. **Baryonic components:** stellar-light likelihood and mass-to-light/IMF uncertainty; HI opacity and missing-flux uncertainty; molecular gas with CO conversion and CO-dark allowances; ionized/hot gas where relevant; helium conventions; remnant/central-black-hole accounting. Upper limits and missing coverage are different states. Dust can constrain gas or stellar attenuation but must not be counted again as the same inferred gas mass.
4. **Geometry ensemble:** posterior samples or weighted alternatives for orientation, disk thickness/flaring, warps, bulge shape, resolved concentrations and exterior matter. Every parameter records its observational constraint or prior and physical resolution.
5. **Gravity predictions:** source-model ID, Newtonian and MOND branch IDs, equation/interpolation-function version, universal constants, external field and boundary settings, grid/solver tolerances, force vectors and projected motion predictions.
6. **Observed comparisons:** raw spectra or line-of-sight stellar velocities, their likelihood/covariance, predicted distributions, residuals and galaxy-level summaries. An inferred circular speed is a derived diagnostic with pressure/geometry assumptions, not direct measured gravity.
7. **Eligibility and limitations:** source coverage by component and radius, resolved scale range, geometry sensitivity, goodness-of-fit checks, reason for every exclusion from a specific test, and source-only/predictive/joint-fit labels.

Keep FITS originals immutable outside Git. Use a compact Parquet catalog and chunked arrays such as Zarr for derived ensembles if benchmarked storage needs justify them. A browseable galaxy page should show observed maps, predicted maps, residuals, uncertainty and missing coverage together.

## Prediction and validation workflow

1. **Metadata and identity audit.** Download small catalogs; deduplicate; join coverage. Verify astrometry against independent stars, native beams, spectral units, missing short-spacing flux and all mask conventions. Acquire source cutouts before survey-wide footprint cubes.
2. **Constrain matter from light and gas measurements.** Combine images through their own instrument responses. Propagate conversion and distance uncertainties. Do not infer stellar masses, gas depth or a CO conversion from the very velocities that will later be called independent gravity tests.
3. **Build allowed 3D scenes.** Use independent geometric constraints where available. Never infer thickness under Newtonian hydrostatic equilibrium and treat it as gravity-independent truth. If dynamics are necessary to constrain geometry, fit geometry jointly under each law and label the resulting score a joint fit.
4. **Calculate actual field solutions.** Start with a validated QUMOND solver and the identical baryonic source for a Newtonian control; cross-check selected cases with AQUAL as a distinct MOND formulation. Record an explicit interpolation function and a single global acceleration scale before confirmation; any fitting of those belongs only in development. Include baryon-constrained exterior fields and uncertain matter beyond the imaging footprint. Do not add independently boosted stars/clouds; MOND is nonlinear. [QUMOND formulation](https://arxiv.org/abs/0911.5464).
5. **Predict telescope data.** Model gas pressure, permitted warps, streaming and lagging components; integrate emission through a finite-thickness volume, including multiple intersections. Convolve with the native spatial and spectral responses. Pressure-supported stars require orbital-distribution modeling; gas physics cannot be assigned to stars. A freely fitted rotation curve is a motion diagnostic and cannot count as a successful gravity prediction.
6. **Evaluate independent information.** Prefer another tracer or independently acquired observation. If only one cube exists, keep a joint likelihood or explicitly conditional held-spectrum score and propagate its input/target dependence. Include channel and spatial covariance, noise nonstationarity and a validated selection mask. Adjacent channels/pixels are not independent tests.
7. **Search residual patterns.** Use resolved mass concentration, gas fraction, cloud organization, radial/vertical structure and independently measured environment. Compare at matched physical resolution and constrain resolution/selection effects. Kinematic coherence estimated from the target velocities is not an independent predictor of those same velocities.
8. **Freeze and transfer.** Select features in development, hold out entire galaxies/groups, and finally transfer across surveys. Preserve the signed direction of residuals; do not count only reduced absolute error. Use multiple-testing controls and report null results. High-quality inclusion must depend on inputs and numerical validity, not a favorable gravity residual.

Synthetic validation must include geometries and motion components absent from the fitting model. Verify solver boundary/grid convergence, mass/flux accounting and recovery under resolution/noise changes. Test whether unmodeled geometry could imitate the proposed new gravity effect. These are needed before a claim about 3D structure.

## Quality tiers and deliverables

- **Population:** integrated quantities or restricted radial profiles; useful for broad scaling tests, not detailed 3D claims.
- **Resolved:** multiple independent spatial elements, source maps and motion likelihoods; model-dependent depth remains explicit.
- **Structure-discriminating:** plausible 3D alternatives give stable predictions for the specific structural hypothesis being tested. This is a hypothesis-specific qualification, not a claim that the entire galaxy is mapped in 3D.

The first deliverable is an actual crossmatched coverage table showing how many galaxies have each required component at each physical resolution. No defensible final sample size can be stated before that join.

A practical staged target is:
1. Prove the end-to-end method on roughly 10-20 already-exposed, well-covered galaxies, including genuine difficult cases and an independent tracer where available.
2. Extend to roughly 100-300 eligible resolved galaxies if the coverage audit supports it.
3. Grow the population catalog to thousands while admitting only the qualifying subset to 3D-structure tests.

These are planning targets, not achieved sample counts or promised access. Do not delay useful projected tests while pretending missing depth has been measured. Prioritize new observations where allowed geometry or missing molecular/stellar coverage, rather than compute, dominates uncertainty.

## Compute and publication

The 5090 can accelerate 3D field solves, volume-to-cube projection and batches of alternative geometries. Benchmark memory, convergence and throughput on the actual pilot before promising runtime. Use CPU reference solutions for numerical checks and CPU catalog work where appropriate. Storage, response calibration and line-of-sight identifiability are likely stronger constraints than GPU availability.

WALLABY documents footprint cubes around 0.5 TB each; fetch cutouts and manage a bounded cache. Publish source registries, exact code/configuration, derived results, hashes and limitations to main at validated milestones. Raw multi-gigabyte observations remain outside Git.

Reproduce the local identity audit from the repository root, choosing a new output filename:

```powershell
& ./work/private/torch-cuda-env/Scripts/python.exe ./scripts/audit_mond_atlas_seed.py --output ./work/gravity-first-principles/mond-atlas-design-001/local-asset-audit-replay.json
```

No full atlas pipeline is implemented or scientifically validated by this design milestone.

