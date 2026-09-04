# SCHEMA — the gravitational scene graph

Generated from `scene_results.json` by `write_report.py`. The authoritative definitions are in `schema.py`, `metadata.py` and `registry.py`; this file is the readable rendering of them.

## Node types

The charter's node bullets, and the schema types implementing them.

| Charter bullet | Schema type(s) |
|---|---|
| Stars or stellar tracer populations | `star_population` |
| Galaxies | `galaxy` |
| Gas cells or voxels | `gas_cell` |
| Central galaxies | `central_galaxy` |
| Intracluster light | `intracluster_light` |
| Black holes | `black_hole` |
| Compact substructures | `compact_substructure` |
| Background lensed sources | `background_source` |
| Observer | `observer` |
| Instrument | `instrument` |
| Voids, filaments, saddles, and boundaries | `void`, `filament`, `saddle`, `boundary` |
| Latent field cells in a candidate universe | `latent_field_cell` |

## Edge types

- `spatial_separation`
- `relative_velocity`
- `membership`
- `light_path`
- `source_source`
- `tidal_pair`
- `orbital`
- `image_family`
- `causal_retarded`
- `shared_covariance`

## Field types

- `matter_density`
- `energy_stress`
- `velocity`
- `temperature_pressure`
- `em_state`
- `candidate_gravitational_state`
- `candidate_vacuum_state`
- `measurement_selection`

## The metadata contract

Every quantity carries all 17 items. A quantity that violates the contract raises `ContractError` at construction, so a malformed quantity can never reach the compiler.

| Charter item | Fields | Conditional | Complete |
|---|---|---|---|
| physical name and operational definition | `name`, `definition` | no | yes |
| scalar/vector/tensor/graph/path/history type | `kind`, `rank` | no | yes |
| units | `dim` | no | yes |
| coordinate and reference frame | `frame` | no | yes |
| transformation under translation/rotation/boost/parity/time reversal | `translation`, `rotation`, `boost`, `parity`, `time_reversal` | no | yes |
| point/region/path/interval it belongs to | `support` | no | yes |
| measurement or derivation source | `source` | no | yes |
| direct/derived/latent/nuisance status | `status` | no | yes |
| resolution and smoothing scale | `resolution_m`, `smoothing_m` | yes | yes |
| full uncertainty and covariance group | `uncertainty`, `covariance_group` | no | yes |
| boundary or gauge convention | `gauge` | yes | yes |
| coarse-graining behaviour | `coarse_grain` | no | yes |
| causal availability | `causal` | no | yes |
| completeness and selection function | `completeness`, `selection` | no | yes |
| allowed operations | `allowed_ops` | no | yes |
| known algebraic dependencies | `depends_on`, `exact_identities` | no | yes |
| independently measurable in the test sample | `independently_measurable`, `measurability_note`, `identifiability` | no | yes |

## The ontology

67 quantities. `id.` is the identifiability class; `cg` is coarse-graining behaviour.

| Quantity | § | Units | Kind | Status | id. | cg | Gauge |
|---|---|---|---|---|---|---|---|
| `G` | 16 | M^-1 L^3 T^-2 | scalar | constant | measured | extensive |  |
| `M_enc` | 4 | M | scalar | derived | constructible | extensive |  |
| `P_e` | 3 | M L^-1 T^-2 | scalar | derived | constructible | nonlinear |  |
| `R500` | 1 | L | scalar | derived | constructible | nonlinear |  |
| `T_x` | 3 | Theta | scalar | direct | measured | nonlinear |  |
| `a0` | 16 | L T^-2 | scalar | constant | measured | extensive |  |
| `alignment_angle` | 8 | 1 | scalar | invariant_descriptor | constructible | nonlinear |  |
| `axis_ratio_q` | 4 | 1 | scalar | direct | measured | nonlinear |  |
| `c_light` | 16 | L T^-1 | scalar | constant | measured | extensive |  |
| `distance` | 17 | L | scalar | nuisance | marginalisable | nonlinear |  |
| `e1` | 13 | 1 | scalar | direct | measured | intensive_linear |  |
| `e2` | 13 | 1 | scalar | direct | measured | intensive_linear |  |
| `ext_axis` | 8 | 1 | vector | invariant_descriptor | measured | scale_defined |  |
| `field_memory` | 15 | T | scalar | latent | non_identifiable | nonlinear |  |
| `g_N` | 6 | L T^-2 | scalar | derived | constructible | nonlinear |  |
| `g_total` | 6 | L T^-2 | scalar | derived | constructible | nonlinear |  |
| `g_vec` | 6 | L T^-2 | vector | derived | constructible | nonlinear |  |
| `graph_degree` | 10 | 1 | scalar | invariant_descriptor | constructible | topological |  |
| `image_position` | 13 | 1 | vector | direct | measured | intensive_linear |  |
| `kappa` | 13 | 1 | scalar | derived | constructible | nonlinear |  |
| `m_bh` | 2 | M | scalar | derived | marginalisable | extensive |  |
| `m_gas_hot` | 2 | M | scalar | derived | measured | extensive |  |
| `m_h2` | 2 | M | scalar | derived | marginalisable | extensive |  |
| `m_hi` | 2 | M | scalar | derived | marginalisable | extensive |  |
| `m_icl` | 2 | M | scalar | derived | marginalisable | extensive |  |
| `m_star` | 2 | M | scalar | derived | measured | extensive |  |
| `mass` | 2 | M | scalar | derived | constructible | extensive |  |
| `n_e` | 2 | L^-3 | scalar | derived | measured | intensive_linear |  |
| `n_wells` | 10 | 1 | scalar | derived | constructible | catalogue_dependent |  |
| `p_member` | 17 | 1 | scalar | nuisance | measured | nonlinear |  |
| `path_density` | 11 | M L^-2 | scalar | invariant_descriptor | constructible | intensive_linear |  |
| `path_void_fraction` | 11 | 1 | scalar | invariant_descriptor | constructible | nonlinear |  |
| `phi_depth_r500` | 6 | L^2 T^-2 | scalar | derived | constructible | nonlinear | yes |
| `phi_depth_saddle` | 6 | L^2 T^-2 | scalar | derived | constructible | nonlinear | yes |
| `phi_depth_scaleradius` | 6 | L^2 T^-2 | scalar | derived | constructible | nonlinear | yes |
| `phi_depth_volume` | 6 | L^2 T^-2 | scalar | derived | constructible | nonlinear | yes |
| `phi_lensing` | 7 | L^2 T^-2 | scalar | latent | non_identifiable | nonlinear | yes |
| `phi_slip` | 7 | L^2 T^-2 | scalar | latent | non_identifiable | nonlinear | yes |
| `position_angle` | 4 | 1 | pseudoscalar | direct | measured | nonlinear |  |
| `psf_fwhm` | 17 | L | scalar | nuisance | marginalisable | nonlinear |  |
| `r_3d` | 1 | L | scalar | latent | marginalisable | intensive_linear |  |
| `r_e` | 4 | L | scalar | derived | measured | nonlinear |  |
| `r_proj` | 1 | L | scalar | direct | measured | intensive_linear |  |
| `rho_env` | 9 | M L^-3 | scalar | invariant_descriptor | constructible | scale_defined |  |
| `rho_star` | 2 | M L^-3 | scalar | derived | measured | intensive_linear |  |
| `sersic_n` | 4 | 1 | scalar | derived | measured | nonlinear |  |
| `shear_m` | 17 | 1 | scalar | nuisance | marginalisable | nonlinear |  |
| `sigma_star` | 5 | L T^-1 | scalar | direct | measured | nonlinear |  |
| `sigma_turb` | 3 | L T^-1 | scalar | derived | marginalisable | nonlinear |  |
| `smoothing_scale` | 1 | L | scalar | nuisance | measured | scale_defined |  |
| `t` | 1 | T | scalar | derived | constructible | intensive_linear | yes |
| `t_since_merger` | 12 | T | scalar | derived | marginalisable | nonlinear |  |
| `tidal_anisotropy` | 6 | 1 | scalar | invariant_descriptor | constructible | nonlinear |  |
| `tidal_tensor` | 6 | T^-2 | tensor2 | invariant_descriptor | constructible | nonlinear |  |
| `time_delay` | 13 | T | scalar | direct | measured | nonlinear |  |
| `upsilon_star` | 17 | 1 | scalar | nuisance | marginalisable | nonlinear |  |
| `v_circ` | 5 | L T^-1 | scalar | derived | constructible | nonlinear |  |
| `v_los` | 5 | L T^-1 | scalar | direct | measured | intensive_linear |  |
| `v_x` | 5 | L T^-1 | scalar | latent | marginalisable | intensive_linear |  |
| `v_y` | 5 | L T^-1 | scalar | latent | marginalisable | intensive_linear |  |
| `v_z` | 5 | L T^-1 | scalar | direct | measured | intensive_linear |  |
| `vacuum_axis` | 15 | 1 | vector | latent | non_identifiable | nonlinear |  |
| `vacuum_order` | 15 | 1 | scalar | latent | non_identifiable | nonlinear |  |
| `x` | 1 | L | scalar | direct | measured | intensive_linear |  |
| `y` | 1 | L | scalar | direct | measured | intensive_linear |  |
| `y_compton` | 13 | 1 | scalar | direct | measured | intensive_linear |  |
| `z` | 1 | L | scalar | nuisance | marginalisable | intensive_linear |  |

## Operational definitions

**`G`** — Newton's gravitational constant.

> units `M^-1 L^3 T^-2` · frame `cluster_rest` · support `global` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `EXTENSIVE` · causal `LOCAL_NOW`
>
> *source*: laboratory. *uncertainty*: negligible. *covariance group*: `constants`. *completeness*: n/a. *selection*: n/a.
>
> *measurability* (measured): a GLOBAL constant; an object-specific value would violate charter criterion 3

**`M_enc`** — Baryonic mass enclosed inside a stated radius of the declared centre, summed over resolved components.

> units `M` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `EXTENSIVE` · causal `LOCAL_NOW`
>
> *source*: integral of the resolved source scene. *uncertainty*: component-wise. *covariance group*: `baryon_budget`. *completeness*: above the completeness threshold plus a statistical unresolved population. *selection*: as the component catalogues.
>
> *measurability* (constructible): yes for BARYONS. A dynamical or lensing 'enclosed mass' is a different object and is NOT this one.

**`P_e`** — Electron thermal pressure, n_e k T.

> units `M L^-1 T^-2` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: n_e times T_x, or SZ y deprojection. *uncertainty*: propagated from n_e and T. *covariance group*: `gas_thermo`. *completeness*: as n_e. *selection*: as n_e.
>
> *exact identities*: `P_e = n_e k_B T_x`
>
> *measurability* (constructible): measurable two independent ways (X-ray n_e*T and SZ y), which is what makes it useful

**`R500`** — Radius enclosing 500 times the critical density. RETAINED WITH A WARNING: it is defined through a mass, so any statistic binned in r/R500 that also involves mass shares a denominator with its own axis.

> units `L` · frame `cluster_rest` · support `global` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: DERIVED from a hydrostatic or lensing mass. *uncertainty*: inherits the mass estimator's error and its equilibrium assumption. *covariance group*: `mass_estimator`. *completeness*: n/a. *selection*: n/a.
>
> *exact identities*: `M(<R500) = 500 rho_crit (4/3) pi R500^3`
>
> *measurability* (constructible): NOT a raw observable. Flagged by this programme's R500 tautology audit (Run AT).
>
> **DERIVED UNDER A THEORY — scoring a candidate law against this is circular.**
>
> *note*: shared-denominator hazard: see shared-denominator-artefacts

**`T_x`** — X-ray spectroscopic gas temperature.

> units `Theta` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: spectral fit to the X-ray continuum and line ratios. *uncertainty*: counts-limited; unusable below ~150 counts in R500. *covariance group*: `xray_spectral`. *completeness*: inside the X-ray detection radius. *selection*: requires enough counts for a spectral fit.
>
> *measurability* (measured): yes -- a spectroscopic measurement, theory-independent
>
> *note*: spectroscopic-like weighting is nonlinear in T, so a temperature read off an averaged spectrum is NOT the average temperature

**`a0`** — Candidate universal acceleration scale.

> units `L T^-2` · frame `cluster_rest` · support `global` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `EXTENSIVE` · causal `LOCAL_NOW`
>
> *source*: fitted globally, once. *uncertainty*: global fit. *covariance group*: `constants`. *completeness*: n/a. *selection*: n/a.
>
> *measurability* (measured): a GLOBAL constant; an object-specific value would violate charter criterion 3

**`alignment_angle`** — Angle between two declared axes (e.g. a galaxy's angular-momentum axis and the local filament axis).

> units `1` · frame `cluster_rest` · support `pair` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: from two independently measured axes. *uncertainty*: propagated from both axes. *covariance group*: `environment+morphology`. *completeness*: where both axes exist. *selection*: joint.
>
> *measurability* (constructible): yes IF both axes come from different data; an angle between an axis and itself is not a variable
>
> *note*: parity: cos(angle) is EVEN; a SIGNED angle would be ODD and would need a handedness convention

**`axis_ratio_q`** — Projected minor-to-major axis ratio of the light.

> units `1` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: second moments or a Sersic fit. *uncertainty*: PSF-dependent at small size. *covariance group*: `morphology_fit`. *completeness*: all detected members. *selection*: detection limited.
>
> *measurability* (measured): available for all seven target clusters

**`c_light`** — Speed of light in vacuum.

> units `L T^-1` · frame `cluster_rest` · support `global` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `EXTENSIVE` · causal `LOCAL_NOW`
>
> *source*: laboratory. *uncertainty*: negligible. *covariance group*: `constants`. *completeness*: n/a. *selection*: n/a.
>
> *measurability* (measured): a GLOBAL constant; an object-specific value would violate charter criterion 3

**`distance`** — Angular-diameter distance to the system.

> units `L` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: measurement model. *uncertainty*: estimator specific. *covariance group*: `cosmology`. *completeness*: n/a. *selection*: n/a.
>
> *measurability* (marginalisable): must be MARGINALISED, never promoted to a gravity variable without evidence

**`e1`** — First component of the measured background-source ellipticity, in the declared sky frame.

> units `1` · frame `sky_equatorial` · support `point` · translation `INVARIANT` · rotation `FRAME_DEPENDENT` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `INTENSIVE_LINEAR` · causal `PAST_LIGHT_CONE`
>
> *source*: MEASURED: a shape estimator on calibrated pixels. *uncertainty*: shape noise ~0.25-0.30 per component, plus a multiplicative shear calibration. *covariance group*: `shear_calibration`. *completeness*: source-detection limited. *selection*: size and S/N cuts; the selection is itself a shear-dependent bias.
>
> *measurability* (measured): THE raw lensing observable. A convergence map is not: it is this, plus an inversion that assumes a gravity law.

**`e2`** — Second component of the measured background-source ellipticity, in the declared sky frame.

> units `1` · frame `sky_equatorial` · support `point` · translation `INVARIANT` · rotation `FRAME_DEPENDENT` · boost `INVARIANT` · parity `ODD` · time reversal `EVEN` · coarse-graining `INTENSIVE_LINEAR` · causal `PAST_LIGHT_CONE`
>
> *source*: MEASURED: a shape estimator on calibrated pixels. *uncertainty*: shape noise ~0.25-0.30 per component, plus a multiplicative shear calibration. *covariance group*: `shear_calibration`. *completeness*: source-detection limited. *selection*: size and S/N cuts; the selection is itself a shear-dependent bias.
>
> *measurability* (measured): THE raw lensing observable. A convergence map is not: it is this, plus an inversion that assumes a gravity law.

**`ext_axis`** — Unit vector of the externally imposed preferred axis (cluster-centre direction, filament axis, or principal tidal eigenvector), whichever the candidate declares.

> units `1` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `VECTOR` · boost `INVARIANT` · parity `ODD` · time reversal `EVEN` · coarse-graining `SCALE_DEFINED` · causal `LOCAL_NOW`
>
> *source*: from the environment reconstruction, INDEPENDENTLY of the probe whose response is being tested. *uncertainty*: depends on the environment catalogue depth. *covariance group*: `environment`. *completeness*: requires a surrounding-structure catalogue, which is the layer most often missing. *selection*: survey footprint around the line of sight.
>
> *measurability* (measured): THIS IS THE GATE-1 CRUX. A constant response tensor is a coordinate stretch UNLESS its axis is fixed by an independently measured direction misaligned with the probe's radial direction.

**`field_memory`** — Accumulated exposure of a region to a declared environmental state, with a declared decay time.

> units `T` · frame `cluster_rest` · support `interval` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `DISSIPATIVE` · coarse-graining `NONLINEAR` · causal `PAST_LIGHT_CONE`
>
> *source*: history integral generated by a candidate universe. *uncertainty*: model. *covariance group*: `candidate_field`. *completeness*: n/a. *selection*: n/a.
>
> *measurability* (non_identifiable): a memory law must predict how the effect DECAYS; 'history matters' is not a law

**`g_N`** — Magnitude of the Newtonian acceleration generated by the RESOLVED BARYONIC scene at the evaluation point.

> units `L T^-2` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: solved from the resolved baryonic source scene. *uncertainty*: inherits the baryon budget and the scene ensemble. *covariance group*: `baryon_budget`. *completeness*: as the source scene. *selection*: as the source scene.
>
> *exact identities*: `g_N = G M_enc / r_3d^2  (spherical case only)`
>
> *measurability* (constructible): constructed, not observed. In the SPHERICAL case it is an exact function of (M_enc, r_3d), so a law reading all three has two directions, not three -- this is the collapse the rank test finds.

**`g_total`** — Magnitude of the total gravitational acceleration predicted by the candidate law on this scene.

> units `L T^-2` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: candidate law applied to the scene. *uncertainty*: law + scene. *covariance group*: `prediction`. *completeness*: n/a. *selection*: n/a.
>
> *measurability* (constructible): a prediction

**`g_vec`** — Gravitational acceleration vector at a point.

> units `L T^-2` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `VECTOR` · boost `COVARIANT` · parity `ODD` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: candidate law applied to the scene. *uncertainty*: law + scene. *covariance group*: `prediction`. *completeness*: n/a. *selection*: n/a.
>
> *measurability* (constructible): a prediction; only its EFFECT on tracers and photons is observable

**`graph_degree`** — Weighted degree of a well in the source network, at a declared linking length.

> units `1` · frame `cluster_rest` · support `graph` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `TOPOLOGICAL` · causal `LOCAL_NOW`
>
> *source*: constructed from the member catalogue and a linking rule. *uncertainty*: linking-length dependent. *covariance group*: `catalogue_partition`. *completeness*: as the member catalogue. *selection*: as above.
>
> *measurability* (constructible): changes discretely under a merge or split, so it must be shown convergent before use

**`image_position`** — Sky position of one image in a multiply-imaged family.

> units `1` · frame `sky_equatorial` · support `point` · translation `COVARIANT` · rotation `FRAME_DEPENDENT` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `INTENSIVE_LINEAR` · causal `PAST_LIGHT_CONE`
>
> *source*: MEASURED: image astrometry. *uncertainty*: 0.1-0.5 arcsec including identification ambiguity. *covariance group*: `astrometry`. *completeness*: identified families only. *selection*: requires a spectroscopic or colour-based family identification, which is itself model-informed.
>
> *measurability* (measured): the positions are raw; the FAMILY ASSIGNMENT sometimes is not, and that distinction must be carried

**`kappa`** — Convergence. A DERIVED product: the shape catalogue inverted under an assumed relation between shear and surface density.

> units `1` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `PAST_LIGHT_CONE`
>
> *source*: DERIVED UNDER A THEORY from e1, e2. *uncertainty*: mass-sheet degenerate. *covariance group*: `lens_model`. *completeness*: as the shape catalogue. *selection*: as the shape catalogue.
>
> *measurability* (constructible): NOT a raw observation. Charter: do not score 'a precomputed convergence map ... as though it were the primitive observation.'
>
> **DERIVED UNDER A THEORY — scoring a candidate law against this is circular.**
>
> *note*: DM-PRESUPPOSING when the lens model assigns a halo to each cluster galaxy by construction (e.g. the CATS Frontier Fields maps), which makes it circular for any does-lensing-follow-light test

**`m_bh`** — Central black-hole mass.

> units `M` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `EXTENSIVE` · causal `LOCAL_NOW`
>
> *source*: M-sigma or a resolved kinematic measurement. *uncertainty*: component-specific. *covariance group*: `m_bh_calibration`. *completeness*: above the catalogue limit; an unresolved population is carried statistically. *selection*: magnitude/flux limited.
>
> *measurability* (marginalisable): component-specific

**`m_gas_hot`** — Hot ionised intracluster gas mass.

> units `M` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `EXTENSIVE` · causal `LOCAL_NOW`
>
> *source*: X-ray surface brightness deprojection. *uncertainty*: component-specific. *covariance group*: `m_gas_hot_calibration`. *completeness*: above the catalogue limit; an unresolved population is carried statistically. *selection*: magnitude/flux limited.
>
> *measurability* (measured): component-specific

**`m_h2`** — Molecular gas mass from a CO line and a conversion factor.

> units `M` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `EXTENSIVE` · causal `LOCAL_NOW`
>
> *source*: CO flux and alpha_CO. *uncertainty*: component-specific. *covariance group*: `m_h2_calibration`. *completeness*: above the catalogue limit; an unresolved population is carried statistically. *selection*: magnitude/flux limited.
>
> *measurability* (marginalisable): component-specific

**`m_hi`** — Atomic hydrogen mass from the 21 cm line flux.

> units `M` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `EXTENSIVE` · causal `LOCAL_NOW`
>
> *source*: 21 cm integrated flux and a distance. *uncertainty*: component-specific. *covariance group*: `m_hi_calibration`. *completeness*: above the catalogue limit; an unresolved population is carried statistically. *selection*: magnitude/flux limited.
>
> *measurability* (marginalisable): component-specific

**`m_icl`** — Diffuse intracluster stellar mass.

> units `M` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `EXTENSIVE` · causal `LOCAL_NOW`
>
> *source*: deep surface photometry below a stated surface-brightness cut. *uncertainty*: component-specific. *covariance group*: `m_icl_calibration`. *completeness*: above the catalogue limit; an unresolved population is carried statistically. *selection*: magnitude/flux limited.
>
> *measurability* (marginalisable): component-specific

**`m_star`** — Stellar mass from a population-synthesis fit to multi-band photometry at a declared IMF.

> units `M` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `EXTENSIVE` · causal `LOCAL_NOW`
>
> *source*: photometry + stellar population model. *uncertainty*: 0.06 dex global M/L offset with 0.045 dex galaxy-to-galaxy scatter (mid-IR route). *covariance group*: `stellar_ML`. *completeness*: above the catalogue limit; an unresolved population is carried statistically. *selection*: magnitude/flux limited.
>
> *measurability* (measured): yes -- but only up to one GLOBAL nuisance (Upsilon* = 0.5 to 0.06 dex)

**`mass`** — Total gravitating rest mass assigned to the node.

> units `M` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `EXTENSIVE` · causal `LOCAL_NOW`
>
> *source*: sum of the component masses below. *uncertainty*: component-specific. *covariance group*: `mass_calibration`. *completeness*: above the catalogue limit; an unresolved population is carried statistically. *selection*: magnitude/flux limited.
>
> *measurability* (constructible): component-specific

**`n_e`** — Electron number density of the hot gas at a point.

> units `L^-3` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `INTENSIVE_LINEAR` · causal `LOCAL_NOW`
>
> *source*: X-ray surface-brightness deprojection with an emissivity model. *uncertainty*: clumping-corrected; the correction is model dependent. *covariance group*: `xray_deprojection`. *completeness*: inside the X-ray detection radius. *selection*: surface-brightness limited; stops at 0.7-1.1 R500 for the Frontier Fields targets.
>
> *measurability* (measured): yes, and it does NOT presuppose dark matter -- emissivity depends on n_e^2 and T, not on mass

**`n_wells`** — Number of distinct gravitational wells above a declared mass threshold within a declared radius.

> units `1` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `CATALOGUE_DEPENDENT` · causal `LOCAL_NOW`
>
> *source*: counted off the member catalogue. *uncertainty*: Poisson plus the deblending choice. *covariance group*: `catalogue_partition`. *completeness*: above the catalogue threshold. *selection*: detection and deblending.
>
> *measurability* (constructible): NOT independently measurable: its value changes when a deblender splits one galaxy into two. A law reading it must converge under merge/split.

**`p_member`** — Probability the galaxy is a cluster member.

> units `1` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: measurement model. *uncertainty*: estimator specific. *covariance group*: `membership`. *completeness*: n/a. *selection*: n/a.
>
> *measurability* (measured): must be MARGINALISED, never promoted to a gravity variable without evidence

**`path_density`** — Baryonic mass density integrated along a source-observer light path.

> units `M L^-2` · frame `cluster_rest` · support `path` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `INTENSIVE_LINEAR` · causal `PAST_LIGHT_CONE`
>
> *source*: integral of the scene density along the ray. *uncertainty*: dominated by line-of-sight structure outside the cluster, which is usually uncatalogued. *covariance group*: `los_structure`. *completeness*: the foreground/background catalogue is the limiting layer. *selection*: as the line-of-sight catalogue.
>
> *measurability* (constructible): requires a line-of-sight structure catalogue that mostly does not exist at the needed depth

**`path_void_fraction`** — Fraction of the path length spent below a declared density threshold.

> units `1` · frame `cluster_rest` · support `path` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `PAST_LIGHT_CONE`
>
> *source*: from the path density profile and a threshold. *uncertainty*: threshold dependent. *covariance group*: `los_structure`. *completeness*: as path_density. *selection*: as path_density.
>
> *measurability* (constructible): as path_density

**`phi_depth_r500`** — Baryonic potential difference between the evaluation point and a boundary. Boundary rule: zero at the fixed overdensity boundary R500 of the system.

> units `L^2 T^-2` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: solved from the resolved baryonic scene, differenced against the named boundary. *uncertainty*: 0.87 dex spread across the four defensible global rules in this family (Run AH), against a 0.9 dex gate margin -- the rule choice is NOT negligible. *covariance group*: `gauge_rule`. *completeness*: as the source scene. *selection*: the volume rule additionally inherits the survey footprint, which is why it is the weakest of the four.
>
> *boundary rule*: zero at the fixed overdensity boundary R500 of the system
>
> *measurability* (constructible): a constructed quantity; the point of the family is that a candidate whose VERDICT changes across the four rules is flagged

**`phi_depth_saddle`** — Baryonic potential difference between the evaluation point and a boundary. Boundary rule: zero at the nearest gravitational saddle of the baryonic potential (a physically located surface, no free constant).

> units `L^2 T^-2` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: solved from the resolved baryonic scene, differenced against the named boundary. *uncertainty*: 0.87 dex spread across the four defensible global rules in this family (Run AH), against a 0.9 dex gate margin -- the rule choice is NOT negligible. *covariance group*: `gauge_rule`. *completeness*: as the source scene. *selection*: the volume rule additionally inherits the survey footprint, which is why it is the weakest of the four.
>
> *boundary rule*: zero at the nearest gravitational saddle of the baryonic potential (a physically located surface, no free constant)
>
> *measurability* (constructible): a constructed quantity; the point of the family is that a candidate whose VERDICT changes across the four rules is flagged

**`phi_depth_scaleradius`** — Baryonic potential difference between the evaluation point and a boundary. Boundary rule: zero at a fixed multiple (10x) of the baryonic scale radius.

> units `L^2 T^-2` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: solved from the resolved baryonic scene, differenced against the named boundary. *uncertainty*: 0.87 dex spread across the four defensible global rules in this family (Run AH), against a 0.9 dex gate margin -- the rule choice is NOT negligible. *covariance group*: `gauge_rule`. *completeness*: as the source scene. *selection*: the volume rule additionally inherits the survey footprint, which is why it is the weakest of the four.
>
> *boundary rule*: zero at a fixed multiple (10x) of the baryonic scale radius
>
> *measurability* (constructible): a constructed quantity; the point of the family is that a candidate whose VERDICT changes across the four rules is flagged

**`phi_depth_volume`** — Baryonic potential difference between the evaluation point and a boundary. Boundary rule: zero at the edge of the reconstructed environmental volume (survey-footprint dependent).

> units `L^2 T^-2` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: solved from the resolved baryonic scene, differenced against the named boundary. *uncertainty*: 0.87 dex spread across the four defensible global rules in this family (Run AH), against a 0.9 dex gate margin -- the rule choice is NOT negligible. *covariance group*: `gauge_rule`. *completeness*: as the source scene. *selection*: the volume rule additionally inherits the survey footprint, which is why it is the weakest of the four.
>
> *boundary rule*: zero at the edge of the reconstructed environmental volume (survey-footprint dependent)
>
> *measurability* (constructible): a constructed quantity; the point of the family is that a candidate whose VERDICT changes across the four rules is flagged

**`phi_lensing`** — Lensing potential: the combination of metric potentials that deflects a null geodesic, differenced against the same boundary rule as the matter potential.

> units `L^2 T^-2` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `PAST_LIGHT_CONE`
>
> *source*: generated by a candidate universe from its sources. *uncertainty*: model. *covariance group*: `candidate_field`. *completeness*: n/a. *selection*: n/a.
>
> *boundary rule*: same boundary rule as the paired matter potential -- the DIFFERENCE of the two is only meaningful if both use one rule
>
> *measurability* (non_identifiable): only its effect on photons is observed

**`phi_slip`** — Difference between the matter and lensing potentials at the same point under one common boundary rule. Zero iff matter and light see the same geometry.

> units `L^2 T^-2` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `PAST_LIGHT_CONE`
>
> *source*: difference of two candidate-generated potentials. *uncertainty*: model. *covariance group*: `candidate_field`. *completeness*: n/a. *selection*: n/a.
>
> *boundary rule*: the common rule cancels in the difference, which is what makes this quantity gauge-safe where each term alone is not
>
> *measurability* (non_identifiable): constructed. It is the charter's 'Do matter and light see the same geometry?' made into a number.

**`position_angle`** — Position angle of the light's major axis, east of north.

> units `1` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `FRAME_DEPENDENT` · boost `INVARIANT` · parity `ODD` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: second moments or a Sersic fit. *uncertainty*: degenerate as q -> 1. *covariance group*: `morphology_fit`. *completeness*: all detected members. *selection*: detection limited.
>
> *measurability* (measured): yes, but it is a SKY-FRAME angle: an alignment statistic built from it must state its frame

**`psf_fwhm`** — Point-spread-function full width at half maximum, as a physical length at the source.

> units `L` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: measurement model. *uncertainty*: estimator specific. *covariance group*: `psf`. *completeness*: n/a. *selection*: n/a.
>
> *measurability* (marginalisable): must be MARGINALISED, never promoted to a gravity variable without evidence

**`r_3d`** — True three-dimensional separation from the cluster centre.

> units `L` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `INTENSIVE_LINEAR` · causal `LOCAL_NOW`
>
> *source*: LATENT: sqrt(r_proj^2 + z^2) with z sampled, not measured. *uncertainty*: inherits the whole line-of-sight depth posterior. *covariance group*: `los_depth`. *completeness*: as r_proj. *selection*: as r_proj.
>
> *exact identities*: `r_3d^2 = r_proj^2 + z^2`
>
> *measurability* (marginalisable): no external galaxy cluster has a measured member depth; this is the ensemble's job

**`r_e`** — Half-light radius from a Sersic fit.

> units `L` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: Sersic fit to a calibrated image. *uncertainty*: PSF-model dependent. *covariance group*: `morphology_fit`. *completeness*: only where a resolved fit exists. *selection*: HST-resolved members only; absent for A370 and MACS J0717.
>
> *measurability* (measured): yes where fitted; MISSING for 2 of 7 target clusters, which is a real inventory hole

**`r_proj`** — Projected separation from the declared cluster centre on the sky.

> units `L` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `INTENSIVE_LINEAR` · causal `LOCAL_NOW`
>
> *source*: measured: angular separation times an angular-diameter distance. *uncertainty*: centroid plus centre definition; the centre choice dominates (BCG vs X-ray peak vs light centroid). *covariance group*: `astrometry+centre`. *completeness*: complete within the field. *selection*: field footprint.
>
> *measurability* (measured): yes, once a centre convention is declared

**`rho_env`** — Baryonic mass density smoothed over a declared scale, around the evaluation point.

> units `M L^-3` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `SCALE_DEFINED` · causal `LOCAL_NOW`
>
> *source*: environment catalogue convolved with a declared kernel. *uncertainty*: dominated by catalogue completeness, not by shot noise. *covariance group*: `environment`. *completeness*: the binding constraint: a magnitude-limited environment catalogue undercounts with distance. *selection*: survey footprint and depth.
>
> *measurability* (constructible): yes where a surrounding-structure catalogue exists at the required radius

**`rho_star`** — Stellar mass volume density at a point, from a deprojected light profile times a mass-to-light ratio.

> units `M L^-3` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `INTENSIVE_LINEAR` · causal `LOCAL_NOW`
>
> *source*: deprojected surface photometry. *uncertainty*: deprojection is not unique for a triaxial source. *covariance group*: `stellar_ML+deprojection`. *completeness*: above the surface-brightness limit. *selection*: surface-brightness limited.
>
> *measurability* (measured): up to the global M/L and the deprojection

**`sersic_n`** — Sersic index of the light profile.

> units `1` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: Sersic fit to a calibrated image. *uncertainty*: degenerate with sky subtraction at large n. *covariance group*: `morphology_fit`. *completeness*: as r_e. *selection*: as r_e.
>
> *measurability* (measured): yes where fitted

**`shear_m`** — Multiplicative shear calibration bias.

> units `1` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: measurement model. *uncertainty*: estimator specific. *covariance group*: `shear_calibration`. *completeness*: n/a. *selection*: n/a.
>
> *measurability* (marginalisable): must be MARGINALISED, never promoted to a gravity variable without evidence

**`sigma_star`** — Aperture stellar velocity dispersion of a member galaxy from a pPXF fit, inside a stated aperture.

> units `L T^-1` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: pPXF fit to an integral-field spectrum. *uncertainty*: template and aperture dependent. *covariance group*: `ifu_kinematics`. *completeness*: the brightest members only. *selection*: IFU coverage and S/N; ~213 members across four Frontier Fields clusters.
>
> *measurability* (measured): yes, and it is a MEASURED alternative to an assumed sigma-luminosity scaling
>
> *note*: an APERTURE quantity, not a resolved map: it is already an average, so it must pass the commutation gate before being used to stand in for resolved internal kinematics

**`sigma_turb`** — One-dimensional turbulent velocity of the hot gas.

> units `L T^-1` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `SCALAR` · boost `FRAME_FIXED` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: X-ray line broadening (microcalorimeter) or a surface-brightness fluctuation argument. *uncertainty*: large; measured directly for very few clusters. *covariance group*: `gas_kinematics`. *completeness*: rare. *selection*: bright cores only.
>
> *measurability* (marginalisable): not available for the Frontier Fields targets

**`smoothing_scale`** — Declared Gaussian smoothing scale at which a field-type quantity is evaluated. A PHYSICAL candidate variable, not only a technical choice (charter section 1).

> units `L` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `SCALE_DEFINED` · causal `LOCAL_NOW`
>
> *source*: declared by the analysis. *uncertainty*: exact by declaration. *covariance group*: `analysis_choice`. *completeness*: n/a. *selection*: n/a.
>
> *measurability* (measured): it is a declaration, so it is exactly known; the question is whether the RESULT depends on it

**`t`** — Epoch to which the node's state refers, as proper time in the cluster frame.

> units `T` · frame `cluster_rest` · support `point` · translation `SHIFTS_BY_CONSTANT` · rotation `SCALAR` · boost `COVARIANT` · parity `EVEN` · time reversal `ODD` · coarse-graining `INTENSIVE_LINEAR` · causal `LOCAL_NOW`
>
> *source*: derived from redshift and an assumed distance-redshift law. *uncertainty*: cosmology-dependent. *covariance group*: `cosmology`. *completeness*: complete. *selection*: none.
>
> *boundary rule*: zero at the observed cluster epoch (t=0 at the lookback time of the cluster redshift); differences only
>
> *measurability* (constructible): requires a distance-redshift relation, which is one of the things under test

**`t_since_merger`** — Time since the last major merger, from shock-front geometry and gas-star offsets.

> units `T` · frame `cluster_rest` · support `interval` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `ODD` · coarse-graining `NONLINEAR` · causal `PAST_LIGHT_CONE`
>
> *source*: inferred from merger diagnostics. *uncertainty*: factor-of-two at best. *covariance group*: `merger_state`. *completeness*: only for visibly disturbed systems. *selection*: requires a shock or an offset to be detected.
>
> *measurability* (marginalisable): crudely; the gas-star offset is a direct observable but the conversion to a time is not

**`tidal_anisotropy`** — Traceless part of the tidal tensor, normalised by its own Frobenius norm; a pure shape, gauge-free.

> units `1` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: from tidal_tensor. *uncertainty*: as tidal_tensor. *covariance group*: `baryon_budget`. *completeness*: as the source scene. *selection*: as the source scene.
>
> *measurability* (constructible): constructed

**`tidal_tensor`** — Hessian of the baryonic potential, d2Phi/dx_i dx_j.

> units `T^-2` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `RANK2` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: second derivative of the solved baryonic potential. *uncertainty*: second derivatives amplify scene noise. *covariance group*: `baryon_budget`. *completeness*: as the source scene. *selection*: as the source scene.
>
> *exact identities*: `trace(tidal_tensor) = 4 pi G rho_local`
>
> *measurability* (constructible): constructed. Its TRACE is fixed by the local density (Poisson), so trace and density are not two independent variables.
>
> *note*: GATE 3 hazard: sourcing this from the CATALOGUE ROW LIST rather than from the smooth density is the named repair in Run AB and changes the verdict

**`time_delay`** — Measured arrival-time difference between two images of the same source.

> units `T` · frame `cluster_rest` · support `pair` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `ODD` · coarse-graining `NONLINEAR` · causal `PAST_LIGHT_CONE`
>
> *source*: MEASURED: light-curve cross-correlation. *uncertainty*: from the light-curve sampling and microlensing. *covariance group*: `time_delay`. *completeness*: essentially nil: cluster-scale time delays exist for a handful of events in the whole sky. *selection*: requires a variable source behind a cluster.
>
> *measurability* (measured): raw and theory-free, and it is the single strongest matter-light consistency constraint. The problem is that almost none exist.

**`upsilon_star`** — Stellar mass-to-light ratio in the declared band, as a GLOBAL nuisance.

> units `1` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: measurement model. *uncertainty*: 0.06 dex, GLOBAL (0.045 dex galaxy-to-galaxy scatter, so it is one nuisance not N). *covariance group*: `stellar_ML`. *completeness*: n/a. *selection*: n/a.
>
> *measurability* (marginalisable): must be MARGINALISED, never promoted to a gravity variable without evidence

**`v_circ`** — Circular speed implied by the gravitational field at a radius, for a tracer on a circular orbit.

> units `L T^-1` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: derived from a candidate law. *uncertainty*: inherits the law and the scene. *covariance group*: `prediction`. *completeness*: n/a. *selection*: n/a.
>
> *exact identities*: `v_circ^2 = g_total * r_3d`
>
> *measurability* (constructible): a PREDICTION, not an observation

**`v_los`** — Line-of-sight velocity relative to the cluster systemic redshift, from a spectroscopic line centroid.

> units `L T^-1` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `SCALAR` · boost `COVARIANT` · parity `EVEN` · time reversal `ODD` · coarse-graining `INTENSIVE_LINEAR` · causal `LOCAL_NOW`
>
> *source*: measured: spectroscopic redshift minus the systemic value. *uncertainty*: typically 20-150 km/s depending on the instrument. *covariance group*: `spectroscopy`. *completeness*: spectroscopic sample only. *selection*: spectroscopic targeting -- NOT the same selection as the photometric member catalogue.
>
> *measurability* (measured): yes -- one of the cleanest direct observables in the whole scene

**`v_x`** — x-component of the node's velocity in the cluster rest frame.

> units `L T^-1` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `VECTOR` · boost `COVARIANT` · parity `ODD` · time reversal `ODD` · coarse-graining `INTENSIVE_LINEAR` · causal `LOCAL_NOW`
>
> *source*: LATENT: transverse velocity is not measured at cluster distances. *uncertainty*: prior only. *covariance group*: `orbit_prior`. *completeness*: spectroscopic sample. *selection*: targeting.
>
> *measurability* (marginalisable): no: proper motions at z~0.3 are far below any current astrometric capability

**`v_y`** — y-component of the node's velocity in the cluster rest frame.

> units `L T^-1` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `VECTOR` · boost `COVARIANT` · parity `ODD` · time reversal `ODD` · coarse-graining `INTENSIVE_LINEAR` · causal `LOCAL_NOW`
>
> *source*: LATENT: transverse velocity is not measured at cluster distances. *uncertainty*: prior only. *covariance group*: `orbit_prior`. *completeness*: spectroscopic sample. *selection*: targeting.
>
> *measurability* (marginalisable): no: proper motions at z~0.3 are far below any current astrometric capability

**`v_z`** — z-component of the node's velocity in the cluster rest frame.

> units `L T^-1` · frame `cluster_rest` · support `point` · translation `INVARIANT` · rotation `VECTOR` · boost `COVARIANT` · parity `ODD` · time reversal `ODD` · coarse-graining `INTENSIVE_LINEAR` · causal `LOCAL_NOW`
>
> *source*: measured (= v_los). *uncertainty*: spectroscopic. *covariance group*: `spectroscopy`. *completeness*: spectroscopic sample. *selection*: targeting.
>
> *measurability* (measured): yes

**`vacuum_axis`** — Unit vector of a latent preferred-direction field.

> units `1` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `VECTOR` · boost `COVARIANT` · parity `ODD` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: generated by a candidate universe. *uncertainty*: model. *covariance group*: `candidate_field`. *completeness*: n/a. *selection*: n/a.
>
> *measurability* (non_identifiable): if it is dynamically generated by the local source it is degenerate with source shape; only an EXTERNALLY generated axis is identifiable (GATE 1)

**`vacuum_order`** — Scalar order parameter of a polarisable vacuum, normalised so that 0 is the unmodified vacuum.

> units `1` · frame `cluster_rest` · support `region` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `NONLINEAR` · causal `LOCAL_NOW`
>
> *source*: generated by a candidate universe. *uncertainty*: model. *covariance group*: `candidate_field`. *completeness*: n/a. *selection*: n/a.
>
> *measurability* (non_identifiable): NOT fittable independently at every location -- the candidate universe must specify what generates it and how it evolves

**`x`** — Cartesian x-coordinate of the node in the cluster rest frame, origin at the declared centre. from the astrometric image centroid.

> units `L` · frame `cluster_rest` · support `point` · translation `COVARIANT` · rotation `VECTOR` · boost `COVARIANT` · parity `ODD` · time reversal `EVEN` · coarse-graining `INTENSIVE_LINEAR` · causal `LOCAL_NOW`
>
> *source*: measured: astrometric centroid on a calibrated frame. *uncertainty*: 0.10 arcsec centroid -> 0.15-0.64 kpc depending on cluster redshift. *covariance group*: `astrometry`. *completeness*: members above the catalogue magnitude limit. *selection*: magnitude-limited spectroscopic membership.
>
> *measurability* (measured): two independent astrometric solutions agree to 0.026 arcsec

**`y`** — Cartesian y-coordinate of the node in the cluster rest frame, origin at the declared centre. from the astrometric image centroid.

> units `L` · frame `cluster_rest` · support `point` · translation `COVARIANT` · rotation `VECTOR` · boost `COVARIANT` · parity `ODD` · time reversal `EVEN` · coarse-graining `INTENSIVE_LINEAR` · causal `LOCAL_NOW`
>
> *source*: measured: astrometric centroid on a calibrated frame. *uncertainty*: 0.10 arcsec centroid -> 0.15-0.64 kpc depending on cluster redshift. *covariance group*: `astrometry`. *completeness*: members above the catalogue magnitude limit. *selection*: magnitude-limited spectroscopic membership.
>
> *measurability* (measured): two independent astrometric solutions agree to 0.026 arcsec

**`y_compton`** — Compton-y parameter: the line-of-sight integral of the electron pressure, from a calibrated SZ map.

> units `1` · frame `cluster_rest` · support `path` · translation `INVARIANT` · rotation `SCALAR` · boost `INVARIANT` · parity `EVEN` · time reversal `EVEN` · coarse-graining `INTENSIVE_LINEAR` · causal `PAST_LIGHT_CONE`
>
> *source*: MEASURED: a calibrated millimetre map. *uncertainty*: beam-convolved; correlated noise across the beam. *covariance group*: `sz_map`. *completeness*: beam and depth limited. *selection*: SZ significance.
>
> *exact identities*: `y = (sigma_T / m_e c^2) integral P_e dl`
>
> *measurability* (measured): RAW and independent of X-ray emissivity, which is exactly why it is worth having. An INTEGRATED Y_500 is NOT raw: its aperture is defined through a mass.

**`z`** — Cartesian z-coordinate of the node in the cluster rest frame, origin at the declared centre. z is along the line of sight and is NOT measured.

> units `L` · frame `cluster_rest` · support `point` · translation `COVARIANT` · rotation `VECTOR` · boost `COVARIANT` · parity `ODD` · time reversal `EVEN` · coarse-graining `INTENSIVE_LINEAR` · causal `LOCAL_NOW`
>
> *source*: sampled: only a scalar redshift is observed along the line of sight. *uncertainty*: posterior; see ensemble.py -- 1 Mpc of depth makes only 5-9% of the measured dispersion. *covariance group*: `los_depth`. *completeness*: members above the catalogue magnitude limit. *selection*: magnitude-limited spectroscopic membership.
>
> *measurability* (marginalisable): NOT independently measurable: cz = H(z) d + v_pec is one equation in two unknowns, and the Finger-of-God distortion makes inferred depth ANTI-correlate with true 3-D radius

