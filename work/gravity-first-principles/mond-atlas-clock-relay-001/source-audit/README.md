# Source-only inventory for clock/relay formula exploration

The strongest immediately usable source is the SPARC published radial mass-model table, paired with its observed rotation curve. It permits a restricted radial prediction experiment. It does not supply observed three-dimensional density, direct clock-rate measurements, energy-transfer histories, gravitational shielding measurements, or lensing.

## Exact inputs and source/response distinction

- `configs/sparc_rotation_curves_full_v1.json`: 175 galaxies, 3391 rows. Each row is `[r_kpc, Vobs_km_s, error_Vobs_km_s, Vgas_km_s, Vdisk_km_s, Vbulge_km_s]`. Only Vobs is the motion response; its error describes random/noncircular/asymmetry uncertainty and excludes inclination systematics.
- `configs/sparc_surface_brightness_exploration_v1.json`: 139 previously exposed development galaxies, 2720 matched rows of `[SBdisk, SBbul]` in solar luminosities/pc². No confirmation sample exists in this supplement.
- `work/gravity-first-principles/map-response-metadata-001/SPARC_Lelli2016c.mrt`: published Table 1 provides distance and error, inclination and error, luminosity, disk scale length, HI mass, and quality. Vflat is a response column and must not be used as a source predictor. The inventory does not retain Vflat.
- `work/private/matched-concentration-001/Rotmod_LTG.zip`: source archive used only for member hash and source-column crosschecks. No raw observations are copied by this audit.
- `work/gravity-first-principles/sparc-pattern-analysis-001/registration.json`: historical 139-name exposed cohort. Whole-galaxy folds can assess internal prediction but cannot restore independent confirmation.

Primary source: Lelli, McGaugh & Schombert (2016), AJ 152, 157, [SPARC catalogue J/AJ/152/157](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/AJ/152/157), dataset DOI `10.26093/cds/vizier.51520157`. Exact local file and archive-member hashes are in `inventory.json`.

## Executed source-only checks

Run `python work/gravity-first-principles/mond-atlas-clock-relay-001/source-audit/audit.py` from the repository root. All 139 historical development objects (2720 rows) have finite source values, positive strictly increasing radii, nonnegative surface brightness, and positive nominal baryonic force-equivalent speed squared. All 139 archive member hashes match the authenticated curve asset, source columns match exactly, and both photometry columns match exactly. No fit, residual, response summary or response numerical eligibility was calculated.

The nominal convention is `Vbar² = Vgas*abs(Vgas) + 0.5 Vdisk² + 0.7 Vbulge²`. Gas velocities may be signed to retain outward central force from a gas distribution with a hole. Gas includes 1.33 for helium; adding that factor again is incorrect. Stellar component velocities are calculated contributions at M/L = 1, not measured separate populations of stars orbiting at those speeds. Their squared contributions scale linearly with M/L.

For acceleration in SI, `gbar = (Vbar²/r) * 1e6/3.085677581491367e19`; the intermediate units are (km/s)²/kpc. Distance changes affect both inferred source amplitudes and radius; inclination changes affect inferred observed circular speeds. These are coupled nuisance effects rather than additional independent data.

## Limits that constrain the proposed formulas

The radial source permits Newton and algebraic MOND controls, plus explicitly phenomenological radial return laws. It cannot uniquely reconstruct distributed three-dimensional secondary generators: the same radial force can correspond to different vertical fields. A radial potential difference derived by integrating the baryonic force is possible, but an absolute well depth needs a declared outer-tail/boundary prescription. `Vbar²*r/G` is a spherical-equivalent mass, not measured enclosed disk mass. Do not label it as such.

Neither SPARC nor this inventory independently measures whether time supplies energy, whether gravity is absorbed, or response delays. A fitted clock-related correction is a statistical proxy until an explicit field/action, energy budget, boundary conditions and independent time-sensitive observations are supplied. HI is not total gas; molecular matter, stellar M/L, warps, streaming and inclination uncertainty remain relevant. The published diagonal velocity errors are not a full channel-noise or radial covariance model.

## Access accounting

A preliminary `Get-Content -TotalCount` inspection encountered minified JSON and printed/truncated existing velocity-response rows. This was reported to the parent before protocol finalization. No outcomes were scored, but the sample is not claimed to be newly blinded. The audit script itself computes source-only checks and preserves the other 36 names without expanding the fit cohort.
