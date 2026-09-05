# Master formula validation matrix

Status: validation contract assembled 2026-08-05. This document consolidates
the project's existing stage gates; it does not change the gravity formula,
constants, evidence split, or any sealed target.

## What a pass would mean

No finite experiment can prove a gravity theory true. The tests below can show
that a frozen formula is competitive over a declared domain, or can falsify it
within that domain. We therefore use three claim levels:

| Claim level | Minimum evidence required |
|---|---|
| **A. Useful low-redshift law** | Protocol integrity, numerical verification, diverse held-out galaxies, raw strong and weak lensing, and joint matter/light tests all pass with one frozen setting. |
| **B. Viable relativistic gravity theory** | Level A plus an action, conservation, stability, causality, a well-posed initial-value problem, Solar-System tests, binaries, and gravitational waves. |
| **C. Credible dark-matter alternative** | Levels A and B plus cosmological background, CMB, structure growth, cosmic shear, cluster abundance, small-scale structure, and time-dependent energy-transfer tests. |

The present empirical `RAR + squared coherence-gated RG` bridge is eligible to
attempt Level A. It cannot earn Levels B or C until its gate and both metric
potentials are derived from one healthy covariant action.

## Rules that apply to every test

1. Use one equation, one source definition, one matter metric, and the same
   universal constants everywhere.
2. Permit at most five universal physical constants, no per-object gravity
   parameters, no object labels, and no lensing-only multiplier.
3. Separate ordinary measurement nuisances from gravity parameters and give
   Sigma and every comparator the same nuisance information and priors.
4. Freeze the equation, constants, target list, masks, covariance, score, and
   pass threshold before opening validation or holdout outcomes.
5. Score raw observables. A halo, convergence, or acceleration map inferred by
   assuming conventional gravity is a comparator or development diagnostic,
   not validation truth.
6. Report aggregate and per-stratum results. A good average cannot conceal a
   failure in dwarfs, bulged galaxies, merging clusters, or another frozen
   class.
7. A single failure is retained and diagnosed while the rest of the matrix is
   completed. It does not automatically kill a candidate that is competitive
   elsewhere.
8. After three materially different, physically derived closures fail the same
   gate, reconsider that mechanism instead of adding a convenience parameter.

## 0. Registration and anti-tuning tests

| ID | Test | Done when |
|---|---|---|
| P01 | Canonical equation lock | The exact action/equation, source, metric coupling, constants, units, and code commit are recorded and hashed. |
| P02 | Parameter accounting | Universal physical, global nuisance, and per-system nuisance counts are reported separately; gravity counts are `<=5`, `0`, and `0` respectively for universal, per-object, and lens-only terms. |
| P03 | No object switch | Static analysis and run manifests show no galaxy/cluster, dwarf/giant, morphology, or relaxed/merger label enters the force law. |
| P04 | No outcome-derived source | Density, coherence, pressure, current, or environment inputs are constructed without scored velocities, lensing masses, halo maps, or residuals. |
| P05 | Frozen split | Development, validation, and untouched holdout systems are named and hashed before fitting; related observations of one object stay in one split. |
| P06 | Frozen analysis | Masks, cuts, covariance, nuisance priors, optimizer, root finder, score, and thresholds are frozen before unblinding. |
| P07 | Comparator parity | Baryons-only GR/Newton, fixed MOND/RAR, and conventional halo comparators use the same catalog, masks, likelihood, nuisance priors, and uncertainty model. |
| P08 | Leakage audit | No validation/holdout outcome, derived target, filename label, cached fit, or publication result can influence formula selection. |
| P09 | Negative controls | Scrambled source geometry, randomized orientations, shuffled object assignments, and null sky positions fail as expected. |
| P10 | Prediction ledger | Every prediction is timestamped before the corresponding target is opened, including expected sign, scale, topology, and declared uncertainty. |

## 1. Definition and mathematical consistency

| ID | Test | Done when |
|---|---|---|
| M01 | First-principles postulates | The field's physical meaning, source, propagation, energy, coupling, and limiting behavior are stated without referring to a desired galaxy or cluster result. |
| M02 | Covariant action | One diffeomorphism-invariant action produces the metric, Sigma, matter, and any vector/tensor field equations. An empirical bridge alone fails this test. |
| M03 | Independent variation | Symbolic and independent hand derivations of all Euler-Lagrange equations agree, including boundary terms. |
| M04 | Conservation identity | The Bianchi/Noether identity implies covariant conservation of total stress-energy and the matter exchange terms close exactly. |
| M05 | Degree-of-freedom count | Hamiltonian/constraint analysis identifies every propagating and constrained mode with no hidden extra scalar or spin-2 ghost. |
| M06 | Gauge consistency | Physical observables are gauge invariant and the numerical gauge leaves no spurious source or solution mode. |
| M07 | Dimensional consistency | Every term and constant has consistent units in SI, geometric, and code units; nondimensionalization is documented. |
| M08 | Positivity/monotonicity | Effective response is positive and single-valued; for a constitutive law, `mu>0` and `d(mu*g)/dg>0` over all observed and simulated support. |
| M09 | No singular physical branch | No pole, branch jump, negative effective coupling, or undefined state occurs over the registered density, acceleration, potential, curvature, velocity, and redshift range. |
| M10 | Healthy kinetic matrix | All physical kinetic eigenvalues are positive on Solar, galaxy, cluster, merger, and cosmological backgrounds. |
| M11 | Healthy gradients | All physical squared propagation speeds are non-negative and finite on those same backgrounds. |
| M12 | Hyperbolicity/well-posedness | The characteristic matrix supplies a well-posed initial-value problem; constraints propagate under numerical evolution. |
| M13 | Causality | Every physical characteristic respects the theory's declared causal cone, with no closed causal pathology in the tested backgrounds. |
| M14 | Energy boundedness | The Hamiltonian or equivalent conserved energy is bounded below for admissible perturbations. |
| M15 | Newton/GR limit | In the registered high-field limit, the fractional extra force is `<1e-5` when `g_bar/a_Sigma >= 1e5`, before detailed local fitting. |
| M16 | Deep-field asymptote | The analytic far-field solution is derived; if the candidate claims a MOND-like limit, it approaches `sqrt(a_Sigma*g_bar)` within 5% for `g_bar/a_Sigma <= 1e-3`. |
| M17 | Matter/light derivation | Massive motion follows `-grad(Psi)` and light follows the Weyl potential `(Phi+Psi)/2`, both derived from the same physical metric without a fitted slip. |
| M18 | Symmetry solutions | Spherical, planar, cylindrical, homogeneous, and isolated point-source limits agree with analytic or asymptotic solutions and expose any geometry dependence. |
| M19 | Superposition/nonlinearity audit | The equation specifies whether components are combined before or after nonlinear response, and the continuum limit is independent of arbitrary source segmentation. |
| M20 | Boundary and total-flux law | Isolated, periodic, and cosmological boundary conditions are physically stated; any field flux, redirected gravity, or memory has a conserved accounting identity. |

## 2. Numerical solver verification

These tests are rerun for every materially new equation, not inherited from an
older Sigma candidate.

| ID | Test | Done when |
|---|---|---|
| N01 | Manufactured scalar solution | The solver recovers a smooth imposed solution and source within the preregistered norm tolerance. |
| N02 | Manufactured vector/tensor solution | Directional and off-diagonal components are recovered, including a nonzero curl/shear case. |
| N03 | Normalized equation residual | Residual is `<1e-6` in every scored production solve, with tail quantiles reported rather than only a mean. |
| N04 | Spatial convergence | Doubling spatial resolution changes every scored observable by `<2%`; the observed order of convergence matches the discretization. |
| N05 | Temporal convergence | Halving the time step changes dynamic observables by `<2%` and preserves constraints. |
| N06 | Domain-size convergence | Enlarging the domain leaves scored inner observables stable within 2%. |
| N07 | Boundary-condition sensitivity | Every physically admissible boundary choice is reported; the selected result is not created by one convenient edge condition. |
| N08 | Conservation in evolution | Total energy, momentum, angular momentum, constraints, and any Sigma flux close to their predicted numerical truncation errors. |
| N09 | Symmetry and rotation covariance | Rotating/translating the same baryonic system rotates/translates predictions without changing invariant scores. |
| N10 | Source discretization | Rebinning, splitting, or merging an identical continuous baryon distribution does not create artificial nonlinear enhancement. |
| N11 | Extreme-state stress test | Very compact, diffuse, empty, multi-centre, rapidly moving, and nearly symmetric sources remain finite and converge. |
| N12 | Determinism and platform check | Repeated CPU/GPU runs agree within frozen floating-point tolerance and record seeds, versions, and precision. |
| N13 | Injection/recovery | Synthetic observations generated from known constants recover them without bias and report calibrated coverage. |
| N14 | Null recovery | Newton/GR and fixed-MOND synthetic controls do not spuriously prefer a complex Sigma law. |

## 3. Galaxy dynamics

The primary gate is at least 48 untouched galaxies, at least 32 from WALLABY,
and at least six systems in every frozen mass, gas, surface-brightness, and
bulge stratum. The overall held-out rotation RMSE must be no worse than
`1.05 x` fixed MOND/RAR and every stratum no worse than `1.25 x`.

| ID | Test | Raw observable and required output |
|---|---|---|
| G01 | Full rotation curves | Predict every retained velocity point, not just outer points or a fitted flat speed; report velocity RMSE, likelihood, residual trend, and coverage. |
| G02 | Dwarf-to-giant transfer | Report frozen low/high baryonic-mass strata separately with the identical constants. |
| G03 | Gas-rich/gas-poor transfer | Separate gas-dominated dwarfs from star-dominated systems to expose dependence on stellar mass-to-light assumptions. |
| G04 | LSB/HSB transfer | Separate low/high surface-brightness systems to test whether the transition follows acceleration, volume density, surface density, or an illicit class proxy. |
| G05 | Bulgeless/bulged transfer | Separate thin disks, large bulges, and pressure-supported systems; include the full 3-D baryonic geometry. |
| G06 | Curve-shape transfer | Score rising, flat, declining, inner-peaked, warped, and asymmetric rotation curves separately. |
| G07 | Observation-space cubes | Forward-model at least 12 H I/CO cubes or resolved velocity fields, including beam smearing, inclination, dispersion, and noncircular flow. |
| G08 | High-resolution inner curves | At least eight systems test the transition and central geometry with high-resolution H-alpha/CO or equivalent data. |
| G09 | Vertical plus radial dynamics | At least eight systems jointly predict vertical force/dispersion and radial rotation using one 3-D field. |
| G10 | Baryonic Tully-Fisher relation | Predict slope, normalization, intrinsic scatter, and residual correlations from the equation without fitting a separate BTFR. |
| G11 | RAR residual structure | Match the held-out RAR while showing no residual trend with radius, mass, size, gas fraction, surface density, bulge fraction, environment, or data quality. |
| G12 | Resolved asymmetry | Predict receding/approaching side differences and 2-D residual directions when a vector, tide, current, or external field is claimed. |
| G13 | Ellipticals and BCGs | Predict stellar-dispersion profiles and anisotropy-marginalized accelerations for pressure-supported galaxies; the existing BCG continue gate is `chi2/N <=5` and `|mean residual| <=0.15 dex`. |
| G14 | Dwarf spheroidals | Predict dispersion profiles, escape speeds, tidal radii, and host-distance dependence without a halo or force parameter per satellite. |
| G15 | External-environment test | Use an independently measured environment and predict the sign and magnitude before kinematics are opened; isolation and dense-environment controls must differ only through physical inputs. |
| G16 | Distance/inclination robustness | Propagate independent distance, inclination, gas, and stellar-population posteriors; the gravity law may not absorb them as per-object force parameters. |
| G17 | Redshift transfer | With constants fixed locally, predict suitable disk and dispersion data at multiple redshifts; any physical/comoving scale convention is fixed in advance. |

## 4. Lensing and joint matter-light tests

At least six untouched clusters are required, including two relaxed and two
disturbed/merging systems. Each must have gas, BCG, intracluster light, member
galaxies, at least three secure image families, one spectroscopic family, and
eight images. A radial acceleration reconstructed from an NFW fit is useful for
development but cannot pass this section.

| ID | Test | Raw observable and required output |
|---|---|---|
| L01 | Strong-lens root completeness | Recover 100% of held-out image roots for every cluster; a missing or invented image is an explicit topology failure. |
| L02 | Image positions | Image-plane RMS is `<=1.25 x` the same-catalog halo comparator and closes at least 75% of the baryons-only-to-halo deviance gap. |
| L03 | Multiplicity, parity, and ordering | Predict the observed number, parity, radial/tangential type, and arrival ordering for every registered image family. |
| L04 | Critical curves and caustics | Predict their location, connectivity, folds, cusps, and topology, not merely total deflection amplitude. |
| L05 | Multi-redshift scaling | One metric correctly scales families across source redshift without family-specific gravity amplitudes. |
| L06 | Strong-lens time delays | Predict measured delays with the same lens potential; source position and cosmographic nuisances remain separate. |
| L07 | Extended arcs | Forward-model arc pixels/surface brightness with PSF and source reconstruction, using frozen regularization and evidence accounting. |
| L08 | Cluster weak shear | Predict per-source `g1`, `g2`, or ellipticity and full covariance; aggregate deviance `<=1.25 x` halo, every stratum `<=1.50 x`, and at least 75% gap closure. |
| L09 | Magnification | Predict number-count or size magnification jointly with shear and strong lensing. |
| L10 | Shear nulls | Cross/B modes and random-point tests have frozen `p>=0.01`; a coordinate or PSF artifact cannot count as Sigma curvature. |
| L11 | Galaxy-galaxy weak lensing | Pass the same 1.25 aggregate, 1.50 per-bin, and 75%-closure rules in at least three baryonic-mass and two morphology/surface-density bins. |
| L12 | Galaxy-scale strong lenses | Predict Einstein radii, extended arcs, and time delays across disk and elliptical lenses with the galaxy constants. |
| L13 | Joint dynamics and lensing | At least eight rotation- and pressure-supported systems use the identical metric/constants; combined deviance `<=1.25 x` halo and neither domain is worse than baryons-only. |
| L14 | Merger direction and offsets | In two development plus at least two untouched mergers, freeze the baryon-derived response axis before lensing; median axial error `<=30 deg`, with the full topology and 75%-closure gates. |
| L15 | Photon universality | Lensing is achromatic and consistent across radio/optical/X-ray/GW messengers within observational errors unless the action explicitly predicts and survives a frequency dependence. |
| L16 | Polarization and birefringence | Predict polarization transport and pass birefringence/rotation constraints; no lensing repair may silently alter polarization. |
| L17 | Independent lens-team robustness | Repeat on alternate image catalogs, source associations, mass/light reductions, and at least one independently implemented lens solver. |

## 5. Cluster dynamics and baryonic source-state tests

These tests determine whether Sigma is responding to the actual baryonic state
rather than merely imitating a static halo. They complement lensing but cannot
replace the raw lensing gate.

| ID | Test | Raw observable and required output |
|---|---|---|
| C01 | Hydrostatic profiles | Predict X-ray/SZ gas density, temperature, and pressure support with nonthermal uncertainty propagated. |
| C02 | Galaxy phase space | Predict member line-of-sight velocity distributions, dispersion profile, caustic/escape envelope, and substructure membership. |
| C03 | Direct ICM bulk velocity | Reproduce response-aware XRISM line-centroid velocities in A2319 development, freeze, then predict A3667 validation and A754 holdout without retuning. |
| C04 | ICM turbulence | Predict or consistently incorporate line broadening/velocity dispersion and its spatial relation to the Sigma source. |
| C05 | Multi-component consistency | Gas, stars, ICL, galaxies, pressure, current, and their uncertainties enter once; removing a component changes predictions by the equation's declared response. |
| C06 | Relaxed/disturbed transfer | The identical constants pass relaxed, cool-core, disturbed, and merging systems separately. |
| C07 | Shock and cold-front geometry | Predict the sign, lag, and orientation of any pressure/current/memory response relative to shock, cold-front, gas, and galaxy observations. |
| C08 | Collision/offset chronology | A time-dependent simulation predicts pre-, during-, and post-pericenter curvature without fitting an observed lensing offset. |
| C09 | Joint cluster likelihood | Strong lensing, weak lensing, SZ/X-ray, galaxy dynamics, and ICM velocities are scored jointly with shared baryons and gravity constants. |
| C10 | Baryon-reconstruction robustness | Repeat with alternate gas deprojection, BCG/ICL mass-to-light, member selection, triaxiality, line-of-sight structure, and calibration posteriors. |

## 6. Solar-System, laboratory, and equivalence-principle vetoes

These are mandatory for Level B, but detailed tuning is intentionally later
than the galaxy/cluster breadth gate.

| ID | Test | Done when |
|---|---|---|
| S01 | Laboratory inverse square | The predicted scale/density/environment dependence is within torsion-balance, Cavendish, and short-range inverse-square constraints. |
| S02 | Universality of free fall | Composition-dependent and self-energy-dependent accelerations satisfy laboratory, MICROSCOPE, and lunar constraints. |
| S03 | Local `G` consistency | Spatial, temporal, and environmental variation of the measured gravitational coupling is within registered bounds. |
| S04 | Gravitational redshift | Atomic-clock, spacecraft, and terrestrial redshift predictions use the same metric potentials. |
| S05 | Cassini/Shapiro delay | The Solar limit has `|gamma-1| <= 2.3e-5`. |
| S06 | Solar light deflection | Deflection and its impact-parameter dependence agree with radio/optical astrometry. |
| S07 | Mercury perihelion | The residual anomalous precession lies within the declared ephemeris uncertainty after standard perturbations. |
| S08 | Full planetary ephemerides | Simultaneously fit/rule against all planets, spacecraft ranging, and asteroid perturbations; Mercury alone is insufficient. |
| S09 | Lunar laser ranging | Nordtvedt effect, geodetic precession, inverse-square behavior, and time variation pass the same action. |
| S10 | Preferred-frame/location PPN | All relevant PPN parameters and preferred-frame/location effects satisfy bounds in the theory's physical frame. |
| S11 | Solar-system stability | Long integrations show no secular instability, anomalous energy loss, or unacceptable orbital polarization. |
| S12 | Screening continuity | The transition from Solar to galactic scales is continuous and predicted; no hand-set system boundary or Solar-only branch is allowed. |

## 7. Relativistic, binary, and strong-field tests

| ID | Test | Done when |
|---|---|---|
| R01 | Tensor-wave speed | `|c_T/c-1| <= 1e-15` on the relevant low-redshift background. |
| R02 | Extra polarizations | Scalar/vector radiation and polarization amplitudes satisfy detector and pulsar-timing limits. |
| R03 | GW damping and distance | Gravitational-wave luminosity distance, friction, dispersion, and decay agree with standard sirens and multimessenger timing. |
| R04 | Binary-pulsar timing | Periastron advance, Shapiro delay, orbital decay, dipole radiation, and strong-field equivalence effects pass jointly. |
| R05 | Compact-object solutions | Stable neutron-star and black-hole solutions exist with regular exterior matching and acceptable mass-radius relations. |
| R06 | Black-hole observations | Shadows, stellar/pulsar orbits, accretion dynamics, and ringdown are consistent where the theory predicts deviations. |
| R07 | Gravitational radiation energy | Wave energy is positive and the binary energy/angular-momentum balance closes. |
| R08 | Strong-field stability | No ghost, tachyonic, gradient, or spontaneous-instability branch is triggered in observed compact objects unless it is quantitatively allowed. |

## 8. Other phenomena attributed to dark matter

These tests are required before claiming that the formula replaces dark matter
as a phenomenon rather than only fitting galaxies and clusters.

| ID | Test | Required prediction |
|---|---|---|
| D01 | Globular/open clusters | Dispersion and escape profiles across external environments without a cluster-specific force scale. |
| D02 | Wide binaries | Relative-velocity distribution with selection, multiplicity, and Galactic-tide systematics forward-modeled. |
| D03 | Tidal streams | Track, precession, width, density, and velocity perturbations from one time-dependent Galactic field. |
| D04 | Satellite survival | Tidal radii, disruption rates, orbital poles, and host-distance dependence for gas-poor satellites. |
| D05 | Dynamical friction | Sign and rate of orbital energy/angular-momentum transfer, including the physical destination of lost energy. |
| D06 | Bars and disk stability | Bar pattern speeds, spiral response, disk heating, and stability without a stabilizing invisible halo. |
| D07 | Compact substructure signals | Strong-lens flux/astrometric anomalies and stream gaps only if the field equations independently generate stable compact structures with a predicted abundance. |
| D08 | Local escape speed | Galactic escape curve and high-velocity stellar distribution from the same Milky Way baryon model. |
| D09 | Halo-shape observables | Polar rings, flaring, streams, satellite planes, and lensing shear receive one baryon-derived 3-D potential, not a fitted halo shape. |

## 9. Cosmology and formation history

| ID | Test | Done when |
|---|---|---|
| K01 | FLRW background | The action admits a stable homogeneous/isotropic solution and predicts `H(z)` with fixed constants. |
| K02 | Big-bang nucleosynthesis | Expansion and any varying coupling/extra radiation preserve light-element abundances. |
| K03 | Recombination and primary CMB | A Boltzmann implementation predicts acoustic peak positions, relative heights, damping tail, and polarization without cold dark matter if that is the claim. |
| K04 | CMB lensing and ISW | The same evolving Weyl potential predicts CMB lensing and early/late ISW correlations. |
| K05 | BAO | Sound horizon interpretation and transverse/radial BAO distances agree across redshift. |
| K06 | Supernova distances | Luminosity distances and calibration nuisances fit jointly with BAO/CMB rather than with a separate expansion rule. |
| K07 | Linear growth | Predict `f*sigma8`, redshift-space distortions, and scale-dependent growth from the same perturbation equations. |
| K08 | Matter power spectrum | Match shape, amplitude, turnover, and nonlinear transition over registered scales. |
| K09 | Cosmic shear | Predict tomographic shear, cross-correlations, and intrinsic-alignment-marginalized covariances. |
| K10 | Galaxy clustering | Predict bias-aware two-point and higher-order clustering without retuning gravity by tracer or redshift. |
| K11 | Cluster abundance | Predict the mass function and SZ/X-ray/lensing selection across mass and redshift. |
| K12 | Lyman-alpha forest | Small-scale power and thermal-history-marginalized forest statistics remain acceptable. |
| K13 | Voids and peculiar velocities | Predict void density/lensing/velocity profiles, bulk flows, and pairwise velocities; redshift is an observable, not removed from the test. |
| K14 | Formation simulations | Cosmological simulations form realistic galaxies, clusters, merger rates, concentrations, and baryon distributions without inserting target halo profiles. |
| K15 | Cross-epoch constants | The same physical constants and scale interpretation work from recombination to today; redshift-specific gravity retuning is forbidden. |

## 10. Statistical robustness, replication, and scientific audit

| ID | Test | Done when |
|---|---|---|
| A01 | Full uncertainty propagation | Measurement, calibration, distance, inclination, stellar population, gas, redshift, PSF, and selection uncertainties propagate to the final score. |
| A02 | Correlated likelihoods | Spatial, spectral, and catalog covariance is used; correlated points are not treated as independent to inflate significance. |
| A03 | Posterior predictive checks | Simulated observables reproduce residual distributions, outliers, morphology strata, and calibration diagnostics. |
| A04 | Coverage and calibration | Confidence/credible intervals have correct injection coverage and the optimizer does not sit on a hard bound. |
| A05 | Selection-function test | Survey and target-selection effects are forward-modeled or bounded with a frozen sensitivity analysis. |
| A06 | Baryon-model ensemble | Results survive independent stellar-population, gas, geometry, distance, and member catalogs without selecting the favorable reconstruction. |
| A07 | Outlier and missingness policy | Frozen robust and non-robust scores, every exclusion, and missing-data mechanism are reported. |
| A08 | Look-elsewhere accounting | The number of formula families, closures, parameter sweeps, and reused development targets is recorded when quoting significance. |
| A09 | Parameter economy | Report likelihood and predictive scores beside universal and per-object parameter counts; compare to halo models fairly rather than declaring `1 vs many` without nuisance accounting. |
| A10 | Prior-art audit | Each mathematical ingredient and limiting formula is compared with published MOND/AQUAL, refracted gravity, scalar-vector-tensor, aether, MOG/STVG, nonlocal, emergent, and lensing frameworks. |
| A11 | Independent implementation | A second solver and ideally an external group reproduce at least one galaxy, one relaxed cluster, and one merger from public inputs. |
| A12 | Reproducible release | Code, frozen manifests, hashes, environment, data provenance, intermediate products, plots, and failure reports are public or independently escrowed. |
| A13 | Adversarial falsification | Researchers can submit geometries and formulas to the simulator; pathological and counterexample configurations are retained, not curated away. |
| A14 | Failure ledger | Every failed stratum remains visible with diagnosis, attempted physically distinct closures, and the exact gate that failed. |
| A15 | Prospective replication | After all development changes stop, at least one new galaxy release and one new cluster catalog reproduce the frozen result prospectively. |

## Decision sequence

The order prevents months of polishing a formula that cannot meet the central
claim, while honoring the rule that one difficult system does not erase broad
success.

1. **Freeze and verify:** P01-P10, M07-M09, M15-M20, and N01-N14.
2. **Core empirical breadth:** G01-G16 and L01-L17. Continue the whole matrix
   when a candidate is close overall but misses one predeclared stratum; park
   the stratum and diagnose it after completing the other independent gates.
3. **Direct source-state evidence:** C01-C10, including the active XRISM chain.
4. **Level-A decision:** advance if the galaxy and raw-lensing gates pass
   simultaneously with one setting. A result that passes only one domain is a
   useful falsification/diagnostic, not a unified law.
5. **Action and relativistic vetoes:** complete M01-M14, S01-S12, and R01-R08.
6. **Dark-matter-alternative claim:** complete D01-D09 and K01-K15.
7. **Final replication:** A01-A15, especially prospective frozen releases.

## Current project position against this matrix

| Area | Current evidence | Status |
|---|---|---|
| Formula/prior-art inventory | 128 scientifically distinct scored rows and 36 relevant published families are registered. | Strong infrastructure; rerun novelty audit for any final action. |
| Galaxy rotation | The empirical bridge scored 10.586 km/s on 131 held-out SPARC outer curves, 1.94% worse than simple MOND; a later two-potential dwarf test beat its frozen MOND comparator. | Promising but does not yet pass the full 48-galaxy morphology/cube/vertical breadth contract. |
| Derived cluster radial fields | The empirical bridge reached 0.1387 dex on 20 CLASH NFW-derived profiles. | Useful development evidence only; NFW-derived targets are not raw validation truth. |
| Raw strong lensing | One RX J2129 result was encouraging, but no formula has passed raw multi-cluster roots, topology, and positions. | Central unresolved gate. |
| Cluster source dynamics | The response-aware A2319 route failed because the public regional NXB model was unacceptable. V19DE2's independent Bullet 2T closure had an unrejected second redshift minimum, and V19DF's reconstructed MACS J0018 member gradient failed shuffle, catalog-transfer and bootstrap gates. In the thermodynamic lane, V19DI/V19DK2 validate direct OGIP writing and deterministic FITS products. V19DL combines all 5,082 cells exactly and both registered regional fits pass, but the Bullet integrated 1T spectrum fails (`reduced chi2=2.7937`). The parity-corrected V19DM2 minimal 2T repair also fails (`2.8023`, `Delta BIC=+10.4164`). V19DN localizes the mismatch below about 2 keV, while V19DO2 shows that scaled blank sky supplies only 2.037% of Bullet soft counts and is not strongly heterogeneous by observation/CCD. | Time-odd/component current, I4 thermodynamic-gradient stress, and I5 baroclinicity remain unadmitted. The response machinery is credible, and blank-sky amplitude is not the soft-failure explanation, but no full 494-region successor is authorized. Next validate an unmerged-response joint likelihood on the two registered regions; keep lensing and action payloads sealed. |
| Numerical framework | Manufactured and field-solver tests exist for earlier candidates. | Must be rerun for the final frozen equation. |
| Solar/relativistic | Several phenomenological screens pass limited Mercury/Cassini diagnostics. | No empirical bridge has earned action-level Level-B validation. |
| Cosmology/structure | Not calculable from the empirical bridge. | Requires a final covariant action and perturbation implementation. |

## Existing machine-readable gates

The exact quantitative core already exists in:

- `configs/theory_stage_gates.json`
- `configs/sigma_v19cc_cross_scale_prediction_gates.json`
- `docs/THEORY_DEVELOPMENT_STAGE_GATES.md`
- `docs/SIGMA_V19CC_CROSS_SCALE_PREDICTION_GATES.md`
- `docs/FORMULA_AND_PRIOR_ART_REGISTRY.md`
- `docs/SIGMA_V19CY_DIRECT_ICM_VELOCITY_EVIDENCE_PLAN.md`

This master matrix is the checklist; those frozen protocols remain the
authoritative source for exact sample hashes and experiment-specific details.
