# Cluster phenomenology publication-readiness goals

This is the binding work program for turning the Item 59 X-COP result into a
data-ready and potentially publishable **bounded cluster-phenomenology paper**. It is
separate from the much higher goal of establishing a universal alternative to general
relativity or eliminating dark matter.

The primary empirical result is retained even though its unchanged formula fails on
galaxies. A failure outside the declared cluster domain blocks universal promotion; it
does not erase a reproducible in-domain result. This policy is enforced by
`configs/research_publication_readiness_policy_v1.json`.

## Claim ladder

Invariant must adjudicate three claims independently:

| Track | Claim | Current status |
|---|---|---|
| Bounded empirical publication | One frozen baryon-conditioned radial law predicts pressure and temperature inside the declared massive-cluster domain. | **Development evidence retained; not data-ready for independent replication** |
| Physical mechanism | The relation is evidence for a particular cause rather than dark matter, nonthermal pressure, baryon error, calibration, geometry, or an empirical surrogate. | **Blocked** |
| Universal gravity theory | One action and parameter set covers galaxies, groups, clusters, local gravity, strong fields, and lensing. | **Blocked** |

The bounded paper must not claim that general relativity is false, that dark matter has
been eliminated, that the exact construction is historically novel, or that light
follows the candidate acceleration law.

## Current evidence

- Item 59 froze 2,025 law-and-nuisance variants and selected one two-kernel law.
- It predicted response-blind radial holdouts in eight X-COP development clusters.
- It transferred without formula or nuisance refitting to four initially sealed X-COP
  clusters from the same survey and reduction family.
- It predicted Planck SZ pressure and XMM temperature directly, using one unscored outer
  pressure boundary per cluster, instead of fitting a released total-mass profile.
- It improved the frozen aggregate confirmation score by about 90% over the strongest
  frozen comparator and retained a positive result under six sensitivity variants.
- Typical absolute confirmation error remained about 19%.
- The selected member-baryon and nonthermal-pressure settings were at the high end of
  their frozen grids.
- The unchanged law failed broadly on 139 SPARC and 11 LITTLE THINGS galaxies and has an
  invalid unscreened high-acceleration limit. Those facts block universal promotion but
  do not reject the cluster-domain observation.
- The first `4^10` screened descendant grammar found no SPARC-X-COP bridge. That is a
  failure-space result, not a reason to hide Item 59.
- A machine-readable manuscript evidence package now preserves all 233 Item 59 candidate
  predictions and residuals across development training, development holdout, and
  same-release confirmation, together with object summaries, counterexamples,
  comparators, ablations, numerical controls, uncertainty warnings, source revisions,
  and access counts. It explicitly records zero independent target rows and is not a
  completed manuscript or independent replication.
- One deterministic standard-library command now recreates the frozen inventory of seven
  primary CSV tables and six standalone SVG figures from that package. The artifact
  manifest binds every output hash and retains the same not-paper-ready claim boundary.

Useful primary context includes the SPARC radial-acceleration relation
(https://arxiv.org/abs/1609.05917), the documented residual MOND problem in X-ray groups
and clusters (https://arxiv.org/abs/0709.0108), X-COP hydrostatic-profile comparisons
(https://arxiv.org/abs/1805.00035), and the finding that the X-COP cluster acceleration
relation departs strongly from the spiral-galaxy relation
(https://arxiv.org/abs/2205.01110).

## Ordered gate table

No independent response rows may open until every gate marked **pre-data** is PASS.
Lensing and universal-theory gates are not prerequisites for the bounded empirical
paper, but they become mandatory if the claim is escalated.

| Gate | Purpose | Required for pre-data | Required for bounded paper | Current |
|---|---|---:|---:|---|
| CP0 | Freeze claim and domain | Yes | Yes | PASS |
| CP1 | Freeze candidate and endpoints | Yes | Yes | PASS |
| CP2 | Prior art and expert positioning | No | Yes | PARTIAL |
| CP3 | Direct-observable data contract | Yes | Yes | PARTIAL; X-COP is bound but the independent packet is absent |
| CP4 | Matched-flexibility comparators | Yes | Yes | PASS on development data; frozen for replication |
| CP5 | Covariance, nuisances, and alternative causes | Yes | Yes | PARTIAL; stress tests complete, correlation-aware sampler improved but did not converge, source covariance and reparameterized marginalization open |
| CP6 | Numerical, synthetic, and leakage controls | Yes | Yes | PASS on development controls; outer-radius and power warnings retained |
| CP7 | Independent source and split freeze | Yes | Yes | PARTIAL; protocol frozen, no lane selected or payload committed |
| CP8 | Unchanged independent thermodynamic replication | No | Yes | NOT STARTED |
| CP9 | Independent mass probes | No | No for bounded paper | NOT STARTED |
| CP10 | Group and domain-boundary map | No | No for bounded paper | NOT STARTED |
| CP11 | Gravity-theory escalation | No | No for bounded paper | BLOCKED |
| CP12 | External reproduction and manuscript package | No | Yes | PARTIAL; development package and primary renderer complete; external replay, source release, reviews, and submission open |

## CP0 — freeze the bounded claim and population

Advance criteria: the paper can be judged true or false without silently requiring a
universal gravity theory.

- [x] **CP0.1** Make the primary track `bounded_empirical_publication`.
- [x] **CP0.2** Define the working claim: a frozen baryon-conditioned nonlocal radial
  relation predicts thermodynamic profiles in the declared massive-cluster domain.
- [x] **CP0.3** State prohibited conclusions: no established alternative to GR, no
  elimination of dark matter, no historical-novelty claim, and no lensing claim.
- [x] **CP0.4** Treat galaxy failure as a domain-boundary result that blocks only broader
  promotion.
- [x] **CP0.5** Define the working title: “A baryon-conditioned nonlocal acceleration
  relation for galaxy-cluster thermodynamic profiles.”

## CP1 — freeze the candidate, forward model, and endpoints

Advance criteria: no formula or endpoint changes are possible after independent target
access.

- [x] **CP1.1** Use the exact Item 59 candidate as the primary law.
- [x] **CP1.2** Freeze `a0`, `y0=0.1`, `beta=1.5`, kernel identities, log-radius scales,
  normalization, units, and edge behavior.
- [x] **CP1.3** Freeze the hydrostatic pressure integration and temperature conversion.
- [x] **CP1.4** Freeze the primary equal-cluster, equal-observable predictive score.
- [x] **CP1.5** Prohibit per-cluster gravitational coefficients.
- [x] **CP1.6** Preserve the X-COP development and same-release confirmation receipts.
- [x] **CP1.7** Produce a compact, journal-ready mathematical specification with every
  operator, boundary condition, unit, and declared degree of freedom.
- [x] **CP1.8** State the formula's declared radial and source domain and behavior outside
  it.

## CP2 — establish prior art and scientific positioning

Advance criteria: the manuscript knows which ingredients are established, which are a
combination, and which exact claim still requires human novelty adjudication.

- [x] **CP2.1** Label the present origin as a potentially new combination of known
  permittivity, nonlocal-kernel, MOND/RAR, and auxiliary-field motifs.
- [x] **CP2.2** Search ADS, arXiv, Crossref, OpenAlex, INSPIRE, and journal full text for
  the exact two-kernel occupancy construction and close behavioral equivalents.
- [x] **CP2.3** Build an equation-by-equation comparison with MOND/AQUAL, refracted
  gravity, nonlocal gravity, STVG/MOG, emergent gravity, NFW/Einasto, and empirical
  pressure models.
- [x] **CP2.4** Record rewrite, known-family, known-combination, structurally unmatched,
  and unresolved labels separately.
- [ ] **CP2.5** Obtain named review by at least one cluster astrophysicist and one
  modified-gravity specialist.
- [ ] **CP2.6** Keep corpus absence non-authoritative; only named human review may support
  a historical-novelty sentence.

The machine-audited comparison is in
`configs/gravity_cluster_prior_art_positioning_v1.json` and
`runs/gravity/publication-readiness/prior-art-positioning-v1.json`. It found no exact
algebraic rewrite of the two-kernel occupancy law, but that absence is not authoritative.
The closest behavioral neighbor found is Penner's February 2026 modified GRAS/AQUAL law
(https://arxiv.org/abs/2602.09249): both make an interior cluster field depend on the
complete radial baryon distribution and an outer boundary. Penner uses the response slope
`beta=-g/(r g')` in an inward-integrated field equation; Item 59 instead uses fixed inward
and symmetric averages of `q[g_bar/a0]`. The whole Item 59 construction is therefore
labeled a potentially new exact combination of known motifs with a close prior neighbor,
not a historically novel gravity law. Named expert reviews remain required.

## CP3 — direct-observable and provenance contract

Advance criteria: every predictor and target has a hash-bound provenance chain, and no
model-derived invisible component is treated as truth.

- [x] **CP3.1** Use measured gas-density and stellar-light profiles as baryonic
  predictors with declared transformations.
- [x] **CP3.2** Predict measured SZ pressure and X-ray temperature rather than released
  hydrostatic, NFW, total-mass, or lensing profiles.
- [x] **CP3.3** Preserve the unscored outer-pressure boundary role.
- [x] **CP3.4** Record zero direct-lensing and inferred-total-mass target use in Item 59.
- [ ] **CP3.5** Create a source manifest for every independent raw or calibrated file,
  release revision, DOI, license, checksum, object identifier, instrument, and reduction.
- [ ] **CP3.6** Bind calibration, background, beam/PSF, mask/selection, bandpass/spectral
  response, and covariance roles for every source packet.
- [x] **CP3.7** Freeze unit conversions, cosmology-dependent distance transformations,
  and allowed redshift uses.
- [x] **CP3.8** Add fail-closed detection for target-derived predictors, halo labels,
  derived masses, and post-response exclusions.

## CP4 — matched-flexibility comparator suite

Advance criteria: the candidate wins, loses, or ties against the strongest conventional
and generic models under comparable fitting freedom.

- [x] **CP4.1** Implement GR plus NFW forward pressure and temperature prediction.
- [x] **CP4.2** Implement GR plus Einasto forward prediction.
- [x] **CP4.3** Implement a flexible hydrostatic mass reconstruction.
- [x] **CP4.4** Implement GNFW and/or polytropic empirical pressure models.
- [x] **CP4.5** Retain Newtonian baryons and empirical RAR/MOND controls.
- [x] **CP4.6** Add a generic spline or Gaussian-process ceiling with carefully separated
  training and predictive scoring.
- [x] **CP4.7** Add same-parameter-count nonphysical and wrong-law controls.
- [x] **CP4.8** Add ablations removing each kernel channel and transition term.
- [x] **CP4.9** Calculate effective degrees of freedom, including selected nuisance-grid
  freedom and boundary information.
- [x] **CP4.10** Freeze likelihood, cross-validation, AIC/BIC, and—where priors are
  defensible—Bayesian model-comparison reporting before independent responses open.

The development-only comparator receipt is
`runs/gravity/publication-readiness/comparator-suite-v1.json`. The frozen candidate's
holdout score is `9.323`, versus `28.438` for the best GR+NFW grid and `30.913` for
GR+Einasto. It also beats the tested GNFW, flexible-HSE, and regularized-RBF ceilings.
This passes the pre-data implementation/freeze gate; it is not independent replication,
does not use full covariance, and does not establish a gravity mechanism.

## CP5 — full uncertainty, nuisances, and alternative causes

Advance criteria: the result is not an artifact of a grid edge, omitted covariance, or
one plausible cluster-astrophysics correction.

- [ ] **CP5.1** Acquire or reconstruct radial pressure covariance.
- [ ] **CP5.2** Acquire or reconstruct temperature covariance.
- [ ] **CP5.3** Propagate gas-density covariance into baryonic mass and acceleration.
- [ ] **CP5.4** Model pressure-temperature and shared-calibration correlations.
- [ ] **CP5.5** Propagate background-subtraction and beam/PSF uncertainty.
- [ ] **CP5.6** Marginalize XMM/Chandra and X-ray/SZ cross-calibration.
- [ ] **CP5.7** Replace the coarse nonthermal-pressure grid with a frozen continuous or
  independently constrained prior.
- [ ] **CP5.8** Marginalize BCG, satellite, missing member, intracluster-light, IMF, and
  stellar mass-to-light uncertainty.
- [ ] **CP5.9** Marginalize outer pressure-boundary uncertainty.
- [ ] **CP5.10** Model gas clumping, centering, projection, triaxiality, and spherical
  approximation error.
- [ ] **CP5.11** Freeze relaxed/disturbed and cool-core/non-cool-core strata.
- [x] **CP5.12** Test covariance inflation and plausible missing-not-at-random behavior.
- [ ] **CP5.13** Compare explicitly against extra nonthermal pressure, extra member
  baryons, calibration shifts, clumping, geometry, mergers, boundary error, and flexible
  ordinary halo explanations.
- [x] **CP5.14** Report which physical causes remain observationally indistinguishable.

The development uncertainty receipt is
`runs/gravity/publication-readiness/uncertainty-program-v1.json`. It tests 36 radial
correlation/error-inflation/shared-calibration covariance scenarios and 12 adversarial
missingness scenarios without deleting real rows. The candidate beats the frozen NFW
comparator in all 36 covariance stress scenarios. A 17-parameter continuous nuisance
program was also attempted with 21,640 forward evaluations, but its multi-chain sampler
did not converge (`max R-hat=12.52`, minimum estimated effective samples `14.09`). CP5.7
through CP5.10 therefore remain open: the failed sampler is recorded as evidence of
strong nuisance degeneracy, not mislabeled as successful marginalization. Full released
or reconstructed source covariance and morphology/merger strata also remain missing.

A follow-up diagnostic then spent 501,636 candidate forward evaluations without opening
confirmation or independent rows. Prior-space Sobol importance sampling collapsed to one
effective sample. Extending the componentwise chains to 500 retained samples per chain
still failed (`max R-hat=17.38`, minimum effective samples `28.02`). Four independently
initialized logit-space affine ensembles materially improved mixing; the largest run
retained 115,200 posterior draws with minimum estimated effective samples above 1,423.
It still failed the unchanged all-parameter criteria (`max R-hat=1.64`, maximum
standardized between-ensemble median spread `0.458`). The bound receipt is
`runs/gravity/publication-readiness/nuisance-sampler-diagnostic-v1.json`. The next valid
move is a frozen reparameterization of the density-calibration-geometry degeneracy and
the six-factor stellar product, or independently calibrated priors—not more
componentwise brute force and not weaker thresholds.

## CP6 — numerical, synthetic, and leakage controls

Advance criteria: the evaluator recovers known injected laws, rejects wrong laws, and is
stable under independent implementations and resolution changes.

- [x] **CP6.1** Preserve deterministic formula enumeration, split, and result receipts.
- [x] **CP6.2** Preserve unit, positive-observable, boundary-alignment, and analytic-limit
  tests.
- [x] **CP6.3** Demonstrate radial-grid and integration convergence.
- [x] **CP6.4** Inject Newton/NFW, MOND/RAR, Item 59, and deliberately wrong synthetic
  universes and verify recovery/rejection.
- [x] **CP6.5** Measure false-selection rates across the full 2,025-variant search.
- [x] **CP6.6** Verify CPU, GPU, and a separately written evaluator agree within frozen
  tolerances.
- [x] **CP6.7** Add target-leakage, object-label, derived-mass, response-informed-cut, and
  duplicate-object mutation controls.
- [x] **CP6.8** Add leave-one-cluster-out, radial-block, and instrument-stratified folds.
- [x] **CP6.9** Freeze missing-row, censoring, extrapolation, and catastrophic-prediction
  rules.
- [x] **CP6.10** Complete a prospective power analysis and stopping rule.

The executable control receipt is
`runs/gravity/publication-readiness/numerical-controls-v1.json`. All five injected
classes—Newtonian, MOND/RAR, Item 59, NFW, and a deliberately wrong reversed-NFW
control—are recovered, and the wrong law is barred from a physical claim. Across 4,096
Newtonian null trials and all 2,025 Item 59 variants, 70 trials select a qualifying family:
a `1.709%` false-selection rate (95% Wilson interval `1.355%–2.154%`). CPU, RTX 5090
CUDA, and a separately written direct scorer agree to better than `2.8e-11` in score.
The candidate remains ahead of NFW in all eight leave-one-cluster-out folds and both
observable/instrument strata, but loses in the outer `0.7–2.0 R500` block. A conservative
paired power calculation calls for 192 independent clusters; a 120-cluster program has
only approximately `72.7%` projected power. These weaknesses are retained as planning
constraints, not used to erase the development result.

## CP7 — freeze an independent source and confirmation split

Advance criteria: an eligible population is frozen without opening candidate-target
responses and has no X-COP overlap or reduction-family dependence hidden as independence.

- [x] **CP7.1** Audit CHEX-MATE, LoCuSS, independent Chandra samples, and available
  ACT/SPT thermodynamic profiles for usable direct observables and covariances.
- [ ] **CP7.2** Select one primary independent thermodynamic sample and one secondary
  replication lane.
- [ ] **CP7.3** Remove and record every X-COP object overlap.
- [x] **CP7.4** Freeze population, selection function, quality cuts, radial range, and
  minimum information requirements.
- [x] **CP7.5** Freeze development and untouched whole-cluster confirmation subsets by a
  predictor-blind rule.
- [x] **CP7.6** Freeze object aliases and cross-survey duplicate detection.
- [x] **CP7.7** Freeze one primary endpoint, absolute-accuracy threshold, comparator
  threshold, maximum catastrophic fraction, and per-observable requirement.
- [x] **CP7.8** Freeze missing-data and exclusion rules before responses.
- [ ] **CP7.9** Write metadata-only source receipts and payload commitments.
- [x] **CP7.10** Require explicit authorization before any independent target row opens.

CHEX-MATE is a representative 118-cluster XMM program with a dedicated temperature
pipeline (https://arxiv.org/abs/2402.18653). LoCuSS offers an independent 50-cluster
X-ray and weak-lensing comparison (https://arxiv.org/abs/1511.01919). Eligibility depends
on actual public payloads, overlap, covariance, and licensing—not the paper abstract.
The metadata-only audit and frozen transformation rules are machine-bound in
`configs/gravity_cluster_independent_data_contract_v1.json`. It found zero fully ready
lanes and opened zero payloads or target rows; CP7.2 therefore remains blocked rather
than treating archive or paper availability as a complete replication packet.

The preselection protocol is machine-bound in
`configs/gravity_cluster_independent_replication_protocol_v1.json` and
`runs/gravity/publication-readiness/independent-replication-protocol-v1.json`. It freezes
the population and quality rules, whole-cluster hash split, duplicate handling, primary
joint score, absolute and comparator thresholds, missing-data rules, stopping rule, and
explicit authorization schema while target access remains false. The conservative power
target is 192 untouched confirmation clusters. A 120–191 cluster run remains useful but
must be labeled underpowered exploratory replication; fewer than 120 may not open as the
primary trial. CP7.2, CP7.3, and CP7.9 stay open until a real source inventory and sealed
file commitments exist.

## CP8 — unchanged independent thermodynamic replication

Advance criteria: the exact formula and declared nuisance hierarchy predict new pressure
and temperature profiles under the frozen contract.

- [ ] **CP8.1** Run source and schema validation before response access.
- [ ] **CP8.2** Open the authorized independent confirmation once.
- [ ] **CP8.3** Make zero formula, kernel, coefficient, threshold, or exclusion changes.
- [ ] **CP8.4** Score both pressure and temperature separately and jointly.
- [ ] **CP8.5** Report absolute residuals, calibration, likelihood, and comparator-relative
  performance.
- [ ] **CP8.6** Require the result not be carried by one cluster, one observable, or one
  radial region.
- [ ] **CP8.7** Run leave-one-cluster-out and frozen symmetric influence trimming.
- [ ] **CP8.8** Report performance by mass, redshift, temperature, morphology, relaxation,
  and instrument strata.
- [ ] **CP8.9** Retain every failed object and its data-quality audit.
- [ ] **CP8.10** Do not repair on confirmation; register any descendant as a new version
  needing a different external sample.
- [ ] **CP8.11** Classify same-release X-COP confirmation as development evidence and this
  gate, if passed, as independent replication.

## CP9 — independent mass probes

This gate is optional for the bounded thermodynamic paper but strongly increases its
physical value. It is mandatory before a gravity-mechanism claim.

- [ ] **CP9.1** Predict cluster galaxy-velocity-dispersion profiles.
- [ ] **CP9.2** Compare with caustic or dynamical measurements without using them for
  formula selection.
- [ ] **CP9.3** Compare hydrostatic and weak-lensing acceleration with like-for-like radial
  definitions.
- [ ] **CP9.4** Separate relaxed and disturbed clusters.
- [ ] **CP9.5** Separate BCG-dominated, member-dominated, and gas-dominated radii.
- [ ] **CP9.6** Test residual correlation with dynamical state, temperature, mass,
  redshift, morphology, and line-of-sight structure.
- [ ] **CP9.7** Test whether one coefficient transfers across cluster mass.
- [ ] **CP9.8** Keep lensing and dynamics as confirmation channels, not target-derived
  predictors.

## CP10 — group-scale and domain-boundary map

This gate is not required for the bounded cluster paper. It is required to claim a
continuous galaxy-to-cluster mechanism.

- [ ] **CP10.1** Assemble a source-audited X-ray group sample spanning roughly
  `10^13–10^14` solar masses.
- [ ] **CP10.2** Freeze whole-group splits and direct thermodynamic endpoints.
- [ ] **CP10.3** Apply the cluster formula unchanged before adding a transition.
- [ ] **CP10.4** Test continuous acceleration, compactness, pressure, entropy, geometry,
  external-field, and timescale variables.
- [ ] **CP10.5** Forbid galaxy, group, and cluster identity labels.
- [ ] **CP10.6** Require a transition to predict held-out groups and both endpoint domains.
- [ ] **CP10.7** Preserve nonmonotonic or multi-regime behavior rather than forcing a
  binary switch.
- [ ] **CP10.8** Treat group failures as domain mapping, not retroactive erasure of the
  cluster result.

## CP11 — gravity-theory and lensing escalation

This gate is not required for the bounded empirical paper. It becomes mandatory when
the wording attributes the relation to gravity or claims an alternative to dark matter.

- [ ] **CP11.1** Derive the weak-field law from an action or a closed field system.
- [ ] **CP11.2** Construct the baryonic source covariantly and causally.
- [ ] **CP11.3** Derive energy-momentum conservation and constraint propagation.
- [ ] **CP11.4** Establish positive energy, no ghosts, no gradient instabilities, and
  hyperbolicity on the declared domain.
- [ ] **CP11.5** Restore Solar-System and high-acceleration GR limits.
- [ ] **CP11.6** Pass strong-field and gravitational-wave propagation constraints.
- [ ] **CP11.7** Generalize the radial operator to two or three dimensions.
- [ ] **CP11.8** Derive both metric potentials and one universal massive-matter/photon
  coupling.
- [ ] **CP11.9** Freeze gravitational slip before direct lensing access.
- [ ] **CP11.10** Predict image positions, parities, shapes, shear, magnification, and time
  delays directly.
- [ ] **CP11.11** Include mass-sheet degeneracy, line-of-sight structure, PSF, source
  redshift, and baryonic-map uncertainty.
- [ ] **CP11.12** Prohibit a separate fitted lensing coefficient or a GR/NFW-derived
  lensing-mass target.

## CP12 — external reproduction and manuscript package

Advance criteria: another analyst can reproduce the bounded claim and audit every
limitation without private state.

- [x] **CP12.1** Provide one command that recreates every primary table and figure.
- [x] **CP12.2** Freeze environment, dependencies, random seeds, hardware tolerances, and
  source revisions.
- [ ] **CP12.3** Publish source, calibration, covariance, transformation, split, and
  exclusion manifests.
- [x] **CP12.4** Publish machine-readable per-row predictions, residuals, object summaries,
  and counterexamples.
- [x] **CP12.5** Preserve blinded and unblinded receipts and access counts.
- [ ] **CP12.6** Obtain a separately written implementation or independent analyst replay.
- [x] **CP12.7** Report all ablations, negative results, nuisance edges, and sensitivity
  envelopes.
- [x] **CP12.8** Report absolute performance alongside relative improvements.
- [x] **CP12.9** Include the three-track claim adjudication in the manuscript evidence
  package; carry it unchanged into the eventual prose manuscript.
- [ ] **CP12.10** Obtain statistical, cluster-astrophysics, and modified-gravity review.
- [ ] **CP12.11** Release code and eligible data artifacts under declared licenses.
- [ ] **CP12.12** Submit only the bounded claim unless CP9–CP11 independently pass.

The development package is frozen by
`configs/gravity_cluster_manuscript_evidence_package_v1.json` and reproduced at
`runs/gravity/publication-readiness/manuscript-evidence-package-v1.json`. It completes
the evidence-assembly portions of CP12. The exact primary artifact inventory is frozen by
`configs/gravity_cluster_manuscript_renderer_v1.json`; one `sigma-cluster-manuscript
render` command recreates all seven tables and six figures and writes the hash-bound
manifest at
`runs/gravity/publication-readiness/manuscript-artifact-manifest-v1.json`. These local
artifacts do not substitute for the still-open independent implementation,
source/covariance release, expert reviews, licensing, or submission decision.

## Frozen primary statistical decision to design before CP7 opens data

The pre-data configuration must choose exact numerical values for:

1. the primary covariance-aware predictive likelihood or score;
2. required absolute pressure and temperature accuracy;
3. required performance against NFW, Einasto, and the strongest empirical comparator;
4. maximum catastrophic-cluster fraction;
5. minimum independent sample size and power;
6. leave-one-cluster-out and influence-trim stability;
7. calibration by radius, observable, mass, redshift, and dynamical state;
8. multiplicity control across secondary endpoints;
9. permitted nuisance priors and whether any posterior hits a prior edge;
10. the stopping and no-repair rule.

These thresholds must be calibrated on simulations, controls, and development data—not
on independent confirmation outcomes.

## Definition of data-ready

The project is `DATA_READY_FOR_INDEPENDENT_CLUSTER_REPLICATION` only when CP0, CP1,
CP3, CP4, CP5, CP6, and CP7 are all PASS, the candidate identity and code are hash-bound,
and independent target authorization is still false. Data-ready means it is safe to ask
for explicit authorization to open the frozen target; it is not a scientific pass.

## Definition of a bounded-paper result

The project is `BOUNDED_CLUSTER_PAPER_READY` only when CP0–CP8 and CP12 pass, allowing
CP9–CP11 to remain explicitly not required. A failed CP8 is publishable as a transparent
replication failure or methods/failure-space result, but not as positive independent
confirmation.

The project may not use `PHYSICAL_MECHANISM_READY` until CP9 and alternative-cause
separation pass. It may not use `UNIVERSAL_GRAVITY_THEORY_READY` until CP10 and CP11 pass
and the unchanged theory also survives the existing galaxy, local, strong-field, and
cosmological ladders.
