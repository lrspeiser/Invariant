# No-dark-matter gravity discovery goals

This is the ordered scientific program for Invariant as of **2026-08-27**. It supersedes
the older gravity sequencing in `DISCOVERY_GOALS.md` and `CONTINUOUS_DISCOVERY_ROADMAP.md`
where they conflict. Work does not advance past a numbered gate until that gate has a
replayable PASS. A REJECT is retained as an exclusion result and sends the search back to
the nearest generative stage; it is not repaired by weakening the later test.

## Claim boundary

The target is one theory that predicts galaxy dynamics, cluster dynamics, and lensing from
observed baryons and universal gravitational fields, with no non-baryonic dark-matter
stress-energy, halo profile, halo label, or object-specific gravitational parameter.

An extra scalar, vector, tensor, or nonlocal degree of freedom is permitted only when it is
declared as part of gravity, universally coupled, and carried through the theory's energy,
stability, lensing, local-gravity, gravitational-wave, and cosmological tests. If such a
field behaves like an unseen matter component, the result must say so; relabelling it does
not satisfy the goal.

SPARC does **not** provide the individual speeds of every star. It provides spatially
resolved circular-velocity tracers, primarily from H I and H-alpha observations, plus
photometric and gas mass models. The first empirical claim is therefore “predict every
admitted rotation-curve tracer within its measurement contract,” not “predict every star.”
The [SPARC project](https://astroweb.cwru.edu/SPARC/) describes 175 late-type galaxies;
the [primary paper](https://arxiv.org/abs/1606.09251) documents the sample and mass models.

A formula fitted separately to every galaxy is a **diagnostic atlas**, not an alternative
to general relativity. Per-object freedom makes the inverse problem deliberately easy so
that the system can measure how the local solutions vary. The scientific climb is to
compress that atlas into one law, lift the law into a covariant action, and survive
independent phenomena that rotation curves alone cannot identify.

## Current measured starting point

- The repository carries the full 175-galaxy, 3,391-point SPARC mass-model distribution.
- The existing name-hash split declares 140 exploration and 35 confirmation galaxies.
  One exploration galaxy fails the preregistered baryonic admission rule, leaving **139
  admitted exploration galaxies and 2,720 fitted points**. Confirmation galaxies have not
  been handed to the full-sample fitting entry point.
- Fourteen existing one-parameter law spaces have been fitted independently to all 139
  admitted exploration galaxies: **1,946 local fits**. Every law's parameter varies across
  the population; 0/14 is constant within the declared intervals. This is evidence that the
  current grammar has not found a universal law, not evidence that no such law exists.
- The current GPU receipts prove high throughput on simpler synthetic/static screens. They
  do not establish the throughput of a covariance-aware rotation-curve evaluator. The first
  gravity benchmark below must measure that rate before extrapolating to trillions.

## The ordered gate ladder

| Gate | Goal | Real problem and data | Advance only if | Current |
|---|---|---|---|---|
| **G0** | Freeze the observational and compute experiment | Replay published Newtonian-baryon, empirical RAR/MOND, deliberately wrong, and flexible GR+halo comparator predictions on the 139 SPARC exploration galaxies; benchmark the real evaluator on the RTX 5090. | Source manifests, units, nuisance policy, radial folds, whole-object splits, baselines, score, formula budget, and stopping rule are frozen; leakage controls fail closed; GPU/CPU score replay agrees within the declared numerical bound. | **PASS** — receipt `runs/gravity/g0-experiment/receipt-v1.json`; 139 galaxies/2,720 rows, zero confirmation evaluator accesses, 2.72 billion measured candidate-point evaluations, zero FP64 GPU/CPU mismatches. |
| **G1** | Build a predictive per-galaxy formula atlas | For each of 139 admitted SPARC exploration galaxies, predict withheld contiguous radial blocks from baryonic observables without a halo input. | All 139 galaxies have at least one dimensionally valid, compressed formula that clears the frozen within-galaxy predictive gate; every surviving formula and every failed family is retained. | **PASS** — the sealed union covers 139/139 after 14,006,081,328 cumulative candidate–galaxy trials. V1's NGC2955 counterexample and the failed 100M-trial repair remain retained; a disclosed interaction repair adds 24 CPU-FP64 survivors. Confirmation remains untouched. |
| **G2** | Collapse cosmetic formulas into real solution classes | Canonicalize and compare every G1 survivor across all 139 real galaxies. | Algebraic rewrites, unit reparameterizations, parameter renamings, and behaviorally indistinguishable formulas are merged or explicitly marked unresolved; mutation controls prove the equivalence detector can separate near misses. | **PASS** — 8,615 survivors collapse to 8,609 structural and 6,326 behavioral classes. Rewrite/rescaling controls merge and near-miss mutations separate. Statuses are 5,385 `KNOWN_FAMILY`, 3,224 `COMBINATION`, and zero novelty claims. |
| **G3** | Discover what generates the per-galaxy variation | Predict a galaxy's G2 formula class and local coefficients from measured baryonic structure—surface brightness, gas fraction, scale lengths, gradients, morphology, and allowed environment measurements—under whole-galaxy cross-validation. | A target-blind meta-law predicts held-out galaxies' local formulas and rotation curves with zero galaxy-ID input and a positive preregistered gain over constant, nearest-neighbor, and empirical-relation baselines. | **PASS AFTER DISCLOSED V2 REPAIR** — fixed shrinkage 0.3 predicts all 139 galaxies with zero leakage. Projected chi-square is 123,472.313, a 5.54% gain over RAR and at least 0.5% over every frozen baseline. The choice followed v1 diagnostics, so this is model-development evidence, not independent confirmation. |
| **G4** | Freeze one universal galaxy law | Refit G3 into one formula with universal gravitational constants and zero per-galaxy gravitational parameters, then open the 35 SPARC confirmation galaxies once. | The frozen law beats Newtonian baryons, meets the preregistered comparison to RAR/MOND and the flexible halo performance ceiling, is calibrated across morphology/surface-brightness strata, and receives no post-open adjustment. | **CONSTRUCTION AUTHORIZED; CONFIRMATION LOCKED** — G3-v2 passed. A compact universal candidate must now clear all exploration comparators before the one-shot confirmation evaluator may open. |
| **G5** | Replicate through independent measurement pipelines | Without refitting the G4 law, predict the 19 high-resolution THINGS rotation curves and 26 LITTLE THINGS dwarf-galaxy rotation curves; separate objects overlapping SPARC from genuinely new objects. | The frozen law passes the same direct-observable score on both the overlap-consistency and new-object subsets; instrument/pipeline changes do not create a systematic residual. | **NOT STARTED.** |
| **G6** | Build and reconcile a cluster formula atlas | On the 12 X-COP clusters, infer local baryon-to-dynamics relations from XMM-Newton density/temperature data and Planck SZ pressure data, with member-galaxy baryons and nonthermal pressure treated as declared nuisances rather than dark matter. | Each cluster first has a radial-holdout local solution; those solutions then compress to the same G4 law with zero cluster-specific gravitational parameter and predict held-out X-ray/SZ observables, not a GR-derived mass map. | **Evaluator schema only; no authorized real packet.** |
| **G7** | Find one cross-scale weak-field operator | Compare G2 galaxy classes with G6 cluster classes and search for a shared local operator, environment law, or nonlocal kernel that generates both. | A single dimensionally valid operator predicts held-out galaxies and held-out clusters, carries no object label, and forward-generates the accepted local formula classes within their uncertainty. | **NOT BUILT.** |
| **G8** | Make the same law bend light | Freeze the G7 operator and test a covariant completion on the 25 CLASH clusters using calibrated HST images, multiple-image positions/parities, and audited time delays where available. | With no lensing-only gravity parameter or GR-derived mass map as target, the same law beats baryon-only GR and is non-inferior to the preregistered standard lens model on direct lensing observables. | **Schema controls only; no real result.** |
| **G9** | Invert the atlas into root-cause theories | Build a verified graph from local formula classes to shared weak-field operators to covariant actions/field equations. Run blinded Newton/Poisson, Yukawa/Klein-Gordon, and AQUAL-like controls before applying it to G7. | Every edge is checked by forward derivation; the candidate action reproduces G7, supplies G8 lensing from the same fields, obeys dimensions and conservation identities, and reports all action classes still observationally indistinguishable. | **Pieces exist; no data-driven inverse graph.** |
| **G10** | Survive local and strong-field gravity | Test the unchanged G9 action against Cassini solar-conjunction tracking, the Double Pulsar, and GW170817/GRB 170817A propagation constraints. | One parameter set passes Solar-System PPN, strong-field timing/radiation, gravitational-wave speed/dispersion/polarization, stability, hyperbolicity, and positive-energy gates. | **Formal controls partial; zero completed direct-observation trial.** |
| **G11** | Survive cosmology without cold dark matter | Evaluate the unchanged theory with the public Planck 2018 CMB likelihoods and DESI DR2 BAO measurements, then structure-growth and cosmological-lensing data. | A declared no-non-baryonic-dark-matter cosmology gives an acceptable preregistered joint likelihood and does not introduce an undeclared effective matter component or a new set of phenomenon-specific constants. | **NOT STARTED.** |
| **G12** | Independent replication and novelty adjudication | Freeze code, theory, data transformations, exclusions, and prior-art corpus; have an independent analyst replay a new-object test and inspect the claimed construction. | Independent replay passes; known rewrites and known-family instances are removed from the novelty claim; human domain review supports the exact bounded claim. | **NOT STARTED.** |

The primary work is therefore **G0, then G1**. Formal completion of old gravity families is
supporting work only when it supplies a candidate needed by this ladder.

## G0 — experiment and throughput freeze

G0 writes the frozen contract in `configs/gravity_g0_experiment.json` and its bound baseline,
fold, data-access, and throughput evidence in
`runs/gravity/g0-experiment/receipt-v1.json`. Together they include:

- Exact source bytes, citations, transformations, units, uncertainty/covariance treatment,
  selection and exclusion rules, and the existing 139/35 split.
- Within-galaxy folds made from contiguous radial blocks, never randomly interleaved points.
  Rotation-curve rows from one galaxy are correlated and cannot masquerade as independent
  objects.
- Baselines: Newtonian/weak-field GR using the same baryons; a frozen empirical RAR/MOND
  relation; a deliberately false law; and a flexible GR+halo model used only as a
  performance ceiling, never as an input, target, rescue, or source of labels.
- A predictive score conditional on SPARC's published random-error column, an explicit warning
  that systematic inclination covariance is unavailable here, a symbolic description-length
  cost, calibration diagnostics, and thresholds frozen from controls before formula search.
- Measured candidates/second, point-evaluations/second, memory, power or GPU time, survivor
  rate, CPU replay rate, and error bound for the actual multi-radius evaluator. Synthetic
  screen throughput may be quoted separately but may not be substituted.

G0 fails if a baseline cannot be reproduced, any forbidden value reaches a proposer, a GPU
survivor cannot be replayed, or a threshold is chosen after candidate outcomes are seen.

### G0 measured result

The checked G0 receipt passed on 2026-08-27. The empirical RAR comparator reduced aggregate
held-out chi-square from `1.697326397883e+06` for Newtonian baryons to
`1.307146893155e+05`. The deliberately wrong high-acceleration boost remained at
`1.622323189132e+06`. The training-radius-only two-parameter NFW-shaped performance ceiling
reached `2.801880693058e+04`. These are comparator checks, not claims that RAR or NFW is the
true law.

On the NVIDIA GeForce RTX 5090, the actual formula scorer evaluated 1,000,000 candidates on
all 2,720 exploration rows: 2.72 billion candidate-point evaluations in about 0.957 seconds,
or about 2.84 billion candidate-point evaluations per second. Its measured GPU memory-pool
increment was 606,820,352 bytes. The 4,096-candidate FP64 CPU/GPU replay had zero finite-status
or tolerance mismatches. Only 0.1053% of candidates in this broad rational grammar were finite
at every row; this is a domain-validity prefilter rate, not an observational-survivor rate.

G0 does not establish a new gravity formula. It authorizes G1 to search under the frozen rules.

## G1 — creative search and the formula atlas

For each galaxy, search three arms:

1. **Structured/Occam:** enumerated dimensionally typed expressions, symbolic regression,
   evolutionary search, e-graphs, nonlocal kernels, and reusable physical operators.
2. **Pseudorandom:** a deterministic random permutation of the same declared ordinal space,
   checkpointed without duplicates.
3. **Creativity-guided:** LLM-proposed new grammars, invariants, representations, cross-domain
   analogies, and recombinations; once admitted, their instances receive the same GPU budget
   and evaluator as the other arms.

The first pilot is 12 morphologically diverse exploration galaxies with at least **10 million
canonical candidates per arm per galaxy**: at least 360 million screened candidates. Production
starts at **100 million distinct candidates per galaxy in total** (13.9 billion across 139),
then scales to **1 billion per galaxy** (139 billion) if new equivalence-class yield remains
positive. The declared stretch ceiling is **10 billion per galaxy, 1.39 trillion total**. It is
not automatic: the actual G0 evaluator throughput, energy/cost, survivor-replay capacity, and
novel-class yield determine whether another decade is scientifically useful.

GPU screening should use a cheap cascade: dimensional/type rejection; 6–12 anchor radii;
all training radii; radial holdouts; CPU/high-precision replay; nuisance and robustness scoring.
The GPU may reject or rank but may not alone accept an observational law.

A local formula may carry at most two declared galaxy-local diagnostic constants in G1. They
are charged in description length and are forbidden after G3. Lookup tables, galaxy IDs,
unbounded splines, target-derived discrepancies, halo quantities, and post-hoc baryonic
recalibration are forbidden. A survivor must compress the curve substantially relative to
tabulating its targets and pass every frozen radial fold. G1 passes only at 139/139 admitted
exploration galaxies; failures stay visible as grammar counterexamples.

The atlas `galaxy-formula-atlas-v1.json` retains the Pareto set, not only the winner: formula
IR, dimensional type, free constants, search arm, ordinal/seed, parent lineage, fit/holdout
scores, residual structure, compute cost, and all counterexamples.

## G2–G4 — from many local answers to one galaxy law

G2 writes `gravity-formula-equivalence-classes-v1.json`. Every proposer supplies a
non-authoritative origin label:

- `known_rewrite`
- `known_family_instance`
- `new_combination_of_known_ideas`
- `proposed_new_construction`
- `uncertain`

An independent equivalence pass then checks algebraic canonical forms, units, limiting cases,
e-graph rewrites, proof-family membership, numerical behavior on adversarial design grids, and
the cited prior-art corpus. Its authoritative status is one of `KNOWN_EQUIVALENT`,
`KNOWN_FAMILY`, `COMBINATION`, `STRUCTURALLY_UNMATCHED_IN_DECLARED_CORPUS`, or `UNRESOLVED`.
“Structurally unmatched” is not “novel.” Disagreement with the proposer preserves the branch
and adds an ambiguity record; it never silently prunes it.

G3 writes versioned `galaxy-formula-meta-law` receipts. It treats the G2 atlas as a population-level
inverse problem: what measured baryonic features generate the observed choice of formula and
coefficients? Evaluation is by whole-galaxy folds. The meta-law must generate the formula for
an unseen exploration galaxy without receiving that galaxy's velocities or identity.

G4 writes `universal-galaxy-law-v1.json`. It removes every local gravitational constant,
freezes the remaining universal values and nuisance hierarchy, and consumes the 35-galaxy
SPARC confirmation test once. A rejected confirmation cannot be retuned on those 35 galaxies;
the next attempt must be a new declared theory version and must use an independent external
confirmation source.

## G5–G8 — external galaxies, clusters, and light

THINGS supplies 19 high-resolution rotation curves in its
[primary mass-model paper](https://arxiv.org/abs/0810.2100). LITTLE THINGS supplies 26 dwarf
rotation curves in its [primary paper](https://arxiv.org/abs/1502.01281) and published tables
through [VizieR J/AJ/149/180](https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=J%2FAJ%2F149%2F180).
These are pipeline- and morphology-transfer tests. Overlap with SPARC must be reported rather
than counted as independent objects.

X-COP supplies thermodynamic profiles for 12 clusters from joint XMM-Newton and Planck data;
the [primary profile paper](https://arxiv.org/abs/1805.00042) and
[project data release](https://dominiqueeckert.wixsite.com/xcop/data) define the starting
source. Cluster “dynamics” is not a rotation curve. The forward model must connect baryons and
gravity to gas density, temperature, pressure/SZ, and member-galaxy kinematics. Hydrostatic
acceleration may be a diagnostic intermediate, but a GR-derived hydrostatic mass or lensing
mass may not be treated as observed truth.

CLASH observed 25 massive clusters. The [official archive overview](https://irsa.ipac.caltech.edu/data/SPITZER/CLASH/overview.html)
links the HST products, catalogues, and lens models; the
[survey paper](https://arxiv.org/abs/1106.3328) documents the design. The accepted G8 target
is the direct image/arc/time-delay likelihood. Published GR lens models are comparators and
calibration aids, not target mass maps.

G7 is the cross-scale compression test. It must write `cross-scale-gravity-operator-v1.json`
and demonstrate that the same weak-field operator generates both the galaxy atlas and cluster
atlas. A galaxy law plus an unrelated cluster patch is a fail.

## G9 — inverse solution graph and root cause

Rotation curves do not determine a unique field theory. Many actions can share one
quasistatic limit; dynamics alone may not separate metric, scalar, vector, nonlocal, emergent,
or modified-inertia explanations. The engine must therefore find an **equivalence class of
causes**, then use clusters, lensing, local tests, waves, and cosmology to shrink it.

G9 writes `inverse-gravity-solution-graph-v1.json` with three node layers:

```text
139 galaxy formula classes + 12 cluster formula classes
                         ↓ verified forward-generation edges
          shared weak-field operators / Green kernels
                         ↓ verified variational-limit edges
             covariant actions and field equations
```

The search has two inverse stages:

- **Meta-symbolic/operator inversion:** enumerate local differential operators,
  environment-dependent laws, and finite-parameter nonlocal kernels whose solutions generate
  the G2/G6 local formula classes.
- **Inverse variational lift:** enumerate typed covariant actions whose Euler-Lagrange equations
  have the accepted G7 operator as their quasistatic limit.

Known controls must recover bounded versions of Newton/Poisson, Yukawa/Klein-Gordon, and an
AQUAL-like nonlinear-Poisson family while rejecting perturbed non-variational controls. For
the discovery run, every candidate edge must be replayed in the forward direction. The graph
retains every observationally equivalent upstream action rather than choosing a “root cause”
by taste.

The search can exhaust a declared finite grammar, not the infinite set of all mathematical
theories. Its strongest negative is “no action in grammar/version/range X generates these
operators.” Every such exclusion is stored in the counterexample ledger with the exact failed
obligation and witness, allowing later grammars to avoid whole dead families.

## G10–G12 — an alternative to GR rather than a curve formula

The unchanged G9 action must be confronted with:

- Cassini's solar-conjunction result, `gamma = 1 + (2.1 ± 2.3) × 10^-5`, from the
  [primary radio-link paper](https://www.nature.com/articles/nature01997).
- The 16-year Double Pulsar analysis, which reports multiple post-Keplerian tests and a
  quadrupolar-radiation test at `1.3 × 10^-4`, in the
  [primary paper](https://arxiv.org/abs/2112.06795).
- The joint GW170817/GRB 170817A constraint on gravity's propagation speed, documented in the
  [LIGO/Virgo publication record](https://dcc.ligo.org/P1700308/public), plus waveform,
  dispersion, and polarization tests.
- The [official Planck 2018 likelihood products](https://esdcdoi.esac.esa.int/doi/html/data/astronomy/planck/Cosmology.html)
  and [DESI DR2 BAO products](https://data.desi.lbl.gov/doc/papers/dr2/).

These gates use one ontology and one parameter set. Screening or environment dependence may
be a prediction of the field equations; it may not be a phenomenon-specific switch fitted
separately to galaxies, clusters, the Solar System, waves, and cosmology.

## Common rules at every gate

1. **Creativity is upstream.** LLMs may propose new representations, operators, analogies,
   grammar extensions, and repairs. They may not score their own proposal or see sealed targets.
2. **Labels do not prune.** A proposer must label rewrite versus combination versus proposed
   construction, but uncertain or conflicting labels preserve the idea until equivalence and
   empirical tests resolve it.
3. **All formulas that work are retained.** Ranking is Pareto over predictive score,
   calibration, complexity, robustness, and compute. No single scalar leaderboard deletes the
   rest of the solution space.
4. **Failures accumulate at family level.** A counterexample should exclude the largest
   justified parameter cell or formula family, not only one string.
5. **No hidden per-object rescue after G3.** Galaxy/cluster IDs, halo values, target-derived
   residual labels, and private constants are prohibited.
6. **Direct observables outrank inferred mass maps.** Derived accelerations and masses are
   diagnostics; final scoring returns to photons, spectra, image geometry, timing, or other
   audited measurements whenever feasible.
7. **No uniqueness from a finite search.** `unique_in_grammar_vN` and
   `observationally_indistinguishable` are valid; unqualified “the unique theory” is not.
8. **No novelty from absence.** Literature absence, proposer labels, and structural distance
   are evidence for review, never proof of originality.
9. **Compute claims are measured.** “Trillion formulas” means distinct declared canonical
   ordinals actually evaluated, with chance matches, duplicates, throughput, energy/cost, and
   CPU replay reported.
10. **The next gate remains locked until PASS.** A failure returns to G1, G3, G7, or G9 with
    its exclusion knowledge; it does not change the downstream threshold.
