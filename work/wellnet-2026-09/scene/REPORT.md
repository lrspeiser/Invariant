# Stage 1: the probabilistic four-dimensional gravitational scene graph

Lane: `work/wellnet-2026-09/scene/`. Every number below is rendered from `scene_results.json` and `test_results.json` by `write_report.py`; none is typed in by hand.

The charter's fundamental data object is a probabilistic 4-D gravitational scene graph, and it says plainly that *"the fundamental data object should not be a spreadsheet row."* Everything this programme had built worked from averaged radial profiles and catalogue rows. This lane is the missing foundation: a scene schema with an enforced metadata contract, an ensemble sampler that keeps a scene a posterior, an averaging-commutation gate that refuses a substitution when the commutator is not negligible, and an availability matrix saying which clusters can actually carry such a scene.

`38` of `38` tests pass. They found **eight bugs** in this lane's own first implementation; each is described where it belongs below.

## 1. The schema

- **15 node types** from the charter's 12 node bullets (its "Voids, filaments, saddles, and boundaries" bullet is expanded to four types, since each has a different support).
- **10 edge types** and **8 field types**, exactly the charter's lists.
- **67 quantities** in the ontology, each carrying all **17** metadata-contract items. The contract audit reports `all_complete = True`.

The contract is enforced at construction, not checked afterwards:

| Rule | Enforced how |
|---|---|
| A potential that shifts under a change of origin must name a boundary rule | `Quantity.__post_init__` raises `ContractError`; `gauge_unsafe` is `empty` |
| A logarithm of a dimensionful quantity is not defined | `ContractError` at construction |
| An unregistered quantity cannot enter a scene | `SceneGraph._check_attrs` raises |
| A bare number cannot enter a scene | must be `Fixed` or `Uncertain`, so "known" and "sampled" can never be confused |
| Units are checkable | exponent vector over (M, L, T, Θ, Q) with exact `Fraction` exponents, not a string |

Of the ontology's quantities, **22** are `constructible`, **14** are `marginalisable`, **26** are `measured`, **5** are `non_identifiable`. That four-way split is load-bearing and is described in §4.

`39` quantities do **not** commute with averaging and may not be read off an averaged scene without clearing the gate in §3. `1` (`n_wells`) additionally depend on how a deblender happened to partition the image.

### Ontology coverage against the charter's seventeen sections

| § | Section | Quantities |
|---|---|---|
| 1 | Location, time, scale, and reference | 8 |
| 2 | Amount and composition of matter | 9 |
| 3 | Thermodynamic and material state | 3 |
| 4 | Source geometry | 5 |
| 5 | Motion of matter | 6 |
| 6 | Local gravitational descriptors | 9 |
| 7 | Spacetime and curvature descriptors | 2 |
| 8 | Directional structure | 2 |
| 9 | Environment and cosmic-web state | 1 |
| 10 | Network-of-wells parameters | 2 |
| 11 | Nonlocal and path variables | 2 |
| 12 | History and memory | 1 |
| 13 | Light and radiation observables | 6 |
| 14 | Cosmological and large-scale parameters | 0 |
| 15 | Speculative latent-state parameters | 3 |
| 16 | Universal constants and transition scales | 3 |
| 17 | Measurement and astrophysical nuisances | 5 |

Section(s) `14` are deliberately empty: cosmological parameters become relevant only after a local and cluster law survives, and populating them now would invite a candidate to fit them before it has earned the right to.

### The five exact identities

Recorded symbolically so the compiler can take the RANK of a candidate's variable set before any fitting. This programme's `variable-lists-collapse` finding is that rich-looking variable sets shrink under identities.

| Redundant given | Relation |
|---|---|
| `g_N` given `M_enc, r_3d` | g_N = G M_enc / r_3d^2 (spherical) |
| `v_circ` given `g_total, r_3d` | v_circ^2 = g_total r_3d |
| `r_3d` given `r_proj, z` | r_3d^2 = r_proj^2 + z^2 |
| `P_e` given `n_e, T_x` | P_e = n_e k_B T_x |
| `R500` given `M_enc` | M(<R500) = 500 rho_crit (4/3) pi R500^3 |

A demonstration scene exercising every node and edge type contains `143` nodes and `54` edges, of which `144` attributes are `Uncertain`. Its structural fingerprint is `f38652e97f564369`. One node is flagged `presupposes_dm`, and the reason is recorded with it:

> `kappa_map` — a lens model that assigns a dark-matter clump to each cluster galaxy by construction is circular for any does-lensing-follow-light test

## 2. The ensemble sampler: a scene is a posterior

The line-of-sight depth of a cluster member is not *noisy*. It is **absent**: `cz = H(z)d + v_pec` is one equation in two unknowns, and the Finger-of-God distortion makes any depth inferred from velocity *anti*-correlate with true 3-D radius. Any single-value substitute is a fabrication, so the sampler produces a posterior:

```
p(z_i | R_i, v_i, morph_i, theta)  proportional to
    n_3d(sqrt(R_i^2 + z_i^2))     <- where the galaxies ARE
  x N(v_i ; 0, sigma_los^2(r_i))  <- cluster phase space
  x p(morph_i | r_i)              <- morphology-density relation
  x S(R_i, z_i)                   <- spatial selection + scene volume
```

**No term is a mass model.** `n_3d` is the analytic Abel deprojection of the observed projected member counts and `sigma_los(r)` is the observed dispersion profile. Neither assumes a halo, an NFW profile, or a gravity law — which is mandatory, since the gravity law is the thing under test. A test asserts mechanically that no NFW or halo appears anywhere in the module.

**Uncertainty is represented by a sampler, not an error bar.** An `Uncertain` value carries a draw function, so a bounded, skewed or multi-modal posterior survives. Depths are drawn *jointly*: a substructure-level bulk offset is drawn first and member depths are conditioned on it, because independent per-member depths would destroy exactly the correlated lumpy geometry a network law is meant to see. The measured mean pairwise depth correlation is `0.0643`.

### Calibration

Coverage is measured on a synthetic cluster of `400` members whose true depths are known:

| Nominal | Empirical | z | Calibrated |
|---|---|---|---|
| 0.50 | 0.517 | +0.70 | yes |
| 0.68 | 0.718 | +1.67 | yes |
| 0.90 | 0.922 | +1.68 | yes |
| 0.95 | 0.955 | +0.48 | yes |

All four levels are calibrated. The posterior is narrower than the spread of true depths by a factor `1.08` — and that ratio is the uncomfortable headline of this section.

**BUG 1, found by the coverage test.** The first version over-covered at every level and returned a posterior *wider* than the truth (information-gain ratio 0.86). The sampler's depth prior ran to ±5 Mpc while the scene declared a 3 Mpc volume: a *projected survey footprint* and a *declared scene volume* are different statements and the code conflated them. Truncating the prior at the boundary the scene actually claims fixed it. That sounds like a conservative error and is not — an over-dispersed depth ensemble washes out the correlated lumpy geometry the whole object exists to preserve.

**What the velocity is worth.** Comparing the depth posterior against the radial number-density prior *alone* gives a width ratio of `0.9543`: the line-of-sight velocity narrows the depth by about 4.6%, and no more. The depth posterior is very nearly the radial prior. That is the Finger-of-God statement quantified, and it means a scene ensemble is not a way of *recovering* depth — it is a way of being honest that depth is not there.

### Effective sample size, and BUG 8

| Morphology term applied as | ESS out of 64 draws |
|---|---|
| exact proposal | 64 |
| importance reweighted | 17.23 |

The first version applied the morphology-density term by importance reweighting. That is formally correct and numerically hopeless: the log weight is a **sum over members**, so its variance grows with N and the effective sample size collapsed to `17.2` of 64 at only `120` members — it would be far worse at the 300 a real cluster has. An ensemble whose ESS has collapsed is a point estimate wearing a posterior's clothes, the exact failure the module exists to prevent. The fix is structural rather than numerical: every factor is a one-dimensional function of `z` for a given member, so all of them go into the exact grid proposal and no weight is needed at all.

### Why the mean scene is not a scene

`E[f(scene)]` against `f(E[scene])` for the mean 3-D member radius:

| | Mpc |
|---|---|
| `E[f(scene)]` — law applied per realisation, then averaged | 1.33 |
| `f(E[scene])` — ensemble collapsed to its mean scene first | 0.958 |
| difference | 0.376 (**28.2%**) |

Collapsing the ensemble to its mean puts every member back in the plane of the sky, and understates every mean 3-D radius by **28%**. This is the charter's "do not collapse uncertainty to a best-fit scene" as a number.

## 3. The averaging-commutation gate

> *"Never replace a resolved scene with an averaged source unless the candidate law has been shown to commute with that averaging operation."*

Given a resolved scene `S`, an averaging operation `A`, a candidate law `F` and the observable `O` (which includes whatever averaging the *measurement* performs), the gate measures how much of the candidate's **deviation from a linear control** survives:

```
dev(scene) = O[F(scene)] - O[F_newton(scene)]
erased     = 1 - dev(A S) / dev(S)
```

Taking the deviation against a linear control on the same scene with the same probe configuration is what makes the number mean something. A linear law has `dev == 0` identically, so the control is exactly zero *by construction* and the gate cannot manufacture an erasure; and whatever the averaging does to any law divides out, leaving only the part attributable to the candidate's own structure.

### The null control

Newtonian gravity is linear in the source and rotationally covariant, so the shell average of the resolved field **equals** the field of the spherically averaged source exactly. The shell-averaged Plummer potential has a closed form, so this is checked against an analytic reference rather than against the gate's own other branch:

| r (kpc) | analytic | quadrature | relative error |
|---|---|---|---|
| 300 | 5.68283e-11 | 5.68246e-11 | -6.55e-05 |
| 1000 | 4.20501e-11 | 4.20389e-11 | -2.65e-04 |
| 2000 | 2.06718e-11 | 2.06741e-11 | +1.13e-04 |

**The gate's quadrature floor is `2.65e-04`**, and every number below must be read against it.

**BUG 3.** The first null control returned 0.24%, not zero — which would have swamped the ~0.4% signal it was built to measure. The cause was the probe lattice, not the physics: a cluster field on a probe shell is not smooth (individual galaxies come close to the shell), so one Fibonacci lattice leaves an error of the same size as the commutator. Raising the point count does **not** fix it — the error does not fall monotonically from 128 to 4096 points, because the near-singular sampling is lattice-structured rather than random. The fix is paired rotated quadrature: average over rigidly rotated copies of the lattice, using the same rotation indices in both branches of every commutator.

| n_dir | n_rot | max relative error |
|---|---|---|
| 256 | 1 | 9.33e-04 |
| 256 | 8 | 2.65e-04 |
| 768 | 24 | 1.22e-04 |

**BUG 2** was found on the way: the spherical-average operation expands one source into `n_dir` copies, and the unchunked pair array for a cluster scene is tens of gigabytes. The averaged branch is now computed from the closed-form shell-averaged Plummer potential, so it is both exact and free.

### The measured erasure matrix

Scene: `4300` sources (`300` galaxies plus a diffuse component, total `7.73e+14` solar masses, intrinsic flattening `q_z = 0.55`), probe radius `1000` kpc, target precision 1%.

| Erasure mode | Law | Averaging | Observable | erased | shift | verdict |
|---|---|---|---|---|---|---|
| nonlinearity | `qumond_rar` | `spherical_average` | `radial_g` | -0.4% | 0.22% | **ALLOW** |
| depth fabrication | `qumond_rar` | `los_collapse` | `radial_g` | -4.3% | 18.60% | **REFUSE** |
| directional, SOURCE axis | `source_axis` | `azimuthal_average` | `quadrupole` | 120.0% | 123.03% | **REFUSE** |
| directional, EXTERNAL axis | `external_axis` | `azimuthal_average` | `quadrupole` | -29.8% | 62.90% | **REFUSE** |
| network | `well_network` | `smooth_L300kpc` | `radial_g` | 100.0% | 32.84% | **REFUSE** |
| network vs merge | `well_network` | `catalogue_merge_150kpc` | `radial_g` | 50.3% | 9.20% | **REFUSE** |
| path | `path_column` | `spherical_average` | `dispersion` | 100.0% | 100.00% | **REFUSE** |
| memory | `memory` | `present_only` | `radial_g` | 100.0% | 20.35% | **REFUSE** |

`7` of `8` substitutions are refused. The one that is allowed is the charter's own A2029-like case:

> Replacing ~300 member galaxies with a spherically averaged source changes the shell-averaged QUMOND field by `0.22%` at `1000` kpc, against a quadrature floor of `2.65e-04`. The charter records about 0.4% for this experiment; the gate reproduces that order and adds that it is radius-dependent. So lumpiness does not explain a factor-of-two cluster discrepancy — and the substitution is nonetheless only admissible while the target precision stays above a percent.

The charter quotes that experiment as a single number. It is in fact strongly radius dependent, and the gate's verdict flips across the range a cluster analysis actually uses:

| r (kpc) | shift from spherical averaging | verdict at 1% |
|---|---|---|
| 200 | 2.09% | REFUSE |
| 300 | 1.08% | REFUSE |
| 500 | 0.58% | ALLOW |
| 700 | 0.31% | ALLOW |
| 1000 | 0.22% | ALLOW |
| 1500 | 0.35% | ALLOW |
| 2000 | 0.11% | ALLOW |

From 2.09% at 200 kpc to 0.11% at 2000 kpc — a factor of 19. Quoting one number for this substitution hides where it is safe and where it is not: the inner cluster is exactly where a resolved scene matters most, and it is also where the largest excess in this programme's cluster results lives.

**The directional pair is the sharpest result here.** Two laws with the same functional form and the same amplitude, differing only in where the preferred axis comes from:

- **directional, SOURCE axis** — 120.0% erased by azimuthal averaging (signal lost: yes; accuracy breached: yes).
- **directional, EXTERNAL axis** — -29.8% erased by azimuthal averaging (signal lost: no; accuracy breached: yes).

Azimuthal averaging destroys the source's own axis and leaves an externally imposed one untouched — indeed it *amplifies* the external one, by removing the competing source quadrupole. That is exactly the distinction GATE 1 of the existing pre-data compiler turns on: a response whose axis is created by the local source is degenerate with source ellipticity, while one fixed by an independently measured external direction is not. The external-axis case is still refused, but on **accuracy** grounds rather than signal loss, and the gate reports which.

**BUG 4 was conceptual, not a coding error, and it is the most important one in this lane.** The first version measured every law against the shell-averaged radial acceleration and duly reported that azimuthal averaging barely touched a directional law. That verdict was an artefact of the *observable*: a traceless directional term integrates to zero over a sphere, so the shell average had already erased the direction before the source averaging got a chance to. **An erasure test is meaningless unless the observable can still see the thing being erased.** The gate now carries three observables — shell mean, P₂ quadrupole, and sightline dispersion — and each erasure mode is tested against whichever retains the relevant structure.

**BUGS 5 and 6** are the same lesson twice more. The path law normalised its column by the mean over the probe shell, which made its correction have zero shell mean *by construction* — the law was built so the observable could not see it, and the gate then reported no erasure. And a spherically averaged scene represented by a finite set of shell directions is not smooth: measured through that representation, radial averaging appeared to **amplify** a path law by a factor of twelve. Fixing it once (one ray, broadcast) was not enough — it removed the scatter within a probe lattice but not between the rotated lattices the observable averages over.

## 4. Feeding the admissibility compiler

The charter says the metadata *"is what allows the admissibility compiler to prune candidate laws before data fitting"*, so `bridge.py` turns a candidate's list of consumed quantities into a verdict per gate using metadata and the availability matrix alone — no data file is opened and no fit is performed.

| Gate | Question |
|---|---|
| S1 units | can every nonlinear argument be made dimensionless? |
| S2 gauge | does it read a gauge-fixed potential? |
| S3 frame | does it read a quantity defined only in one named frame? |
| S4 coarse | does it read something that will not survive averaging? |
| S5 causal | is every input on the past light cone? |
| S6 identifiable | free latent field, or theory-contaminated product? |
| S7 rank | does the read set collapse under an exact identity? |
| S8 available | is every input actually observed, on the same cluster? |

S8 is new and is only possible once a scene layer exists. It turns the charter's "non-identifiable on the available data ... requires a different experiment" into a computable statement.

Screening `13` representative candidates against `67` indexed quantities:

| Candidate | Taxonomy | Decisive gate |
|---|---|---|
| `newton` | admissible | `--` |
| `rar_qumond` | admissible | `--` |
| `rar_with_radius` | admissible_but_redundant | `S7_rank` |
| `potential_depth` | convention_dependent | `S2_gauge` |
| `external_axis_tensor` | admissible | `--` |
| `source_axis_tensor` | admissible | `--` |
| `well_network` | mathematically_inconsistent | `S4_coarse` |
| `path_law` | admissible | `--` |
| `memory_law` | non_identifiable | `S6_identifiable` |
| `matter_light_slip` | non_identifiable | `S2_gauge` |
| `turbulent_pressure_source` | mathematically_inconsistent | `S3_frame` |
| `reads_kappa_as_data` | theory_contaminated | `S6_identifiable` |
| `raw_temperature_scale` | mathematically_inconsistent | `S1_units` |

| Taxonomy | n |
|---|---|
| admissible | 5 |
| admissible_but_redundant | 1 |
| convention_dependent | 1 |
| mathematically_inconsistent | 3 |
| non_identifiable | 2 |
| theory_contaminated | 1 |

Three real defects were caught without opening a file: a nonlinear function applied to a dimensionful temperature (S1), a turbulent velocity that is defined only in one named frame (S3), and a well count that changes when a deblender splits one galaxy into two (S4).

**BUG 7.** The first version of S6 failed a candidate whenever any input was not *directly observed* — which flagged Newtonian gravity itself, because `g_N` and `r_3d` are both constructed. That verdict was true and useless. Each quantity now carries a four-way class: `measured`, `constructible` (determined by the resolved scene through a declared procedure), `marginalisable` (integrated over by the scene ensemble — this class is the entire reason Stage 1 exists), and `non_identifiable` (a free latent field with no observational handle). Only the last fails the gate.

Two further branches are worth naming. A candidate reading a gauge-fixed potential is `convention_dependent`, carrying the measured 0.87 dex spread between defensible boundary rules against a 0.9 dex gate margin. And a candidate scored against a convergence map or an NFW-defined R500 is `theory_contaminated` — it is being tested against a product of the theory it is meant to replace. The charter forbids this explicitly; the bridge now makes it mechanical.

## 5. The gold-cluster availability matrix

The charter's Corpus E asks for clusters carrying `10` overlapping layers. **Corpus E is satisfied by no cluster.** The binding constraints are `L7_weak_lensing`, `L9_time_delays`.

| Cluster | imaging members | bcg | icl | member ifu | xray | sz | weak lensing | strong lensing | time delays | environment |
|---|---|---|---|---|---|---|---|---|---|---|
| Abell 2744 | raw | partial | raw | raw (LaTeX) | raw | raw (pixels) | **absent** | raw | **absent** | partial |
| MACS J0416.1-2403 | raw | partial | raw | raw (LaTeX) | **absent** | raw (pixels) | **absent** | raw | **absent** | raw |
| MACS J0717.5+3745 | partial | partial | raw | **absent** | raw | raw (pixels) | **absent** | raw | **absent** | partial |
| MACS J1149.5+2223 | raw | partial | raw | raw (LaTeX) | raw | raw (pixels) | **absent** | raw | raw (LaTeX) | partial |
| Abell S1063 | raw | partial | raw | raw (LaTeX) | raw | raw (pixels) | **absent** | raw | **absent** | partial |
| Abell 370 | partial | partial | raw | raw (pixels) | raw | raw (pixels) | raw | raw | **absent** | partial |
| Abell 2029 | partial | raw | **absent** | **absent** | raw | raw | **absent** | **absent** | **absent** | raw |

| Cluster | layers usable / 10 | raw & tabulated | confirmed absent |
|---|---|---|---|
| MACS J1149.5+2223 | 9 | 6 | 1 |
| Abell 370 | 9 | 4 | 1 |
| Abell 2744 | 8 | 5 | 2 |
| Abell S1063 | 8 | 5 | 2 |
| MACS J0416.1-2403 | 7 | 5 | 3 |
| MACS J0717.5+3745 | 7 | 3 | 3 |
| Abell 2029 | 5 | 4 | 5 |

### What is missing, and where

**Intracluster light** — absent for 1 of 7: Abell 2029.

**Internal member-galaxy IFU kinematics** — absent for 2 of 7: MACS J0717.5+3745, Abell 2029.

**Hot gas: X-ray density and temperature** — absent for 1 of 7: MACS J0416.1-2403.

**Wide-field weak lensing (per-source shapes)** — absent for 6 of 7: Abell 2744, MACS J0416.1-2403, MACS J0717.5+3745, MACS J1149.5+2223, Abell S1063, Abell 2029.

**Strong-lensing image families** — absent for 1 of 7: Abell 2029.

**Strong-lensing time delays** — absent for 6 of 7: Abell 2744, MACS J0416.1-2403, MACS J0717.5+3745, Abell S1063, Abell 370, Abell 2029.

### The structural findings

1. **Weak lensing is the hard ceiling.** A public *raw* shear catalogue exists for exactly one of the seven, Abell 370 (18,556 measurements to 6.2 Mpc). For the other six there is no per-source catalogue and no public binned shear profile either: those profiles exist only as figures, and what the papers tabulate is NFW masses, which presuppose a dark-matter halo. And Abell 370 is one of the two clusters *without* resolved Sérsic parameters for its members. **No target has both.**

2. **Time delays essentially do not exist at cluster scale.** The complete census of measured cluster-scale delays is SN Refsdal in MACS J1149, SN H0pe in PLCK G165.7+67.0, SN Encore/Requiem in MACS J0138−2155, and three cluster-lensed quasars. Exactly one target has one. It is the single strongest matter–light consistency constraint available, and there is one.

3. **Two whole layers are invisible to a catalogue search.** The 213 member-galaxy velocity dispersions and the SN Refsdal delays are both **raw and machine-readable — inside arXiv LaTeX source**. Granata et al. deposit only their Appendix B structural tables at CDS; the dispersions are Appendix C. A VizieR-only inventory records both layers as absent. This is a new failure mode: *a published data-availability statement can be narrower than the paper.*

4. **The IFU layer is not what the charter asked for.** Every Frontier Fields σ measurement is a **single aperture** value (1.5 arcsec, corrected to R_e/8) — one number per galaxy, not a resolved map. The only resolved member kinematic maps anywhere are SAMI's, which cover **no target cluster** and exactly one X-COP cluster (Abell 85). The charter's "predict the complete line-of-sight velocity distribution after projection, PSF convolution and aperture integration" cannot be tested on any target cluster today.

5. **The SZ and environment layers are anti-correlated across the sample.** The three southern primaries (A2744, MACS J0416, AS1063) have literally zero SDSS and zero DESI spectroscopy, while MACS J0717 and MACS J1149 — the two clusters entirely outside the ACT footprint (measured ACT DR6 declination maximum +20.796) and absent from SPT — have the best DESI environment data. Abell 2029 is the only target with an X-COP Compton-y *profile with full covariance* plus deep SDSS+DESI, and it is the one primary Bolocam omits.

6. **Only one target supports an external-tidal-axis reconstruction.** Abell 2029: 2,289 spectroscopic members in the cluster redshift slice out to 15.8 Mpc, a dedicated 8.7 Mpc survey, DESI DR1, and a published filament field through the position — four independent, mutually checkable layers. For A2744, MACS J0416 and AS1063 the widest spectroscopy reaches 4.1, 5.5 and ~5.2 Mpc, about one virial radius, and the only degree-scale product is photometric.

### Products that presuppose a gravity theory

The charter requires raw observations. These catalogued products are not raw, and the matrix says so at the point of use:

- **Abell 2744 / L6_sz** — RAW for the y map, ACT y0/fy0 and the SZ significances; DERIVED for Y5R500 and M_SZ
- **Abell 2744 / L10_environment** — RAW redshifts; photo-z DERIVED (SED template fit, no gravity assumption)
- **MACS J0416.1-2403 / L6_sz** — RAW y map and ACT y0; DERIVED Y5R500/M_SZ
- **Abell 370 / L10_environment** — RAW redshifts and photometry; photo-z DERIVED (SED fit)
- **Abell 2029 / L5_xray** — RAW counts; the hydrostatic mass products are DERIVED
- **Abell 2029 / L6_sz** — RAW for Y-PROF + covariance.  The pressure profile is DERIVED but only GEOMETRICALLY (Abel deprojection, spherical symmetry, a temperature to convert y to P) -- it does NOT presuppose dark matter.
- **Abell 2029 / L10_environment** — RAW redshifts.  The Sohn+2019b MEMBERSHIP column is DERIVED (caustic/phase-space assignment presumes a dynamical mass model) and the Tempel filament field is DERIVED (a Bisous marked point process on the redshift-space galaxy field).  Use the redshifts, not the labels.

More broadly, and recorded in the code rather than in prose: a Planck `Y5R500` integrates inside 5×R500 where R500 comes from an assumed GNFW pressure template and the Y–M relation; an `M_SZ` is hydrostatic/NFW-calibrated; the CATS Frontier Fields convergence maps assign a dark-matter clump to each cluster galaxy by construction, which makes them circular for any does-lensing-follow-light test. Against that, the X-COP `Y-PROF-COVMAT` product is a *measured* Compton-y radial profile with its full bin–bin covariance — its companion pressure profile is derived, but only **geometrically** (Abel deprojection, spherical symmetry, a temperature to convert y to P) and does not presuppose dark matter. It is the strongest single asset in the inventory.

## 6. Acquisition method and new traps

VizieR ASU (TAP returns 403). Every pull asserted the row count AND the column list, passed -out.all=1 explicitly, percent-encoded '+' as %2B, and checked both fuzzy-fallback detectors (#Name echo, CatalogsExamined). No cone-search null was trusted; all coordinate matching was done numerically against fully downloaded tables with a positive control.

Ten silent-failure modes were triggered live during acquisition. Six are new to this programme's record:

- The VizieR fuzzy-fallback trap is MIRROR-DEPENDENT. The same bad ID J/A%2BA/621/A41 serves 5.9 MB of an unrelated real catalogue at HTTP 200 from CDS but returns a clean 'Error=Table or Catalog not found' from vizier.cfa.harvard.edu. CfA is not merely faster, it is more truthful on bad IDs -- use it for existence tests.
- A SILENT ALIASED CATALOGUE that the CatalogsExamined detector cannot see: -source=J/A%2BA/590/A30 returns #Name: J/A+A/590/A31, a DIFFERENT PAPER, with CatalogsExamined=0 and no error. The #Name echo is the only detector that catches it.
- #Name: echoes the PARENT for a subtable request (J/A%2BA/594/A27/psz2 -> #Name: J/A+A/594/A27), so the echo test must be a PREFIX test; exact string equality false-positives.
- NOIRLab Data Lab TAP returns errors as HTTP 200 VOTABLE. A wrong column name (target_ra; the real one is mean_fiber_ra) produced <INFO name='QUERY_STATUS' value='ERROR'> inside a normal 200 and a line-based parse rendered it as an empty count for every cluster INCLUDING the positive control. Grep for QUERY_STATUS ERROR first.
- A subtable that looks like the main catalogue can hold a handful of rows: J/ApJS/247/25/table10 has 18 rows while the real SPT-ECS catalogue is /cand with 470. Enumerate subtables via METAtab before querying, and note that METAtab's `records` column sits after a free-text comment column so it must be requested explicitly.
- A cone search on a table with no sky columns returns 0 rows for EVERY position including the positive control: Tempel+2014 table2 carries only Cartesian x,y,z. Without a control this reads as 'no filaments anywhere in the sky'.
- Sexagesimal coordinate columns coerce silently: three catalogues serve RAJ2000 as 'h:m:s' strings, and a naive numeric cast gave RA=0.0000, Dec=-30.0000 and a fabricated 187 arcmin survey extent -- a plausible wrong number, not a crash.
- FOOTPRINT PRESENCE IS NOT COVERAGE. MACS J0717 has 233 SDSS spectra within 60 arcmin and ZERO at the cluster redshift; the SDSS main sample is far too shallow at z=0.545. A row count alone scores it as covered.
- A published data-availability statement can be narrower than the paper. Granata+2026 deposits only its Appendix B structural tables at CDS; the 213 velocity dispersions are Appendix C and exist only in the arXiv LaTeX source. A VizieR-only inventory records this layer as absent.
- Time-delay tables can contain no numerals at all: every cell of Kelly+2023's delay table is a \def macro resolved elsewhere in the manuscript source.

**Sealed data.** KiDS and the wide binaries were not loaded, looked at, or queried at any point in this lane.

## 7. What could NOT be established

1. **The gate's floor is a quadrature floor, not machine precision.** It sits at `2.65e-04` because the field on a probe shell is near-singular where a galaxy passes close to it. Every erasure number here is at least an order of magnitude above it, but a commutator below ~0.1% cannot be resolved by this implementation. A multipole-expansion or adaptive-quadrature observable would be needed.

2. **The commutation gate has been run on synthetic scenes only.** That is deliberate — the standing constraint forbids computing any gravity-relevant statistic on a real cluster while a confirmation set is being sealed — but it means the erasure fractions are properties of a representative synthetic cluster, not measurements of A2744. The synthetic scene is sized to the charter's A2029-like experiment so the orders of magnitude are comparable, and no stronger claim is made.

3. **The path and memory laws are illustrative.** They are built to exercise the two remaining charter erasure modes with the minimum structure that makes them non-trivial. The measured 100% erasure is a true statement about *those* laws under *those* operations; a different path or memory law could behave differently, and the gate must be re-run per candidate. It is a measurement device, not a theorem.

4. **The ensemble marginalises depth; it does not recover it.** The velocity term narrows the depth posterior by only 4.6%. Nothing in this lane makes line-of-sight depth measurable, and no analysis downstream should be designed as though a scene ensemble had solved that problem.

5. **Transverse velocities are sampled from a prior with no data constraint at all.** Proper motions at z ≈ 0.3 are far below any current astrometric capability, so `v_x` and `v_y` carry a prior and nothing else. A candidate law that depends on the full velocity vector is not testable on cluster members, and S6 marks those inputs `marginalisable` rather than `measured` so the fact is not lost.

6. **Filament and cosmic-web catalogues were only partly surveyed.** Tempel+2014 was checked positionally for all targets; a systematic sweep of DisPerSE-class catalogues was not completed. Note in advance that most such catalogues are derived under an assumed cosmology and bias model, so they presuppose structure formation in a dark-matter universe and would need the `theory_contaminated` flag.

7. **One inventory row has an expiry date.** BUFFALO's six-cluster release is announced but not yet on the HLSP. If it lands with per-source shapes, the weak-lensing row changes from one cluster to six and the binding constraint on Corpus E moves to time delays alone. Re-check it.

8. **Whether a member IFU aperture dispersion can stand in for a resolved map has not been tested.** It is an *average* already, so by this lane's own governing rule it should pass the commutation gate before being used that way. Building that test needs a resolved kinematic model of a member galaxy, which is Stage 2 work.

## 8. Files

- `metadata.py` — the 17-item parameter metadata contract, enforced
- `registry.py` — the populated ontology
- `schema.py` — nodes, edges, fields, SceneGraph, realisations
- `ensemble.py` — Job 2 — the probabilistic scene sampler
- `commutation.py` — Job 3 — the averaging-commutation gate
- `bridge.py` — the pre-data prescreen the compiler consumes
- `inventory.py` — Job 4 — the availability matrix
- `run_scene.py` — driver, writes scene_results.json
- `test_scene.py` — 38 tests, writes test_results.json
- `write_report.py` — renders this file and SCHEMA.md

Wall time for a full run: `106` s.
