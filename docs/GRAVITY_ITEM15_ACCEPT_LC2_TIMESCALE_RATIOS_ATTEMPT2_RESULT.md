# Gravity roadmap Item 15 — ACCEPT/LC2 direct-cooling attempt 2

## Result

**Decision: `REJECT_ITEM15_ACCEPT_LC2_TIMESCALE_EXPLORATION`.**

The clean 18-cluster exploration found a favorable but non-significant timescale pattern.
Adding one nested-selected timescale cell to the frozen source-variable baseline reduced
held-out mean squared error by **9.09%** (`R2` from `0.141` to `0.219`). The gain was positive
in every preregistered cooling, temperature, redshift, and lensing-source slice and remained
slightly positive under fixed weak-lensing error weighting. It is not promoted: the full
selection procedure repeated inside 99 label permutations gives **`p=0.22`**, and that is the
only one of 15 frozen gates that fails.

This is a nonpromoted lead, not a formula discovery, a causal cooling-time effect, or evidence
against general relativity. The five confirmation clusters remain sealed.

## Why this is a materially new Item 15 lane

Attempt 1 used galaxy catalog proxies and had no direct hot-gas cooling time. This attempt uses
the published ACCEPT deprojected density, temperature, cooling function, and isochoric cooling
time profiles at 20, 50, and 100 kpc. It pairs those predictors, only after the sample freeze,
with individual literature weak-lensing `M500` values in LC2-single.

Sources:

- ACCEPT: Cavagnolo et al. 2009, [arXiv:0902.1802](https://arxiv.org/abs/0902.1802)
- LC2: Sereno 2015, [arXiv:1409.5435](https://arxiv.org/abs/1409.5435)
- LC2 VizieR catalog: `J/MNRAS/450/3665/single`

LC2 `M500` is a heterogeneous, GR/cosmology/model-standardized literature quantity. It is not
direct shear, image positions, magnification, or the direct-CLASH gate in roadmap Item 60.

## Response-blind sample and access boundary

The metadata-only ACCEPT x LC2 audit found 93 overlapping ACCEPT identities. All identities
used in earlier cluster tests were excluded. During source research, a Herbonnet et al. 2020
source archive was downloaded and six numeric response rows were accidentally displayed. The
remedy was deliberately broader than the known exposure: exclude **every physical system in
the paper's 100-cluster sample**. Forty-seven of those systems overlap the ACCEPT x LC2 universe.

After all predecessor and incident exclusions, 24 candidates remained. One,
`MACS_J2214.9-1359`, had no retrievable archived predictor profile and was excluded before the
sample freeze, with no replacement. The exact 23-object predictor universe was ranked by direct
`tcool(20 kpc)` and split into 12 short- and 11 long-cooling systems. A deterministic salted hash
then assigned:

- 18 exploration clusters: 9 short cooling and 9 long cooling;
- 5 sealed confirmation clusters: 3 short cooling and 2 long cooling;
- five outer folds with counts `4, 4, 4, 4, 2`.

All 18 exploration LC2 rows were retrieved and all 18 passed the frozen positive-mass,
positive-error, fractional-error, fold, and cooling-stratum quality gates. Confirmation response
queries, paid model calls, and post-response formula cells are all exactly zero.

## Frozen physical construction

For each radius `r` in `{20, 50, 100} kpc`, the response-blind pipeline uses the published
isochoric cooling time and integrates the deprojected electron density shells:

`Mgas(<r) = integral[4 pi r^2 mu_e m_p n_e(r) dr]`, with `mu_e=1.17`.

It constructs an intentionally incomplete baryon-only free-fall clock,

`tff,b(r) = sqrt[2 r^3 / (G Mgas(<r))]`,

and a sound-crossing clock,

`ts(r) = r / sqrt[gamma kT(r)/(mu m_p)]`, with `gamma=5/3` and `mu=0.61`.

The cosmic-age clock uses the frozen flat `H0=70`, `Omega_m=0.3` cosmology. Candidate programs
organize the dimensionless logs of `tcool/tff,b`, `tcool/ts`, `tcool/tcosmic`, and `ts/tff,b`
across radii.

The published ACCEPT `5/2` and `3/2` cooling columns differ by the expected constant `5/3` and
are counted as one observable equivalence class. Published ACCEPT hydrostatic masses never
enter a predictor. The free-fall diagnostic omits stars and the BCG and is not represented as a
complete physical collapse time.

## Formula search and novelty labels

Before `M500` access, PCG64 seed `1501520260828` fixed **262,144** program cells across 12
families. The bank contained:

- known/rewrite controls: direct one-radius cooling/free-fall, cooling/crossing, and
  cooling/cosmic transforms;
- known-family combinations: radial phase gradients, localized threshold shells, and
  log-periodic clocks;
- combinations: acoustic/collapse competition and hierarchy span;
- potentially new syntheses: clock entropy, cross-radius interference, resonance shells, and
  phase-locked cooling.

Those labels describe provenance, not adjudicated historical novelty. The raw cell count is not
a count of independent mathematical laws.

The strong fixed baseline contains all raw quantities used to create the clocks: ACCEPT entropy,
temperature, electron density, gas mass, direct cooling time, luminosity (`log10(1+Lbol)` so
published zero/missing entries are retained), redshift, radial gradients/concentrations, and
broad LC2 source-family nuisance indicators. Thus a candidate must add nonlinear organization,
not merely rediscover a raw input or a linear log ratio.

Nested five-fold selection ran on the RTX 5090. The search performed 94,371,840 candidate-cluster
score evaluations for the observed target and 9,437,184,000 including 99 complete selection-null
reruns. Runtime was 9.31 seconds; CPU/GPU component disagreement was at most `1.67e-15`.

## Numerical outcome

| Measurement | Frozen baseline | Selected full model | Outcome |
|---|---:|---:|---:|
| Unweighted held-out MSE | 0.20548 | 0.18680 | 9.09% lower |
| Held-out `R2` | 0.141 | 0.219 | positive increase |
| Error-weighted held-out MSE | 0.18376 | 0.18124 | 1.37% lower |
| Full-selection null | — | `p=0.22` | fails `p<=0.05` |

The target is `log10[M500/Mgas(<100 kpc)]`; the ratio has median 963 and range 80–7,014 because
the denominator is only gas inside 100 kpc, not a complete cluster baryon mass. The exploration
sample has median `M500=7.94e14 Msun`, median `tcool(20 kpc)=2.98 Gyr`, and median
`tcool(100 kpc)=9.09 Gyr`.

Four outer folds selected `cooling_threshold_shell`, labeled a known-family combination. One
selected the nonqualifying direct cooling/free-fall rewrite. The four shell cells use different
radii (20, 50, and 100 kpc), different core-entropy or redshift modulations, materially different
thresholds, and one reversed coefficient sign. That is family-level recurrence, not one stable
equation.

Every frozen two-way slice has a positive MSE gain, but the high-temperature, high-redshift,
long-cooling, and Applegate slices have much smaller gains. Only three exploration objects are
in the Applegate slice, reinforcing the small-sample boundary.

## What this does and does not say in plain language

The result says that arranging measured cooling and dynamical clocks in certain nonlinear ways
can modestly improve prediction of a model-dependent lensing-to-inner-gas discrepancy in these
18 clusters. It does **not** show that cluster age creates extra gravity, that old clusters contain
more baryons, or that gravity remembers cooling history. A random-label search with the same
formula-selection freedom produced an equal or larger apparent gain about 22% of the time.

The correct action is therefore to retain the direction as a possible clue while refusing to
spend the sealed confirmation set on it. A confirmation-worthy version would first need either
an independent field derivation that fixes the radius and functional form, or a materially larger
fresh direct-observable dataset.

## Replay

The authoritative artifacts are:

- `configs/gravity_item15_accept_lc2_timescale_ratios_v2.json`
- `runs/gravity/roadmap/item-15-accept-lc2-timescale-ratios-v2-source/`
- `runs/gravity/roadmap/item-15-accept-lc2-timescale-ratios-v2.json`

Replay checks:

```powershell
python -m pytest tests/test_gravity_item15_accept_lc2_timescale_ratios.py -q
python -m sigma_theory_compiler.gravity_item15_accept_lc2_timescale_ratios check --root .
```
