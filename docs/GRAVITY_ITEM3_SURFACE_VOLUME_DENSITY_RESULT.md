# Gravity roadmap Item 3: surface versus volume density result

## Decision

**`INCONCLUSIVE_ITEM3_SURFACE_VOLUME_DENSITY_QUALITY_GATE`**

The first Item 3 attempt does not pass.  Its qualifying surface/volume transition variables
fail both the retrospective galaxy/cluster diagnostic and a fresh direct-dynamics group
test.  Representation failures independently make the attempt ineligible for confirmation.

The immutable receipt is
`runs/gravity/roadmap/item-03-surface-volume-density-v1.json`:

- file SHA-256: `5742559f50abb7939ef68f909870c9d8c66e0d9788d34aa7f7e1b0696d762bf4`
- content SHA-256: `a0dff681242e6264f95433833d19dede6394af5014f73165695cfafe1c9addd9`

## First-principles variable

For one target-blind baryonic profile, the attempt defined a spherical-equivalent enclosed
mass for every population:

`M_eq(r) = g_bar(r) r^2 / G`

It then formed two dimensionless Poisson-source diagnostics at the same universal
acceleration `g_dagger=1.2e-10 m/s^2`:

`u_surface = pi G Sigma_bar / g_dagger = g_bar / g_dagger`

`u_volume = 4 pi G r rho_shell / (3 g_dagger) = D_M g_bar / (3 g_dagger)`

where `D_M=d log(M_eq)/d log(r)`.  A smooth transition kernel
`w(u)=u/(1+u)^2` measures where each source crosses unity, its logarithmic radial width,
their overlap, and their integrated-area asymmetry.

The simple ratio `u_surface/u_volume=3/D_M` is exactly an Item 1 mass-dimension rewrite.  It
was frozen as a nonqualifying control and cannot be relabeled as a new variable.  Only the
dual transition location, width, overlap, and area structure was allowed to qualify.  This
is a potentially new synthesis of known Poisson and threshold ideas; historical novelty is
not established.

## Frozen data boundary

The derivation, eight cross-scale model families, ridge penalties, five whole-object folds,
fresh sample, density features, responses, permutation count, quality rules, and
confirmation prohibition were committed at
`ea75fbb44225811d676444251e4f85bbc064873b` before Item 3 features or fresh member rows
were opened.

The fresh AXES sample excludes all 270 groups used in any Item 2 role.  From 253 remaining
eligible groups, salted target-blind selection froze:

| Cleaned members | Exploration | Sealed confirmation |
|---|---:|---:|
| 10–14 | 40 | 20 |
| 15–24 | 40 | 20 |
| 25+ | 40 | 20 |
| **Total** | **120** | **60** |

The 120 exploration queries returned 3,096 member rows and 549,153 bytes.  They did not
query published group dispersion, `R200`, mass, X-ray, halo, or lensing columns.  No Item 2
group target was reused and no Item 3 confirmation group was queried.

## Representation result

The frozen quality rules fail:

- two SPARC galaxies and nine CLASH clusters have profiles that do not satisfy the exact
  five-point, positive, strictly ordered density-profile contract;
- 33 fresh groups have non-strict luminosity-quantile radii because one member carries
  enough light to make adjacent 25/50/75/90 percent radii coincide;
- no failed object was replaced and no threshold was relaxed.

The diagnostic therefore retains 137 galaxies, 11 clusters, and 87 fresh groups.  The fresh
groups still cover all three richness strata (18, 29, and 40 objects), but the attempt cannot
advance to confirmation.

## Retrospective cross-scale result

This lane reuses the already-seen Item 1 per-object coefficient labels and is development,
not independent confirmation.  The nested selector chooses the nonqualifying binary
galaxy/cluster proxy in all five folds.

| Model | Overall coefficient `R^2` | Galaxy `R^2` | Cluster `R^2` |
|---|---:|---:|---:|
| Binary population proxy / nested selection | 0.641 | -0.015 | -0.251 |
| Surface amplitude | -0.009 | -0.177 | -16.939 |
| Item 1 dimension control | 0.034 | -0.399 | -13.822 |
| Dual transition location | 0.020 | -0.344 | -14.686 |
| Dual transition shape | 0.083 | -0.432 | -12.193 |
| All surface/volume transitions | 0.039 | -0.418 | -13.524 |

The global proxy score comes from population separation.  It is worse than a held-out mean
inside both populations and has negative rank correlation in both.  The conclusion also
holds in the shared surface-amplitude interval containing 24 galaxies and all 11 valid
clusters.

## Fresh direct-dynamics result

The primary target is the gapper velocity dispersion recomputed from member redshifts.  It
is not a catalog mass, halo fit, lensing mass, or response-derived input feature.

| Model | Overall held-out `R^2` | MSE |
|---|---:|---:|
| Strong luminosity/size/richness/redshift/environment baseline | **0.457** | **0.00627** |
| Baseline plus all seven frozen density features | 0.398 | 0.00696 |
| Nested selection | 0.457 | 0.00627 |

Every fold selects the nonqualifying baseline.  Density augmentation is negative in every
richness bin and has `R^2=-0.204` in the lowest-richness bin.  A 199-run richness-stratified
permutation test gives `p=0.795`: the observed density increment is worse than baseline and
more null permutations outperform it than not.

The density-augmented family narrowly improves the MAD response overall, but it loses badly
on the two luminosity-normalized virial responses and does not repair the negative
low-richness result.  Only the untouched-confirmation gate passes; the other eleven gates
fail.

## Meaning and next test

This attempt rules out the frozen dual-transition summaries as a sufficient cause on these
representations.  It also demonstrates why formula novelty and predictive score must be
separate: the dual-transition construction is structurally different from Item 1, yet real
data reject it, while the known population proxy looks globally successful without being a
within-population cause.

Item 3 remains open because the discrete member-light quantiles are a weak volume-density
proxy and the cross-scale cluster profiles are too short for the frozen kernel.  A second
attempt must use genuinely smooth, directly measured density profiles—preferably resolved
stellar-plus-gas galaxy surface density and X-COP gas volume density—together with direct
dynamics or forward X-ray/SZ observables.  It must freeze its mapping before targets, retain
the current failed transition family as an equivalence region, and keep all 60 group
confirmation targets sealed.

## Replay

```powershell
python -m sigma_theory_compiler.gravity_item3_surface_volume_density_experiment --root . --check
python -m pytest tests/test_gravity_item3_surface_volume_density.py -q
```

The run used eight cross-scale model families, three fresh-group model families, five
whole-object folds, 199 stratified permutations, zero paid model calls, zero direct-lensing
likelihood evaluations, and zero confirmation accesses.
