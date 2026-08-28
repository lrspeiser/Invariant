# Gravity roadmap Item 3 attempt 2: smooth density profiles

## Decision

**`REJECT_ITEM3_SMOOTH_DENSITY_CROSSOVER_EXPLORATION`**

The smooth-profile experiment rejects the frozen local surface/volume-density crossover
family.  It is a much stronger negative than attempt 1 because the representation quality
passes: ten galaxies and all eight exploration clusters supply usable, directly measured
radial profiles.  The admissible density formulas nevertheless lose to ordinary controls,
fail within each population, and fail the object-level permutation test.

The immutable receipt is
`runs/gravity/roadmap/item-03-smooth-density-profiles-v2.json`.

## What was tested

For a radial baryonic density `q_d(r)` measured on support dimension `d`—a disk surface
density for galaxies (`d=2`) or a cluster gas volume density (`d=3`)—the frozen derivation
first measured the target-blind profile scale

`H = |d log(q_d)/dr|^-1`.

It then constructed the same physical pair for both populations:

`Sigma_eff = q_d (2H)^(d-2)`

`rho_eff = q_d / (2H)^(3-d)`

`u_surface = pi G Sigma_eff / g_dagger`

`u_volume = 4 pi G r rho_eff / (3 g_dagger)`.

The log contrast is exactly

`log10(u_surface/u_volume) = log10(3H/(2r))`.

That contrast is therefore a profile-scale rewrite and cannot count as novelty.  Only
joint amplitude, contrast, acceleration, and transition-kernel interactions were allowed
to qualify.  This is a potentially new synthesis of known ingredients, not an established
historically new law.

Nine families were frozen: a constant, a baryonic-acceleration polynomial, surface-only,
volume-only, scale-contrast, a forbidden population-label control, and three qualifying
dual-density/crossover families.  Selection and penalty tuning occurred only inside
training objects.

## Real data and target boundary

The galaxy lane combines official Leroy/THINGS radial H I, H2, and stellar surface-density
profiles with SPARC observed rotation curves.  It is retrospective development because
the repository used the SPARC targets earlier; it is not independent confirmation.

The cluster lane uses released X-COP XMM electron-density profiles as baryonic inputs and
released Planck SZ pressure profiles to form the direct hydrostatic target

`g_dyn = -(mu m_p n_e)^-1 dP_e/dr`.

Released hydrostatic masses, total masses, gas fractions, NFW fits, and lensing masses are
forbidden and were not read as targets or features.  The primary cluster calculation uses
gas baryons for all eight objects; a frozen robustness test adds released stellar-mass
profiles for the five clusters where they exist.

The eight exploration clusters and four reserved clusters were selected by salted hash
inside stellar-profile-available/unavailable strata before profile access.  A2029, A3158,
A644, and RXC1825 remain unopened.

## Representation result

The frozen quality contract passes:

| Population | Admitted objects | Primary radial points |
|---|---:|---:|
| Galaxies | 10 | 313 |
| Clusters | 8 | 62 |
| **Total** | **18** | **375** |

DDO 154 is retained as the sole failure: its independent density profile has seven points
and overlaps only two SPARC rotation points.  It was not replaced.  Scale-length clipping
is below five percent for every admitted object.

Blank molecular-gas entries in the Leroy table are treated as zero rather than inferred
from the dynamics.  This is a real limitation, particularly for the galaxy lane, and is
one reason the rejection is limited to the tested local density family rather than every
possible baryonic-density theory.

## Held-out result

Whole objects—not radial rows—define five folds.  Galaxies and clusters receive equal
total loss weight, each object receives equal weight within its population, and each
object's radial points divide its weight equally.

The unrestricted selector chooses the forbidden galaxy/cluster label in all five folds.
That diagnostic achieves global `R^2=0.814`, but only `R^2=0.283` within galaxies and
`R^2=0.028` within clusters.  Its apparent global strength is mostly population
separation, not a density cause.

The selector restricted to admissible density formulas chooses the reduced crossover in
four folds and dual-density amplitude in one.  Its performance is:

| Admissible density selector | Overall | Galaxies | Clusters |
|---|---:|---:|---:|
| Held-out `R^2` | 0.352 | **-3.178** | **-0.656** |
| RMSE, dex | 0.330 | 0.396 | 0.246 |

The positive global score is another population-separation artifact.  Inside each
population the density selector is substantially worse than predicting that population's
held-out mean.  It also loses to the baryonic-acceleration baseline in both populations:

| Qualifying minus acceleration-baseline `R^2` | Value |
|---|---:|
| Overall | -0.089 |
| Galaxies | -0.402 |
| Clusters | -0.403 |

An object-stratified 499-permutation test gives `p=0.696`; the observed incremental MSE
change is negative.  Adding measured stellar mass for the five available clusters also
leaves the qualifying selector worse than the acceleration baseline (`Delta R^2=-0.028`).

Only two of eleven gates pass: representation quality and untouched confirmation.  Every
predictive, causal, robustness, and label-independence gate fails.

## Meaning

This result closes the tested **local** surface/volume-density route.  Direct smooth
profiles do not rescue the coarse attempt-1 result, and the label control again shows how
cross-scale separation can look impressive without identifying a common cause.

The result does not reject nonlocal interior/exterior kernels, baryonic boundaries,
environmental density, or density-dependent field equations.  Those are structurally
different roadmap items and must not be silently retuned under Item 3.  It also does not
establish GR, dark matter, or any alternative theory.

## Replay

```powershell
python -m sigma_theory_compiler.gravity_item3_smooth_density_profiles --root . build-exploration-sources --raw-dir work/item3-density-v2-raw
python -m sigma_theory_compiler.gravity_item3_smooth_density_experiment --root . --check
python -m pytest tests/test_gravity_item3_smooth_density_profiles.py tests/test_gravity_item3_smooth_density_experiment.py -q
```

The run evaluated nine formula families, five outer folds, 499 object-level permutations,
189,375 ridge fits, zero paid model calls, zero direct-lensing likelihoods, and zero
confirmation accesses.
