# Exhaustive screened-nonlocal permutation result v0

## Bottom line

The first finite first-principles-inspired grammar did **not** find a formula that
bridges galaxies and galaxy clusters. That is a useful failure-space result, not a
gravity discovery.

Invariant represented all 1,048,576 cells in the frozen `4^10` grammar. Exact
dynamics-equivalent cells were collapsed only after they were counted, leaving
196,612 distinct dynamics behaviors. Analytic limits admitted 185,140 behaviors for
GPU scoring. Every admitted behavior was evaluated on the already-exposed SPARC,
LITTLE THINGS, and X-COP responses.

No behavior matched all three incumbent results. No behavior kept both galaxy scores
within 5% of the RAR while remaining within 20% of the existing X-COP result.

This means the simple tested mechanism--one positive, screened, multiplicative
nonlocal response on top of Newtonian gravity, the empirical RAR, simple MOND, or
standard MOND--does not solve the cross-scale problem on these data and parameter
levels.

## What "every permutation" means

The frozen grammar exhausts these ten four-valued factors:

| Factor | Values tested |
|---|---|
| Base dynamics | Newton/GR, empirical RAR, simple MOND, standard MOND |
| Spatial kernel | symmetric exponential, interior exponential, exterior exponential, symmetric Gaussian |
| Occupancy threshold `y0` | 0.03, 0.1, 0.3, 1.0 |
| Log-radius range | 0.125, 0.25, 0.5, 1.0 |
| Coupling `alpha` | 0, 0.5, 1.0, 1.5 |
| High-acceleration screen | 1, 10, 100, 1000 times `a0` |
| Screen power | 1, 2, 3, 4 |
| Compactness threshold | `1e-8`, `1e-7`, `1e-6`, `1e-5` |
| Environment shape | two compactness and two boundary-sensitive shapes |
| Lensing response | scalar control, linear transition, quadratic transition, metric control |

This is every cell of a declared finite grammar, not every imaginable mathematical
theory. The four lensing branches have identical rotation-curve and hydrostatic
predictions, so all were counted but none was selected without direct lensing data.
When `alpha=0`, all response factors disappear exactly; those duplicate controls were
also counted before collapsing.

## Formula evaluated

For each radial baryonic profile the search constructed

    y = g_bar/a0

    Q = y/(y + y0)

    X = K_kernel,ell[Q]

    C = g_bar r/c^2

    b = |d^2 ln M_b(<r)/d(ln r)^2|

    zeta = (C/C_star)^p (1 + b)^q

    T = [1 + (y/y_screen)^n]^-1 zeta^m/(1 + zeta^m)

    g = g_base(g_bar) exp(2 alpha T X).

The high-acceleration screen, positive effective permittivity, bounded cluster probe,
and minimum nonlocal materiality were checked before empirical scoring. The weak-field
action that motivates this form is in `SCREENED_COMPLETION_ATTEMPT_V0.md`; it remains a
scaffold, not a complete covariant theory.

## Exhaustive accounting

| Stage | Distinct dynamics behaviors | Raw grammar cells represented |
|---|---:|---:|
| Frozen grammar | 196,612 | 1,048,576 |
| Positive effective permittivity | 196,612 | 1,048,576 |
| Local high-acceleration limit | 194,052 | 1,038,336 |
| Bounded cluster probe | 187,460 | 1,011,968 |
| All analytic gates | 185,140 | 1,002,688 |
| Nonzero qualifying theories | 185,136 | 740,544 |

The four additional admitted behaviors are the zero-coupling base-law controls.

Empirical scoring used 139 SPARC galaxies with 2,720 radial rows, 11 LITTLE THINGS
galaxies with 199 rows, and 12 already-exposed X-COP clusters. It made zero new target
queries, opened zero sealed confirmation rows, used zero direct lensing rows, and made
zero paid model calls. Float64 GPU scoring on the RTX 5090 took about 1.35 measured
seconds.

## Results

Lower score is better. A ratio of 1.0 matches the incumbent for that population.

| Result | Count |
|---|---:|
| At or better than the SPARC RAR | 6,118 |
| At or better than the LITTLE THINGS RAR | 31,171 |
| At or better than the Item 59 X-COP result | 1,777 |
| At or better on both galaxy populations | 406 |
| At or better on LITTLE THINGS and X-COP | 133 |
| At or better on SPARC and X-COP | 0 |
| At or better on all three | 0 |

The absence of any SPARC-plus-X-COP survivor is the central result. This is not one
counterexample killing a theory. It is a broad trade-off across 139 galaxies and the
cluster development profiles within this entire finite representation.

### Least-bad compromise

The minimax rule selected `SNB-67868` as a diagnostic representative:

- empirical RAR base;
- exterior exponential kernel with log-radius range 1.0;
- `y0=0.1`, `alpha=1.0`;
- high-acceleration screen at `y=1` with power 2;
- compactness threshold `C_star=1e-6`;
- balanced compactness-boundary activation.

Its ratios were 1.696 on SPARC, 1.001 on LITTLE THINGS, and 1.656 on X-COP. In plain
language, it preserved the dwarf-galaxy benchmark but was about 70% worse than the
SPARC benchmark and 66% worse than the existing cluster result. It is retained for
diagnosis, not promoted to a fresh-data gate.

### The trade-off is structural

The best SPARC behavior also improved LITTLE THINGS, but was 18.85 times worse than
the X-COP incumbent. The best X-COP behavior improved the cluster score by about 21%,
but was 17.07 times worse than the SPARC RAR. The best LITTLE THINGS behavior improved
that score by about 14%, but was 21.61 times worse on X-COP.

The grammar can specialize to galaxies or clusters. It cannot create one smooth
universal bridge between them.

## What was learned about the proposed first principles

The useful idea that survives is the **architecture**, not this formula:

1. a local auxiliary field can generate a finite-range nonlocal kernel;
2. a dimensionless transition can restore local GR analytically;
3. dynamics and lensing can be separated through two metric potentials without giving
   photons and heavy elements different gravitational charges;
4. exhaustive finite grammars can map a large failure region quickly and reproducibly.

What failed is the assumption that a single positive multiplicative factor
`exp(2 alpha T X)` can preserve an RAR-like disk law while recovering the X-COP
profile advantage using only acceleration, compactness, and radial boundary curvature.

The result does not rule out MOND, refracted gravity, nonlocal gravity, TeVeS-like
theories, dark matter, or the broader auxiliary-field architecture. It only rejects
promotion of the tested grid and functional form. The data are already exposed, so
even a success would have been hypothesis generation rather than independent
confirmation.

## Next structurally different grammar

Do not merely make this grid denser. The next search should vary mechanisms that can
break the observed SPARC-X-COP trade-off:

1. an additive auxiliary-field channel instead of only multiplicative amplification;
2. a disk-preserving form such as `g=g_RAR + T_cluster Delta_g`, with `T_cluster`
   derived from continuous baryonic predictors and no object label;
3. thermodynamic state `P/(rho c^2)`, entropy slope, and pressure support as possible
   source variables;
4. geometry tensors that distinguish disks, groups, and clusters continuously rather
   than through a class switch;
5. signed or phase-like response, subject to positive-energy and stability gates;
6. a second auxiliary field so the local MOND/RAR channel and finite-range cluster
   channel need not be the same operator;
7. a causal two-potential lensing sector, with all photon-response branches frozen
   before direct lensing access;
8. a fresh galaxy-group bridge set before spending a direct-lensing confirmation.

Each next grammar must again declare a finite Cartesian product, enumerate every raw
ordinal, collapse only mathematically exact observational equivalences, run analytic
gates first, and retain mismatches as failure-space evidence rather than allowing a
single object to prune an idea.

## Reproduction record

Scientific freeze commit:

    435869fa133d2a8f8f71dd91f8fd518fcad8d99b

Commands:

    python -m sigma_theory_compiler.screened_nonlocal_boundary_permutation preflight
    python -m sigma_theory_compiler.screened_nonlocal_boundary_permutation evaluate
    python -m sigma_theory_compiler.screened_nonlocal_boundary_permutation aggregate

Machine-readable receipts are in `runs/gravity/screened-nonlocal-boundary-v0/` and
`runs/gravity/screened-nonlocal-boundary-v0.json`.
