# Same-card galaxy transfer: limited comparisons, no validated universal law

All 54 fixed local/cluster cards were evaluated on the existing NGC3198 source,
with the same global constants, eleven source variants, three distances and three
inclinations. The two comparators were recomputed on the same new source.
All 56 configurations meet the registered target-force numerical gates.
Across all 54 action cards, none lowers the inclination-covariance loss relative
to RAR in any of the 99 declared scenarios.

Of the 13 cards that were within the historical local screens and improved the
cluster-pressure comparator, **three improve the nominal galaxy random-error
score; none improves the nominal inclination-covariance score**. The three are
shape `m=2`, `a0=1.2e-10 m/s²`, at lengths .1, 1 and 10 pc. The sampled length
effects are much smaller than the field-resolution uncertainty. These are
conditional development comparisons, not three validated gravity laws.

The source expansion still has substantial angular ringing. That limitation,
the source/response uncertainties, and the absence of direct outer-star and
lensing predictions prevent physical validation or a family-level exclusion.
The goal remains active.

## Method and controlled changes

The new implementation derives the potential and its first three radial
derivatives from exact Green integrals of a single interpolated density source.
It includes both tangential Hessian directions and the density-gradient terms
of the action variation. The accompanying derivative note gives the equations
and independent controls. No old C1 potential spline is differentiated through
third order.

A positive central continuation matches the first measured surface density and
its radial derivative while giving zero radial derivative on the symmetry axis.
Every measured source knot is unchanged. The largest component total-mass change
is 0.0513%, below the fixed 1% source gate. This central continuation is an
unmeasured source assumption, not new data.

The physical length remains fixed under every distance scenario. A homology by
factor `d` requires evaluating `ell/d` in nominal coordinates, followed by
`v=sqrt(d*r_nominal*g)`. Scaling velocities alone would have changed the
nonzero-length theory. All 5,346 candidate/scenario length records were checked.

The same 24 previously exposed gas-traced velocity points and 97 numerical
control radii are retained. The calculation makes 4,032 model/distance/source-grid
force profiles across 24 field runs, then 5,544 model/scenario velocity records.
No new raw data or reserved systems were opened. No winning nuisance setting
or new action card was selected.

## Nominal results at a fixed universal length of 1 pc

Lower loss is better within each column. The random-error and covariance columns
are descriptive diagnostics, not calibrated discovery significances. The
inclination treatment is a shared rank-one covariance and remains an uncertain
response model.

| Shape | a₀ (m/s²) | Random-error loss | Inclination-covariance loss | Velocity RMS (km/s) |
|---|---:|---:|---:|---:|
| 0.5 | 5e-11 | 540.233 | 19.393 | 43.466 |
| 0.5 | 1.2e-10 | 199.736 | 24.147 | 27.031 |
| 0.5 | 2e-10 | 53.955 | 27.090 | 15.329 |
| 1 | 5e-11 | 404.105 | 30.472 | 37.670 |
| 1 | 1.2e-10 | 47.086 | 18.168 | 14.847 |
| 1 | 2e-10 | 24.492 | 15.103 | 10.470 |
| 2 | 5e-11 | 332.908 | 39.291 | 34.818 |
| 2 | 1.2e-10 | 10.320 | 7.415 | 8.226 |
| 2 | 2e-10 | 35.169 | 9.233 | 11.418 |
| RAR comparator | 1.2e-10 | 17.077 | 3.973 | 7.575 |
| Newtonian baryons | — | 836.703 | 26.568 | 54.469 |

For the illustrative `m=2, a0=1.2e-10, ell=1 pc` card, the random-error loss
improves on RAR in 66/99 scenarios, the 5 km/s-floor diagnostic in 50/99, and
the inclination-covariance diagnostic in 0/99. Its nominal random-error comparison
retains its sign when dropping the most influential radial point or trimming
both tails. This is one galaxy; removing a radial point is not independent
object replication.

Across the nominal 54-card grid, the largest velocity change from the same
shape/acceleration card at zero length is only 0.00179 km/s, or 2.48e-5
fractionally. That change is below numerical force resolution. The sampled
lengths can suppress the local quadrupole while scarcely changing galaxy or
cluster predictions at this resolution. They have not demonstrated a correction
to the galaxy curve's shape.

## What passed numerically, and what did not become accurate

Maximum target-force changes across every card, source variant and distance are:

| Check | Observed maximum | Registered limit |
|---|---:|---:|
| Coarse/fine field resolution | 0.582% | 2% |
| Map reconstruction resolution | 1.077% | 3% |
| Inner/outer boundary change | 0.00156% | 0.5% |

All registered inward branches and reflection checks pass. No failed point was
removed to obtain numerical admission. The 201-test focused suite passes,
including eleven new derivative/source tests. Forty-three relevant controls ran
inside the campaign before any new gravity prediction.

**Target-force convergence does not establish pointwise source accuracy.** At the
fine nominal grid, the negative portion of the projected density integrates to
20.1% of the positive physical source's grid integral, and its relative L1 density
error is 67.1%. With half the assumed disk heights, those diagnostics are 35.4%
and 109.4%. They improve with resolution but remain large. These are artifacts
of the finite spectral representation; the declared input density is positive.
They are neither evidence for negative matter nor observational residuals.
The full derivative-dependent source has therefore not been independently
validated, even though the registered target-force comparisons converge.

Other unresolved issues include correlated gas-source/velocity errors, the
conditional inclination treatment, warps and noncircular motion, unobserved
outer stellar/CO material, and environmental effects omitted by the isolated
boundary. Raw comparative losses are not physical falsifications. Quality-verified
and uncertainty-resolved counterexample counts remain zero.

## Reproducible evidence

The independent replay verified 35 input snapshots, all 5,544 velocity records
and radial-influence diagnostics, and 2,184 numerical comparisons. A
Sherman-Morrison calculation independently reproduced the covariance scores to
3.12e-14 scaled difference. Velocity reconstruction agreed exactly. This validates
the recorded calculation and scoring, not its unresolved source approximation.

The plot was visually checked. Exact executed code, configuration and source
bytes are preserved under `work/gravity-first-principles/length-ngc3198-001`.

| Evidence | SHA-256 of result.json |
|---|---|
| length-ngc3198-001 | 430f1f69a50e29079a4ad09a8fdaef003798336f2b10ca64e8ed369c23ca6009 |
| length-ngc3198-verification-001 | 9b49628026c56fcc44b10af9ca04c04cf934d0dda8f65fd298a08fd5dd3f239c |

Reproduction commands use new output directories, with `PYTHONPATH=src` and one
BLAS thread:

```text
python scripts/run_gravity_length_ngc3198.py --output <new-run>
python scripts/report_gravity_length_ngc3198.py --run <new-run> --verification <new-verification> --outputs <new-output-directory>
```

## Next discriminating work

Resolve and independently assess the angular source and derivative errors before
extending the length toward the disk's physical thickness. Larger lengths may
make the mechanism relevant to galaxy structure, but they could also amplify
the present source artifacts. Any extension requires a fresh registered global
card grid and repeat local, cluster and galaxy calculations without per-regime
retuning. The present finite 0–10 pc scan is not an exclusion of all lengths.

The broader queue retains source-conserving coherence alternatives, thermodynamic
source-model repair, and covariant matter/light coupling. A first-principles
origin for the trial kernel and constants, dynamics/stability, direct stellar
and lensing predictions, and independent confirmation remain outstanding.
