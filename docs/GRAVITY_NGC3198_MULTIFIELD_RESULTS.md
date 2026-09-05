# Observed-source multifield gravity test

Seventy-two of 81 frozen bounded-TRIMOND configurations now pass the declared
numerical checks. Every one has greater error than the RAR comparator across
all 99 matched source/geometry scenarios, under all three error treatments.
That is 7,128 scored card/scenario records from **one** development galaxy,
not 7,128 independent astronomical tests. Nine configurations with p=2 and
coupling 10 remain numerically unresolved and unscored. No universal gravity
formula has been established, and no entire formula family is rejected.

The initial 63-card admission was followed by a frozen numerical refinement
of the two source variants responsible for the nine coupling-6 exceptions.
All nine then passed: the maximum combined scalar/auxiliary discrepancy is
1.7756%, below the unchanged 2% limit. Historical failed comparisons
remain in the parent receipt; the refinement supersedes only the named
resolution checks for those nine cards. It changes neither source parameters
nor the selected response radii. The other nine cards still fail map and field
resolution requirements; they have no scored velocities or losses.

## Formula and physical interpretation

The tested action is F=Q(x)−|q−s p|²−w|p×q|², where p=∇ψ/a₀,
q=∇χ/a₀, x=p², s=λ/(1+x)^P and w=β/(1+x)². Q is the previously
registered saturated scalar action. Newtonian ψ is sourced by the full
continuous baryonic density, and the auxiliary equation is div(Aq)=div(sp),
with A=I+w(xI−ppᵀ). The physical potential includes all action derivatives.
The general tripotential framework is prior art; this experiment does not
claim its invention. [Milgrom (2023)](https://arxiv.org/abs/2305.19986)

All cards use a₀=5e−11 m/s², shapes 0.5, 1 and 2, and β=0, 0.5 or 2.
P=1 uses λ=0.25, 0.75, 1.5 or 3; P=2 uses λ=0.25, 0.75, 3, 6 or 10.
The constants are global across all scenarios. This grid was chosen within
the prior conditional Cassini-summary coupling bounds before these multifield
galaxy predictions. That summary screen is not a full Solar System pass.

At fixed source, the auxiliary physical force scales as λ². At the nominal
source's outermost numerical probe, 20 kpc, all six auxiliary kernels contribute
outward radial force: −0.309 to −0.567 (km/s)²/kpc per λ². Increasing coupling
therefore deepens the outer deficit there, although it improves some interior
speeds. This sign statement covers the declared nominal source and radius.
The total inward force remains positive for the admitted cards.

In exact spherical symmetry, q=sp and this auxiliary correction is zero.
Consequently the existing spherical cluster-pressure deficits are unchanged;
this is an analytic transfer, not a fresh cluster-data test. A nonspherical
cluster and lensing calculation remains outstanding.

## Fixed-card comparisons

The table and figure show β=2, P=2 at two declared couplings, chosen after
inspection for compact illustration. They are not newly fitted parameters or
promoted winners. The machine-readable summary preserves all 81 cards.
The empirical RAR comparator is described by
[McGaugh, Lelli and Schombert (2016)](https://arxiv.org/abs/1609.05917).

| Shape m | Coupling | Median speed ratio | RMS (km/s) | Random-error loss | Inclination loss |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.5 | 3 | 0.675 | 42.78 | 514.09 | 17.34 |
| 0.5 | 6 | 0.696 | 40.85 | 442.23 | 13.44 |
| 1 | 3 | 0.692 | 36.99 | 381.19 | 27.07 |
| 1 | 6 | 0.722 | 35.05 | 318.25 | 19.01 |
| 2 | 3 | 0.705 | 34.18 | 312.39 | 35.53 |
| 2 | 6 | 0.735 | 32.37 | 256.37 | 26.32 |
| RAR | — | 0.941 | 7.58 | 17.10 | 3.98 |

The primary loss averages squared residuals divided by published random-error
variance. The other diagnostics use a shared inclination covariance and a
5 km/s floor. These are descriptive losses, not calibrated likelihoods.
Independent replay verified every stored velocity, selected force, residual,
all three losses, radial strata, largest-residual removal and symmetric trim
for the 6,237 original and 891 refined scored records. Neither radial
influence diagnostic reverses any admitted comparison. Removing the only
galaxy leaves no sample, so object-level influence is undefined.

## Numerical and provenance checks

The source is the same conditional axisymmetric S4G/THINGS/HERACLES
reconstruction used by the scalar predecessor. Eleven source variants and
nine geometry choices retain all 24 selected response radii in 2–20 kpc.
The 97 numerical probe radii are unchanged. Primary Newtonian replay is exact
against stored predecessor forces at those radii. This run accesses derived
snapshots only; it opens no new raw observations or reserved holdout.

The initial calculation retains 144 auxiliary solutions across coarse/fine,
map and boundary comparisons; the refinement adds six higher-resolution
solutions. Coarse/fine grids use 1,025/2,049 radial nodes, 192/320 angles and
Legendre order 48/80. The follow-up uses 4,097 radial nodes, 640 angles and
order 160 on the same density. New Newtonian changes in the two refined
variants are 0.3480% and 0.3219%. The old scalar term is retained and its
absolute coarse/fine change is added to the auxiliary refinement change.
This sum is a numerical discrepancy budget, not a rigorous error bound.
The primary boundary and map gates remain 0.5% and 3% respectively.

An unequal two-cloud control conserves internal force to a maximum normalized
residual of 2.09e-07.
Independent β=0 source integration agrees with the flux solution to
2.53e-06.
An initial independent-control coordinate-basis bug failed before galaxy
scoring; that failed run and its exact input bytes remain preserved. The
corrected control and full run pass. A later lint cleanup binds loop variables
explicitly in the refinement runner; the exact executed version remains
snapshotted, and each loop's workers completed before its variables changed.
All 162 focused implementation checks pass; lint also passes for the new code.

Input snapshots and both scientific result hashes were verified independently.
Parent result SHA-256: `94a119f5a0ec415ef227b758e5f4e86a8f0e95454569ae2d96bc1102d1300b80`.
Refinement result SHA-256: `bb650959e3cf70eb47418d3cfd185459bc5e2b73b4d57fae46445187ecb200ca`.

## Limits and next work

This response is gas-traced circular motion, not a direct outer-star sample.
The source/response covariance, stellar mass conversion, outer map coverage,
vertical lift, warps, noncircular motion and environmental field remain
unresolved. Distance changes use homologous source scaling; inclination
changes affect velocity deprojection only, not source-map reprojection.
Shared geometry/calibration may correlate source and response; shared raw
measurements have not been established. Quality-verified and uncertainty-
resolved counterexample counts are both zero. No independent failure stratum
or unchanged independent replication is claimed from this one-galaxy pilot.

Keep the λ=10 cards pending numerical/source refinement. A physically distinct
next route is a higher-spatial-derivative action, which can introduce a universal
length and alter the connection between compact-system and galactic response.
Published GQUMOND supplies examples of that mechanism, not a demonstrated
solution to this project's three-regime problem.
[Milgrom, Generalizations of QUMOND (2023)](https://arxiv.org/abs/2305.01589)
Any proposed successor must include its full variational derivative, conservation
and boundary checks before observational scoring. Sigma's unfinished
thermodynamic-source route also remains open after measurement-model repair.
Matter/light coupling, stability and untouched cross-regime validation remain
required. The discovery goal stays active.
