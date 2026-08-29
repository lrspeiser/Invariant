# Item 59 X-COP forward-observable gate result

## Decision

`ITEM59_XCOP_FORWARD_OBSERVABLE_GATE_PASSED_DEVELOPMENT_EVIDENCE`

One frozen baryon-conditioned law predicted held-out Planck SZ pressure and XMM
temperature profiles substantially better than every frozen comparator. It passed on
response-blind radii in eight development clusters and then transferred without a formula
or nuisance refit to four previously sealed X-COP clusters.

This is the roadmap's clearest forward-observable success so far. It is not a discovery of
new gravity. The test assumes spherical hydrostatic balance, supplies one measured outer
pressure point per cluster, lacks a complete joint covariance, and comes from one survey
and reduction family. The global counterexample policy therefore classifies it as
`QUALITY_LIMITED_EVIDENCE_RETAINED` even though every frozen Item 59 performance gate
passed.

## Frozen test

The scientific design was frozen in commit `5bea3db1` before the pressure or temperature
rows of A2029, A3158, A644, or RXC1825 were parsed. The eight development clusters had
already been exposed by Item 3. Within those eight, target-blind hashing assigned 80 rows
to selection and 76 rows to radial holdout. The four confirmation clusters supplied 77
additional scored rows.

For each cluster, the evaluator used:

- the released XMM electron-density profile as the gas-mass predictor;
- the released cumulative stellar profile where available, or one globally selected
  stellar-to-gas bracket where absent;
- the outermost usable Planck SZ pressure point as an unscored boundary condition;
- hydrostatic integration to predict all usable interior SZ pressure rows;
- `kT=P_e/n_e` with one global XMM/Planck cross-calibration factor to predict XMM
  temperature rows.

It never opened the released hydrostatic-mass, total-mass, gas-fraction, NFW, entropy, or
lensing products. There were zero inferred-total-mass rows and zero direct-lensing
likelihood evaluations.

## Selected law

The selection evaluated 2,025 frozen law-and-nuisance variants. Of these, 1,863 belonged
to four qualifying creative families. The winner was the cross-scale boundary family:

```text
q(r) = (g_bar/a0) / (g_bar/a0 + 0.1)

g(r) = g_bar(r)
     + beta * [g_bar(r) * K_in[q](r) + a0 * K_sym[q](r)]
```

`K_in` is an inward radial occupancy average and `K_sym` is a symmetric radial occupancy
average, both using the frozen logarithmic-radius scale 0.25. The universal selected
coefficient was `beta=1.5`.

The origin label is
`new_combination_of_known_permittivity_and_auxiliary_field_ideas`. That means the system
combined known field-response motifs in a potentially new way; it is not a claim of
historical novelty or a completed action derivation.

The globally selected nuisance values were:

- outer nonthermal-pressure fraction: 0.30 with the frozen linear radial ramp;
- published stellar-mass scale: 1.30;
- stellar-to-gas mass ratio when the stellar profile was unavailable: 0.20;
- XMM/Planck temperature cross-calibration: 1.00.

The first three values are at the high end of their frozen grids. That is scientifically
important: the result prefers more member baryons and more nonthermal support, so broader
independent nuisance constraints are required before interpreting `beta` as gravity.

## Relative predictive result

The score is an equal-cluster, equal-observable mean squared log residual divided by the
released fractional error, with a frozen 5% error floor. Lower is better.

| Split | Cross-scale law | Empirical RAR | Training-only cubic pressure shape | Newtonian baryons |
|---|---:|---:|---:|---:|
| Development selection | 9.624 | 99.210 | 213.783 | 676.687 |
| Development radial holdout | 9.323 | 95.993 | 315.903 | 673.385 |
| Four sealed confirmation clusters | 9.648 | 98.493 | 277.904 | 668.347 |

Against the strongest aggregate comparator, the candidate improved the radial-holdout
score by 90.29% and the independent-confirmation score by 90.20%.

The result was not carried by only one observable:

| Confirmation observable | Cross-scale score | Strongest comparator score | Minimum improvement |
|---|---:|---:|---:|
| Planck SZ pressure | 3.177 | 19.580 | 83.77% |
| XMM temperature | 16.119 | 177.406 | 90.91% |

The candidate also beat every comparator separately for both observables on the
development radial holdout.

## Absolute accuracy

Relative success does not mean a precision fit. Across the 77 confirmation rows, the
median absolute log residual was 0.1723, corresponding to a typical multiplicative error
of about 19%. The root-mean-square log residual was 0.2358. Pressure and temperature had
nearly identical median fractional discrepancies, about 18.6% and 18.8% respectively.

The standardized temperature score remained high because many released statistical
errors are much smaller than the residual model and cross-calibration uncertainty. The
present formula should therefore be regarded as a strong profile-shape and scale lead,
not a final precision description of intracluster gas.

## Confirmation by cluster and counterexamples

The candidate beat all three comparators in the aggregate score of each sealed cluster:
A2029, A3158, A644, and RXC1825. It also beat the strongest comparator separately for
pressure and temperature in every one of those clusters. Thus there were zero raw or
systematics-stable confirmation counterexamples at the frozen cluster-observable level.

This absence does not make future mismatches terminal. Any later single counterexample
will be retained and audited for density, pressure-boundary, member-baryon, calibration,
projection, and covariance effects. One mismatch cannot eliminate this representation,
and no finite collection of these clusters can prune the broader family.

Removing any one confirmation cluster preserved a positive advantage. Symmetrically
trimming the strongest and weakest cluster advantages also preserved it.

## Frozen systematic envelopes

The candidate retained a positive confirmation advantage over every baseline under all
six frozen variants:

| Variant | Minimum improvement over every baseline |
|---|---:|
| density minus released error | 91.17% |
| density plus released error | 88.79% |
| outer pressure boundary minus error | 90.76% |
| outer pressure boundary plus error | 89.55% |
| low member-baryon bracket | 89.34% |
| high member-baryon bracket | 90.25% |

These are useful sensitivity checks, not a substitute for a joint covariance or an
independent reduction.

## What this result means

In plain language, ordinary gas and stellar matter by itself produces far too little
inward pressure gradient under Newtonian gravity. The galaxy RAR helps but still falls
well short. A radial field term that responds to how much baryonic acceleration occupies
the interior and nearby region generates the extra acceleration scale needed to reproduce
the observed pressure and temperature shapes much more closely.

The same frozen term worked on new clusters after being selected on the old ones. That is
the important step forward: the system did not merely fit a derived mass or coefficient;
it integrated a proposed acceleration law into two observable thermodynamic profiles.

The unresolved interpretation is equally important. The extra term may be standing in
for modified gravity, collisionless mass, underestimated member baryons, nonthermal
pressure, projection, or a combination. Item 59 distinguishes the formula's predictive
utility from its physical cause; it does not yet distinguish those causes.

## Next test

Item 60 must confront a frozen descendant with direct CLASH lensing observables—image
positions, parities, shapes, shear, magnification, and time delays—without using a
GR/NFW-derived total-mass profile as the target. The X-COP result and all nuisance-edge
warnings must remain frozen. No refit to the four confirmation clusters is allowed.

## Reproduction

```powershell
python -m sigma_theory_compiler.gravity_item59_xcop_forward_observable_gate replay
python -m pytest tests/test_gravity_item59_xcop_forward_observable_gate.py -q
```

The replay verifies the frozen archive/member boundary and byte-recreates the evaluation
and aggregate result. Paid model calls: zero. GPU use: none.
