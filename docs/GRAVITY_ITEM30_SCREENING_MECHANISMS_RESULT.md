# Item 30 screening-mechanism result

## Decision

`INCONCLUSIVE_ITEM30_QUALITY`

Neither the universal-gravity track nor the phenomenon/publication track is promoted. Only 562 of
800 frozen exploration galaxies pass the unchanged response-quality rules, below both the 600-
object floor and the 75% retention floor. The gate is not lowered. On the unchanged diagnostic,
the selected screening response improves on a simple baryonic virial model but is decisively worse
than ordinary structural and stellar-population/environment controls. No partial slice passes.
The 200 confirmations remain sealed.

## Fresh target-blind sample

Item 30 reused the response-blind predictor table frozen before Item 12 and excluded every identity
or sky position assigned a role in Items 2 and 10--29. The audit combined 4,260 normalized
identities with 10,596 predecessor coordinate rows, including all 160 Item 24 ALFALFA roles. Of
6,333 valid MaNGA predictor records, 2,123 fail the exact-identity veto and another 55 fail the
60-arcsecond coordinate veto, leaving 4,155 fresh objects.

Official SDSS DR17 GEMA 2.0.2 environment fields then leave 1,511 objects with complete five-Mpc
footprint, at least 40% redshift completeness, and finite `Q_LSS`, `eta_k`, neighbor-distance, and
neighbor-count predictors. The sample freezes exactly 250 galaxies in each of four internal-
potential by environment cells: 200 exploration and 50 confirmation per cell. Every one of five
outer folds receives 160 exploration objects. No response column was read while exclusions, roles,
folds, candidates, or gates were created.

The later SDSS response query requested all 800 exploration `plateifu` values and returned all 800.
It requested zero confirmation identities. A source-schema incident is explicitly receipted: the
frozen parser treated SkyServer's leading `#Table1` metadata line as the CSV header and failed
closed on the first exploration-only chunk. The correction removes only blank and comment lines;
it changes no query, column, object, candidate, or gate. The response values were not displayed or
used to change the science.

## Equal-capacity screening search

Four mechanism niches receive exactly 65,536 distinct raw parameter cells and balanced enhancing
and suppressing polarity:

1. chameleon-like potential/thin-shell screening;
2. symmetron-like density/symmetry-restoration screening;
3. Vainshtein-like derivative/crossover screening; and
4. a potentially new dual-invariant internal-potential plus density boundary, optionally coupled
   to GEMA baryonic isolation.

Response-independent positivity, boundedness, materiality, monotone-screening, local-reference,
and exact-signature gates admit 180,100 of 262,144 cells:

| Niche | Raw | Admissible |
|---|---:|---:|
| Chameleon thin shell | 65,536 | 17,566 |
| Symmetron restoration | 65,536 | 64,604 |
| Vainshtein derivative screening | 65,536 | 34,268 |
| Dual-invariant boundary | 65,536 | 63,662 |

Every admitted cell has `0.5 <= mu <= 1.5` on the frozen galaxy-domain audit and fractional local
response no larger than `9.99985e-6`. Chameleon and symmetron cells are known-family controls;
Vainshtein cells are effective spherical projections; the dual-invariant niche is labeled only a
potentially new synthesis. No historical-novelty claim follows from that label.

## Response quality

The frozen rules retain 562 galaxies, or 70.25%. The minimums were 600 and 75%. Among the 238
failed objects, 233 have predictor `g`-band S/N below five, seven fail the stellar-dispersion
range, and seven fail the velocity-span range; some objects fail more than one rule. This is a real
sample-design miss, so the formal result remains inconclusive even though the negative diagnostic
is strong.

## Measured diagnostic

The physical baseline is a Sersic-dependent baryonic virial prediction with one globally trained
stellar-mass scale. The structural baseline adds ordinary mass, size, surface-density, shape,
color, redshift, and signal-quality variables. The strongest flexible baseline additionally uses
quadratic/interacting structural variables, spectral clocks, and GEMA environment fields.

| Model | Held-out MSE | Screening change relative to model |
|---|---:|---:|
| Baryonic Sersic virial | 0.016385 | **23.76% better** |
| Structural ridge | 0.008034 | **55.50% worse** |
| Flexible stellar-population/environment ridge | 0.005177 | **141.32% worse** |
| Selected screening candidate | 0.012493 | -- |

All five folds select enhancing dual-invariant cells with amplitude `0.5`, but all five also set
the optional GEMA environment coupling to zero. Thus the apparent family stability points to an
internal potential/density boundary, not an observed environmental screening effect. Exact
thresholds, scale factors, powers, and sharpness values vary across folds; two folds select the
same cell.

Every low/high half of stellar mass, internal potential, external tidal strength, and projected
neighbor density improves over the simple virial model by 4.41% to 35.33%. Every one of those same
slices is 111.66% to 195.55% worse than the flexible ordinary model. The selected formula is worse
than the flexible model for 391 of 562 individual galaxies. No preregistered partial slice reaches
the required 5% improvement over flexible.

The complete 180,100-cell selection is repeated inside 99 null trials. Its guarded result is
`p=1.0`; the observed improvement over flexible is negative, so it cannot be interpreted as a
phenomenon lead. The stable niche choice is failure-space information, not positive evidence.

## Controls and disclosed evaluation correction

- Synthetic signals from all four niches are recovered in all five folds.
- The known-GR control does not spuriously prefer screening.
- The largest CPU/GPU difference across 4,096 candidates is `1.94e-16`.
- All admitted cells pass the frozen local-response gate.
- The RTX 5090 evaluates 40,486,480,000 observed-plus-null training residuals; the candidate matrix
  is built in 0.218 seconds.
- Paid API calls, API spend, post-response formula generation, and confirmation reads are zero.

The first GPU execution completed scoring but failed closed before writing a result because the
slice evaluator emitted `improvement_vs_flexible_nuisance` while the partial collector requested
`improvement_vs_flexible`. A separately tested adapter supplies a read-only alias to the identical
frozen prediction, requires bit-exact equality of every duplicate slice MSE and improvement, and
removes the redundant MSE before serialization. It changes zero candidates, responses, fits,
nulls, thresholds, or decisions. The incident and correction are both replayable receipts.

## Interpretation and next action

This test does not establish chameleon, symmetron, Vainshtein, dual-invariant screening,
environment-dependent gravity, historical novelty, or an alternative to GR or dark matter. The
simple virial gain shows that a bounded internal potential/density transition can absorb some
missing ordinary galaxy structure, but the much stronger ordinary controls explain substantially
more. The environment term being zero in every fold is specifically negative evidence for the
tested scalar GEMA coupling.

Preserve all four tested regions, the internal-only dual-invariant selection, the 82,044
physics-invalid cells, and the 391 object counterexamples in the failure-space database. Do not
retune the 562 opened responses or query the 200 confirmations. Advance the numbered roadmap to
Item 31, vacuum polarization or gravitational permittivity, on a fresh response with equal-capacity
known-medium, running-permittivity, nonlocal-polarization, and newly synthesized vacuum-response
structures. The Item 12/13 age relation and Items 20, 22, 25, and 29 remain separate unchanged
replication hypotheses; Item 30 does not alter their evidence.
