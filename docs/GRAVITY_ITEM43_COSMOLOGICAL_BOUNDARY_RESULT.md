# Gravity Item 43 cosmological-boundary result

Date: **2026-08-29**

## Decision

`NONPROMOTED_ITEM43_COSMOLOGICAL_BOUNDARY_RESULT_RETAINED`

Item 43 found a repeatable finite-horizon formula on a fresh strong-lens exploration
sample, but the result does not pass promotion. In nested whole-lens testing on 28
previously unopened S4TM lenses, adding the frozen horizon coordinate improves the
matched boundary-free formula by **7.88%**. It is nevertheless **6.83% worse** than
an ordinary ridge control using the same measured coordinates, has paired
`p=0.6681`, and is not stable to every leave-one-lens or trimmed audit.

The unchanged formula then fails its cross-scale test on all 20 CLASH clusters. Its
equal-cluster loss is **128.46**, versus **2.42** for the S4TM-selected matched
boundary-free formula and **41.21** for fixed MOND/RAR. This is strong evidence
against this exact finite-horizon scaling as a universal galaxy-to-cluster law. It
is not grounds to erase cosmological boundary coupling as a family: CLASH is an
already exposed, model-dependent lensing reconstruction, the stellar-mass and lens
models are imperfect, and no independent unchanged replication has failed.

The complete aggregate receipt is
`runs/gravity/roadmap/item-43-cosmological-boundary-v1.json`. Seven S4TM confirmation
lenses remain sealed. No paid model call was made.

## What was tested

The search generated exactly 262,144 formulas in four equally sized mechanism
niches:

1. expansion-rate running, using powers of `H(z)/H0`;
2. cosmic-age memory, using powers of `t0/t(z)`;
3. scale-factor dependence, using powers of `1+z`; and
4. finite-horizon dependence, using powers of
   `1 + R H(z)/(c x_ref)`.

Every formula had the common weak-field form

`nu = 1 + A u^(-p)/(1 + u/u_t) B_cosmic`,

where `u=g_bar/a0`. The generator allowed positive, zero, and negative cosmological
exponents, required a Newtonian high-acceleration limit, and admitted 186,989
formula cells. The four idea families began with equal capacity. No response value,
object identity, or post-response formula entered generation.

The matched boundary-free control searches the same `A`, `p`, and `u_t` grid with
the cosmological exponent fixed to zero. It therefore tests the incremental value
of the cosmological coordinate rather than merely asking whether a flexible
modified-gravity curve can fit the lenses.

## Prospective S4TM experiment

The primary data are the 40 grade-A S4TM galaxy-scale strong lenses. During source
schema inspection, five all-column rows were accidentally exposed. Those five
identities were permanently excluded and none of their values was used to design a
formula or sample. Predictor-only queries then acquired redshift, size, Einstein
radius, lens shape, and Chabrier stellar mass for the remaining 35 lenses.

A deterministic whole-lens split assigned 28 objects to exploration and seven to
sealed confirmation before any of the remaining Einstein masses were read. The
exploration lenses span foreground redshift `0.0625-0.3214` and Einstein radius
`0.54-1.53 arcsec`. The test approximates each early-type galaxy with a projected
de Vaucouleurs stellar profile and predicts the lens-model mass inside the published
Einstein radius. Cold gas is unavailable and is treated as missing baryonic data,
not silently set equal to a measured zero.

Candidate selection is nested inside five whole-lens folds. Thus each reported
exploration prediction comes from a formula selected without that lens's Einstein
mass. The ordinary ridge is cross-fitted the same way.

| Model | Out-of-fold loss | Relative to selected formula |
|---|---:|---:|
| Baryonic Newton | 0.91086 | selected is 81.45% better |
| Fixed MOND/RAR | 0.78454 | selected is 78.46% better |
| Matched boundary-free search | 0.18341 | selected is **7.88% better** |
| Ordinary coordinate ridge | **0.15815** | selected is **6.83% worse** |
| Cosmological-boundary search | **0.16896** | — |

All five outer folds select the finite-horizon niche with a negative boundary
exponent. The formula selected on all 28 exploration lenses is candidate `259715`:

`nu = 1 + 6 u^(-0.4)/(1 + u/10) [1 + R H(z)/(c 10^-6)]^(-1.5)`.

In plain language, it says the extra gravitational multiplier becomes weaker as an
object occupies a larger fraction of the cosmic horizon. This is not a derivation;
it is the best member of the frozen empirical grammar.

Relative to the strongest overall control, 13 lenses are raw counterexamples. Only
one, `SDSSJ1116+0729`, remains a mismatch across the frozen stellar-mass,
effective-radius, and background-cosmology variations relative to the strongest
fixed physical control. The executable policy therefore classifies it as
`ISOLATED_EMPIRICAL_COUNTEREXAMPLE_RETAINED`: one uncertainty-resolved object is a
reason to inspect and retest, not a veto.

## Unchanged CLASH transfer

Candidate `259715` was carried to the existing 20-cluster, 84-point CLASH
acceleration table with no selection or retuning. Published CLASH redshifts supply
`H(z)` and the published radii supply the horizon fraction. The same zero-slip
weak-field multiplier is used for light as for motion.

| Model | Equal-cluster loss |
|---|---:|
| Baryonic Newton | 134.39 |
| Fixed MOND/RAR | 41.21 |
| S4TM-selected matched boundary-free formula | **2.42** |
| Unchanged finite-horizon formula | **128.46** |

All 20 clusters are raw counterexamples to the finite-horizon formula relative to
the matched boundary-free control. The reason is mechanistically useful: cluster
radii make the horizon coordinate much larger, and the selected exponent `-1.5`
suppresses almost all of the extra gravity. A universal scale law that behaves this
way cannot bridge these galaxy and cluster representations.

These 20 mismatches are correlated evidence from one exposed catalogue, not 20
independent replications. No uncertainty-resolved CLASH count is claimed because
the NFW-derived total accelerations, baryonic mass model, and shared calibration
systematics were not re-derived from raw observations here.

## Compute, creativity, and claim boundary

The NVIDIA GeForce RTX 5090 performed 26,624,080 candidate-point-fold evaluations.
The selected CPU and GPU losses agree to `2.78e-17`. There were zero post-response
candidate cells, zero confirmation accesses, and zero paid model calls.

After the exploration responses were opened, an implementation audit found that
the first evaluator executed the four frozen stellar-mass and effective-radius
variants but omitted the two already frozen background-density variants. The
evaluator was repaired to execute `Omega_m=0.25` and `0.35` with flat complements.
This added no formula, coefficient, object, or response access and changed no
nominal selection or score; all six predeclared variants now run and replay.

The horizon idea itself is not historically new. The potentially creative element
is the particular synthesis of a horizon fraction with a baryonic-acceleration
multiplier and its prospective cross-scale falsification. The pipeline labels that
distinction explicitly; it does not call an empirical combination a new theory.

Item 43 does not establish cosmological boundary gravity, an alternative to general
relativity, the absence of dark matter, a covariant field equation, or a historically
new law. It establishes that the system can freeze a large cosmological grammar,
run a response-blind real-data test, distinguish a boundary increment from its
matched non-boundary parent, use the GPU for nested search, and learn a specific
cross-scale failure without pruning the whole family.

The next roadmap item is Item 44, scale hierarchy. Candidate `259715`, its lone
uncertainty-resolved S4TM mismatch, and its strong cluster-scale failure pattern
should all remain in the counterexample database. The seven S4TM confirmation
responses should remain sealed rather than being spent on a candidate that failed
the predeclared promotion and cross-scale gates.
