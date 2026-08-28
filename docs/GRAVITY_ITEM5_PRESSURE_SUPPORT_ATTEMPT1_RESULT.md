# Gravity roadmap Item 5 attempt 1: raw H I pressure support

## Decision

**`INCONCLUSIVE_ITEM5_PRESSURE_SUPPORT_QUALITY_GATE`**

This first real-data pressure-support attempt does not pass and does not close Item 5. It
does establish two useful counterexamples: unsmoothed finite-difference pressure slopes
are not a valid representation for most of the frozen dwarf sample, and the proposed
nonlocal pressure-coherence terms do not beat the classical local formula on the five
galaxies where that representation remains physical.

The immutable receipt is `runs/gravity/roadmap/item-05-pressure-support-v1.json`.

## Frozen idea and creativity boundary

The test combined ordered H I rotation with random H I support measured by the line-width
dispersion and surface density. The dispersion contains unresolved thermal and turbulent
broadening. The known classical asymmetric-drift relation was

`V_c^2 = V_rot^2 - sigma_HI^2 d ln(Sigma_HI sigma_HI^2)/d ln(R)`.

That formula and its simple rescalings were explicitly nonqualifying. Three potentially
new structures were allowed to qualify: pressure curvature, a Gaussian nonlocal
log-radius pressure slope, and cumulative interior pressure memory. Historical novelty
was never claimed.

## Real-data and sealing boundary

The source uses Iorio et al. 2017 3D-BAROLO LITTLE THINGS profiles for raw rotation,
dispersion, and H I surface density. Oh et al. 2015's separate 2-D bulk-field reduction is
the primary corrected-circular-speed target. These are different reduction pipelines but
use the same telescope cubes, so this is not independent-telescope confirmation.

Sixteen fresh galaxies remained after excluding DDO 154, which Item 3 had opened, and the
alternative DDO 216 reduction. A salted split assigned 11 galaxies to exploration and five
to confirmation. Only exploration archive members and target queries were opened. The
confirmation galaxies DDO 101, DDO 210, DDO 50, NGC 2366, and WLM remain unread by the
extractor and unqueried as targets.

## Representation failure

Only five of eleven exploration galaxies pass the frozen representation. Raw local
finite-difference slopes make `V_classical^2` nonpositive somewhere in CVnIdwA, DDO 216,
DDO 53, DDO 87, NGC 1569, and UGC 8508. The failures are retained and were not replaced.

This is not surprising in hindsight: the source paper explicitly warns that fluctuations
in surface density and dispersion make numerical pressure derivatives unstable and uses
smooth functional forms. The frozen attempt deliberately did not specify a post-access
smoothing rescue, so changing it now would be response-informed method tuning.

## Held-out evidence on the five valid galaxies

The remaining 66 radial rows were evaluated with five whole-galaxy outer folds. The
classical local pressure family was selected in every fold. It achieved held-out
`R^2=0.853`; the qualifying-only selector achieved `R^2=0.818` and had worse
equal-galaxy MSE (`0.01018` versus `0.00830`). Its incremental MSE improvement relative to
the best rotation/classical control was `-0.00188`, with galaxy-level permutation
`p=0.224`.

Even the internal Iorio self-consistency diagnostic does not favor our raw derivative:
its classical prediction MSE is `0.00141`, worse than rotation alone at `0.00104`. That is
another representation warning, not evidence against the correctly smoothed physical
formula.

Only two of seven gates pass: positive held-out `R^2` in both frozen pressure strata and
untouched confirmation. Quality, internal positive control, universal qualifying
selection, incremental control improvement, and permutation gates fail.

## Failure-space boundary and next move

Later searches should skip:

- unsmoothed pointwise finite differences of `ln(Sigma_HI sigma_HI^2)` as a generally
  physical pressure-support operator;
- the tested curvature, 0.35-dex Gaussian-slope, and cumulative-memory features on these
  opened raw profiles;
- any claim that reproducing an internally corrected Iorio curve is independent evidence.

Item 5 now requires a second real-data attempt with its smoothing representation frozen
before access on a materially different source. The five reserved galaxies remain sealed;
they must not be used to repair this failed exploration.

## Replay

```powershell
python -m sigma_theory_compiler.gravity_item5_pressure_support check-sample
python -m sigma_theory_compiler.gravity_item5_pressure_support_experiment --check
python -m pytest tests/test_gravity_item5_pressure_support.py -q
```

The run tested eight families, five whole-galaxy folds, 499 galaxy-block permutations,
zero paid model calls, and zero confirmation accesses.
