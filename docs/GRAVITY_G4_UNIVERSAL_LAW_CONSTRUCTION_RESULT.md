# G4 compact universal-law construction result

Date: **2026-08-27**

## Decision

`BLOCK_G4_EXPLORATION_CONSTRUCTION`

The sealed receipt is `runs/gravity/g4/universal-galaxy-law-construction-v1.json`. No
confirmation galaxy was evaluated and the one-shot confirmation stage remains unauthorized.

## Search completed

The finite family contains the six target-blind, dimensionless local features `log_y`, radius
relative to the disk peak, gas fraction, disk fraction, bulge fraction, and baryonic log slope,
plus every commutative degree-two product. It therefore contains 27 terms. Each term was paired
with 19 frozen shrinkages and two coefficients fitted only on the outer-training galaxies:
**513 formula cells, 71,307 candidate-galaxy evaluations, and 1,395,360 candidate-point
evaluations**. The formulas have zero per-galaxy gravitational constants.

Observer distance, galaxy identity, held-out velocity during fitting, sample-maximum-normalized
mass proxy, the dimensioned gas-to-disk regularizer used in G3 diagnostics, and halo quantities
were excluded.

## Best valid formula

The best positive, finite whole-galaxy prediction was

```text
V^2 = V_RAR^2 + r g_dagger [0.03475306099478
                             + 0.01350704618429 log(g_bar/g_dagger)].
```

Its cross-validated exploration chi-square was **125,143.249**, versus **130,714.689** for
the frozen empirical RAR and **1,697,326.398** for Newtonian baryons. This is a **4.26%**
improvement over RAR. All 139 galaxies and 2,720 points were predicted, every prediction was
positive and finite, and no surface-brightness, gas-fraction, or bulge-fraction quartile
regressed by the allowed 10%.

## Why it blocked

The frozen NFW-shaped two-parameter-per-galaxy performance ceiling has chi-square **28,018.807**.
With the predeclared slack of two chi-square units per point, the G4 limit is **33,458.807**.
The compact candidate exceeds that limit by **91,684.442**. That is not a near miss.

The winning term depends only on acceleration and recalibrates the known empirical RAR base.
It is therefore classified `KNOWN_FAMILY`, not as a new theory or a historically novel
construction. The result proves only that this declared 513-cell compact family contains no
formula that clears the complete G4 exploration gate. It does not exclude other mathematical
families.

The next scientifically justified work returns to G3/G4 model generation and must explain the
large object-to-object residual variation without introducing hidden gravitational constants.
