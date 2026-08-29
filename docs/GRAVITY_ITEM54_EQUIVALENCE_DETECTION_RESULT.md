# Gravity roadmap Item 54: equivalence detection result

## Outcome

**PASS: LAYERED EQUIVALENCE IS OPERATIONAL AND LINEAGE IS PRESERVED.**

Invariant tested all 878 unique Item 52 representative ordinals with three layers:

1. exact ordinal identity;
2. frozen symbolic rewrite rules; and
3. rounded behavioral equality across the primary 112 predictor rows and four baryonic-mass
   shift environments, for 560 predictor cells per formula.

The system detected known adversarial rewrites correctly, found one real behavioral alias
pair, and preserved every ordinal, lineage record, and protected archive reference.

## Control test

The ten-row adversarial suite passed all five checks:

- exact duplicate detected;
- commutative product rewrite detected;
- two zero-producing structures collapsed;
- equal-operand max/min unary forms collapsed; and
- an adjacent but unequal outer-parameter case was not merged.

These controls show that the implemented rules work. They do not prove that the detector
recognizes every possible algebraic identity.

## Real archive result

| Layer | Input formulas | Classes | Aliases |
|---|---:|---:|---:|
| Exact ordinal | 878 | 878 | 0 |
| Frozen symbolic rules | 878 | 878 | 0 |
| Five-environment behavior | 878 | **877** | **1** |

The real behavioral alias class contains ordinals `341577670407` and `341715123975`.
They are symbolically different:

- one takes a weighted maximum of transformed environment-time and density-time features;
- the other takes a weighted minimum of an environment-time feature and the dimensionless
  ratio `Rb/(c*t)`.

Yet they produce the same rounded response on all 560 declared predictor cells. This is
exactly why lineage must survive equivalence compression: two apparently different
mechanisms can be observationally indistinguishable in the current environments without
being algebraically identical everywhere.

Invariant may evaluate one representative for efficiency, but the record retains both
ordinals, both Item 52 region memberships, and both mechanism descriptions.

## Preservation audit

- Original ordinals deleted: zero.
- Lineage records deleted: zero.
- Protected Item 53 archive references checked: 128.
- Protected archive references deleted: zero.
- Formula families pruned: zero.

Behavioral equivalence is explicitly scoped to rounded equality on the five declared
predictor environments. It is not labeled a global algebraic identity, a proof of the same
physics, or historical non-novelty.

## Interpretation

In lay terms, the engine can notice when two differently written ideas behave identically
in every test we have currently applied, avoid needlessly calculating both, and still
remember that they came from different conceptual routes. If new data separate them later,
both original ideas remain recoverable.

This improves search efficiency and protects creativity, but it does not improve the
current gravity fit or establish an alternative to general relativity.

## Reproduction

- `runs/gravity/roadmap/item-54-equivalence-detection-v1.json`
- `runs/gravity/roadmap/item-54-equivalence-detection-v1-source/equivalence-manifest.json`
- `runs/gravity/roadmap/item-54-equivalence-detection-v1-source/control-test-result.json`

```powershell
python -m sigma_theory_compiler.gravity_item54_equivalence_detection replay
```

The next task is Item 55: causal-variable testing with ablations, matched subsets, and
out-of-distribution splits to determine whether features are causes, proxies, or dataset
labels.
