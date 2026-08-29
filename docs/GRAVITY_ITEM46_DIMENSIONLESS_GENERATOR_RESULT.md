# Gravity roadmap Item 46: dimensionless generator

## Outcome

Item 46 is complete as a **retrospective, non-promoted structural clue**. The system derived a complete bounded set of dimensionless monomials from a frozen physical units matrix before fitting any response. Its selected formula uses baryonic radial-profile slope, position within the baryonic body, and the cosmological age-expansion product.

The candidate improved balanced loss by 5.01% over Item 44 and by 13.69% over a scale-free search. It was 14.26% worse than Item 45, the strongest control, and was worse than Item 45 in both the S4TM and CLASH populations. Its paired sign-flip result was `p=0.3326`. The decision is:

`NONPROMOTED_ITEM46_DIMENSIONLESS_GENERATOR_RESULT_RETAINED`

No formula family is pruned. The selected group remains useful as a separately generated profile-structure hypothesis, but it is not confirmation, a new law of gravity, a dark-matter-free result, or a novelty claim.

## Exact dimensional generation

The frozen variables were `R`, `R_b`, enclosed baryonic mass `M`, `G`, `a0`, `c`, `H(z)`, cosmic age `t(z)`, and positive radial mass slope `q=d ln M_bar(<R)/d ln R`. Their length/time/mass matrix has exact rank 3 and nullity 6.

Three exact checks were used:

- rational Gaussian elimination gives rank 3;
- SymPy's exact nullspace gives rank 3 and six basis directions;
- bounded integer enumeration independently verifies zero dimensional residual for every emitted exponent vector.

With exponents in `[-2,2]` and L1 complexity at most 6, the generator found 184 canonical primitive Pi monomials. They span all six nullspace directions. Incrementing the `R` exponent in every one creates a nonzero length residual, so all 184 dimension-breaking negative controls reject.

The minimum-complexity independent basis is:

1. `q`;
2. `H t`;
3. `R/R_b`;
4. `a0/(c H)`;
5. `R_b/(c t)`;
6. `R_b c^2/(M G)`.

This basis is not a discovered law. Buckingham-Pi analysis says any admissible relationship can be written as an arbitrary function of six coordinates; it does not determine that function.

## Frozen empirical search

- Scientific-freeze commit: `aaa921d77b14899dbe74260737072aef9d7808da`
- Symbolic Pi recipes: 184
- Distinct Pi behaviors on development predictors: 184
- Parameter cells per Pi recipe: 4,096
- Raw formulas: 753,664
- Admitted formulas: 582,728
- Real development data: 28 S4TM galaxy lenses and 84 points from 20 CLASH clusters
- Reserved S4TM confirmations opened: 0 of 7
- Post-response candidates added: 0
- Paid model calls: 0
- RTX 5090 candidate-point-fold evaluations: 338,387,616
- CPU/GPU selected-loss difference: `6.66e-16`

All variables were converted to coherent SI values. Pi products were evaluated as exact exponent dot products in log space, avoiding numerical overflow. The feature receipt contains no observed response, target, or uncertainty. Replacing all source responses and uncertainties leaves the full feature receipt unchanged in the executable leakage test.

## Selected formula

Define

`q = d ln M_bar(<R) / d ln R`

and the selected dimensionless group

`Pi = R q^2 / (R_b H t)`.

The coordinate is

`H_pi = 1 / [1 + |log10(Pi)|]`.

The full-data descriptive candidate is

`nu = 1 + 4 u^-0.65 / (1 + u/300) * (0.05 + 0.95 H_pi)`,

where `u=g_bar/a0`.

In plain language, this asks whether the gravity enhancement depends on how quickly enclosed baryonic mass is growing with radius, adjusted for where the measurement lies relative to the baryonic size. The `H t` factor adds a mild cosmic-age/expansion correction. This is a generated dimensionless combination; historical novelty is unestablished and no causal interpretation has been derived.

Three outer folds selected `R q^2/(R_b H t)` and two selected the nearby `R q^2/R_b`. That is a repeatable structural theme but not a single fully stable formula.

## Comparative results

| Model | Balanced loss | S4TM loss | CLASH loss |
|---|---:|---:|---:|
| Item 45 universal interaction | 0.76148 | 0.18782 | 1.33514 |
| Item 46 dimensionless generator | 0.87009 | 0.24753 | 1.49265 |
| Item 44 scale hierarchy | 0.91598 | 0.16644 | 1.66552 |
| Minimum-complexity Pi basis | 0.91731 | 0.16225 | 1.67237 |
| Matched scale-free search | 1.00812 | 0.18094 | 1.83531 |
| Ordinary ridge | 1.87782 | 0.26047 | 3.49517 |
| MOND/RAR | 20.99692 | 0.78454 | 41.20930 |
| Baryonic Newton | 67.65046 | 0.91086 | 134.39006 |

The exhaustive bounded combinations improved by 5.15% over searching only the six independent primitive-basis coordinates. This says combinations of valid Pi directions can matter empirically. It does not say that this particular combination is fundamental.

Against Item 45, the Item 46 candidate remained worse under all four frozen mass-scale alternatives, by 3.73% to 18.24%. This is evidence that Item 45 remains the stronger current phenomenological lead, not a reason to erase Item 46's distinct slope-based idea.

## Counterexamples and uncertainty

Relative to Item 45, 32 of 48 objects were raw counterexamples: 20 S4TM galaxy lenses and 12 CLASH clusters. Thirteen remained counterexamples under all four global mass-scale shifts: 4 S4TM and 9 CLASH.

Those counts are recorded, not treated as votes that kill a formula. One or many discrepancies can reflect imperfect measurements, baryonic reconstruction, lens modeling, a missing variable, or an incorrect law. The existing data are model-derived and retrospective, and the four mass shifts are only a narrow uncertainty audit. Terminal rejection and family pruning are therefore false.

Leave-one-object and trimmed summaries consistently preserve the negative comparison with Item 45. This shows the deficit is broad rather than caused by one outlier. It does not establish universal failure on fresh data.

## Reproducibility

- Aggregate content SHA-256: `0de7202a74c5ebc6e71ebf2a257f0d17de484c6c91caa87dd9589d3ccf4a7b4e`
- Aggregate file SHA-256: `7094f9a9c1ce581cfcc473c816110878061e9da4455ae72d0ac9c4a05694b06b`
- Response-blind feature file SHA-256: `ab1405fe006693689943bc50a920d8aa7526cc13ea6464462c4278734ab10cd2`

Replay with:

`python -m sigma_theory_compiler.gravity_item46_dimensionless_generator replay`
