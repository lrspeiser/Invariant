# Archimedes first-principles discovery control

## Question

Can Invariant start from a heterogeneous dataset, recover a compact physical relation, reject
plausible alternatives, and provide a first-principles route that explains the relation rather
than stopping at curve fitting?

This control uses Archimedes' law because the final formula is simple and the derivation is easy to
audit. It is a known-answer calibration, not a novelty benchmark.

## Data presented to the discovery boundary

The force table has 2,187 noisy training rows and 128 shifted holdout rows. Every row contains:

- measured buoyant force;
- fluid density, local gravity, and displaced volume;
- depth, ambient pressure, object density, and a dimensionless shape factor as nuisance variables.

The holdout uses values absent from training. Multiplicative measurement perturbations range from
`199/200` to `201/200`, and every measurement carries a one-percent exact rational interval.

A separate 243-row pressure-sensor table varies density, gravity, vertical separation, depth, and
ambient pressure. The discovery search does not receive either target exponent vector.

## Discovery result

Dimensional filtering left 98 admissible force relations. Independent Fraction and SymPy scoring
agreed on the same winner:

\[
\frac{F_b}{\rho g V}=\text{constant}.
\]

The winner's exact noisy relative span was `2/199`, and the same span held on the shifted data. The
combined measurement intervals intersect at

\[
\left[\frac{19899}{20000},\frac{20099}{20000}\right],
\]

which contains the exact constant one. The runner-up,
`buoyant_force/(gravity*displaced_volume*object_density)`, had relative span `1013/796`.

The pressure table independently selected, from 20 admissible relations,

\[
\frac{\Delta p}{\rho g\Delta h}=1
\]

with zero exact span. Its nearest competitor had relative span 209.

## Explanation produced

The recorded proof plan connects the two discoveries:

1. The sensor data gives the hydrostatic pressure gradient
   \(\Delta p=\rho g\Delta h\).
2. Pressure force on a closed submerged surface is
   \(F_z=-\oint p n_z\,dA\).
3. The constant ambient-pressure term cancels because
   \(\oint n_z\,dA=0\).
4. Substituting the hydrostatic gradient leaves
   \(\rho g\oint z n_z\,dA\).
5. The divergence theorem gives
   \(\oint z n_z\,dA=\int_V 1\,dV=V\).
6. Therefore \(F_b=\rho gV\).

An exact SymPy rectangular-prism surface calculation independently checks the ambient cancellation,
the volume identity, and the final pressure resultant.

## Controls

- Changing depth, ambient pressure, object density, or shape factor leaves the winning invariant at
  exactly one in intervention rows.
- Dropping density, gravity, or volume is rejected.
- Replacing volume with depth cubed is dimensionally legal but rejected by the data.
- Replacing fluid density with object density and adding shape dependence are rejected.
- A deliberately confounded 729-row slice sets object density equal to fluid density. Five candidates
  then tie, so the system returns `blocked_unidentifiable` instead of fabricating a unique law.
- The receipt fails closed if the formula, exponent vector, proof status, source, or seal changes.

## Reproduce

```text
python -m sigma_theory_compiler.archimedes_first_principles run --root .
python -m pytest -q tests/test_archimedes_first_principles.py
```

The committed receipt status is `PASS_BOUNDED_FIRST_PRINCIPLES_DISCOVERY_CONTROL`.

## Claim boundary and next step

This is a synthetic, known-answer calibration with declared physical dimensions. It demonstrates a
working path from heterogeneous data through relation discovery, stress testing, identifiability
checking, and a symbolic first-principles explanation. It does not show historical novelty or that
the relation could be recovered from arbitrary raw sensor columns without metadata.

The next meaningful step is to replace the generated grids with a real buoyancy lab CSV while
leaving the search, controls, holdout policy, and derivation checker unchanged.
