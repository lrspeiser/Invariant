# G4 conditional formula generator result

Date: **2026-08-27**

## Decision

`BLOCK_G4_CONDITIONAL_GENERATOR`

The sealed receipt is `runs/gravity/g4/conditional-formula-generator-v4.json`. The run generated
one formula instance for each of **139 SPARC exploration galaxies and 2,720 published rotation-curve
points**, with zero per-galaxy fitted gravitational constants. The 35 confirmation galaxies were
not accessed.

## What the generator tested

The formula-of-formulas lane conditioned every one of the **8,609 G2 structural classes** on each
of seven bounded, target-blind galaxy descriptors: baryonic compactness, surface density, gas
dominance, bulge dominance, vacuum fraction, radial span, and RAR-speed coherence. Affine maps were
tested exhaustively; the best 64 class/condition pairs received quadratic and cubic expansions.

Two explicit concept lanes tested the user-supplied hypotheses:

- **Baryonic focusing:** a radial correction focuses toward occupied baryonic regions and away
  from adjacent low-density regions. The grammar contains 48 operators over surface-brightness or
  baryonic-acceleration occupancy, interior/exterior kernels, thresholds, and radial scales.
- **Secular speed synchronization:** a nonlocal relaxation moves a local circular-speed prediction
  toward an interior or symmetric radial mean, optionally weighted by the number of RAR orbital
  cycles in 1, 3, or 10 Gyr. The grammar contains 32 operators. This is a static phenomenological
  test, not a derivation of resonance or energy transfer.

For each concept operator the amplitude is generated from a galaxy descriptor,
`alpha(x) = alpha_max sigmoid(a + b x)`. The search represented **261,348 formula cells**. After 24
structural classes were rejected by target-blind domain counterexamples, **260,676 cells** reached
numerical scoring: **36,233,964 candidate-galaxy evaluations** and **709,038,720 candidate-point
evaluations**. All concept features and galaxy descriptors have tests proving that poisoning the
observed velocities and uncertainties does not change them.

## Best generated law

The winner came from baryonic focusing, not an atlas rewrite. Define

```text
q(r)       = SB_star(r) / (SB_star(r) + 100 L_sun pc^-2)
s_g        = tanh((median_r log(1 + SB_star(r)) - 4) / 4)
alpha_g    = 2 sigmoid(-2 + 4 s_g)
I_in[q](r) = normalized interior exponential mean of q at log-radius scale 0.25
I_out[q](r)= normalized exterior exponential mean of q at log-radius scale 0.25

V_pred^2(r) = V_RAR^2(r)
              + alpha_g r g_dagger I_in[q](r) (1 - I_out[q](r)).
```

This single generator has two enumerated amplitude constants plus fixed threshold/kernel choices.
It produces **136 distinct coefficient values** across 139 galaxies from surface density alone; it
never receives galaxy identity or observed velocity as an input. Its chi-square is **106,187.797**,
compared with **130,714.689** for empirical RAR: an **18.76% improvement**. The worst predefined
population stratum regresses by 1.86%, inside the 10% limit, and every prediction is positive and
finite.

The lane comparison matters:

- best baryonic-focusing generator: **106,187.797**;
- best pair of distinct concept operators: **107,864.890**;
- best conditional affine atlas generator: **120,733.577**;
- best synchronization/resonance-inspired generator: **122,215.594**;
- previous G4 nonlocal-profile result: **120,016.785**.

The focusing hypothesis therefore supplied a real measured gain beyond the existing formula
atlas, while the synchronization hypothesis was retained but did not win this finite test.

## Why this is not yet a gravity theory

The unchanged NFW-shaped performance ceiling plus slack is **33,458.807**. The focusing law exceeds
it by **72,728.990**, so the confirmation gate remains locked. The run also selects its construction
on an already inspected exploration population; it is model-development evidence, not independent
replication.

The equation is a phenomenological radial operator applied to the empirical RAR. It does not derive
a three-dimensional field equation, conservation law, lensing prediction, causal propagation, or
an alternative to general relativity. Its origin status is conservatively `COMBINATION`: the
specific tested mixture may be useful, but this run does not establish historical novelty.

The strongest next generative move is to derive a signed focusing kernel from a field/action or
transport principle, then require that one operator to predict galaxies, clusters, and lensing. More
parameterizations selected on the same SPARC exploration population would improve fit evidence but
would not resolve identifiability or establish first principles.
