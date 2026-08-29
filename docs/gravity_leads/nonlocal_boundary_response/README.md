# Nonlocal boundary response

## Status

This is the leading gravity-related empirical result. Item 59 passed its frozen
development and same-release confirmation gates on X-COP pressure and temperature
profiles. The exact beta = 1.5 unscreened representation is not a universal law.

Primary records:

- [Item 59 forward-observable result](../../GRAVITY_ITEM59_XCOP_FORWARD_OBSERVABLE_GATE_RESULT.md)
- [Item 60 direct-lensing readiness result](../../GRAVITY_ITEM60_DIRECT_CLASH_LENSING_GATE_RESULT.md)
- [Item 61 cross-scale result](../../GRAVITY_ITEM61_CROSS_SCALE_GATE_RESULT.md)
- [Items 62-70 formal gates](../../GRAVITY_ITEMS62_70_FORMAL_GATES_RESULT.md)
- [Items 71-72 confirmation and novelty](../../GRAVITY_ITEMS71_72_FINAL_GATES_RESULT.md)

## Core idea

The gravitational response at a radius may depend on the distribution of weak- and
strong-baryonic-field regions across neighboring radii, rather than only on the local
baryonic acceleration.

The selected phenomenological law was

    q(r) = (g_bar/a0) / ((g_bar/a0) + 0.1)

    g(r) = g_bar(r)
         + beta [g_bar(r) K_in[q](r) + a0 K_sym[q](r)]

where K_in is an inward radial occupancy average and K_sym is a symmetric radial
average on a fixed logarithmic-radius scale.

## What was learned

- Selection considered 2,025 frozen law/nuisance variants, including 1,863 creative
  variants.
- Eight development clusters supplied response-blind radial holdouts.
- The formula and nuisance choices transferred without refitting to four sealed
  clusters: A2029, A3158, A644, and RXC1825.
- It improved the confirmation score by 90.20% over the strongest frozen comparator.
- It beat all comparators for pressure and temperature in every confirmation cluster.
- Typical absolute confirmation error remained about 19%; this is a profile-shape and
  scale lead, not a precision theory.
- The result remained positive under six frozen density, outer-boundary, and
  member-baryon sensitivity variants.
- The preferred nuisance settings requested relatively high member baryons and
  nonthermal pressure. Modified gravity is therefore not the unique explanation.
- Item 60 correctly refused to open direct CLASH target rows because an acceleration
  curve does not define photon motion.
- The unchanged formula failed broadly on 139 SPARC and 11 LITTLE THINGS galaxies.
- Its unscreened high-acceleration limit approaches 2.5 times ordinary gravity, which
  analytically violates the declared Solar-System domain.

No individual galaxy killed the formula. Universal promotion failed because two
independent galaxy populations showed broad unchanged failure and the local limit has
a hard analytic contradiction.

## Relationship to known work

Known ingredients include MOND/RAR acceleration transitions, refracted-gravity
permittivity, nonlocal modified-Poisson kernels, TeVeS-like auxiliary fields, and
source-kernel gravity. The exact two-kernel occupancy expression was not located in the
scoped search. Its current label is potentially new synthesis of known motifs, not
historical novelty.

Useful starting papers:

- McGaugh, Lelli, and Schombert, radial acceleration relation:
  https://arxiv.org/abs/1609.05917
- Matsakos and Diaferio, refracted gravity:
  https://arxiv.org/abs/1603.04943
- Rahvar and Mashhoon, observational tests of nonlocal gravity:
  https://arxiv.org/abs/1401.4819
- Bekenstein, relativistic MOND/TeVeS:
  https://arxiv.org/abs/astro-ph/0403694
- Ettori et al., X-COP hydrostatic profiles:
  https://arxiv.org/abs/1805.00035

## First-principles gap

The present law has no covariant action, metric, universal transition variable,
conservation proof, stability proof, causal evolution, photon coupling, strong-field
limit, or cosmology. Its radial kernels are empirical operators.

## Suggested next steps

1. Replace the empirical kernel with a local auxiliary field whose Green function
   generates a finite-range nonlocal response.
2. Derive a universal dimensionless activation variable from baryonic compactness,
   boundary structure, thermodynamic state, and acceleration.
3. Add a high-acceleration screen before any new galaxy or cluster targets are opened.
4. Derive the two weak-field metric potentials so dynamics and lensing are predictions
   of the same field content.
5. Freeze scalar-only, transition-linked, and metric-like photon-response branches.
6. Test the transition on groups, then direct lensing, then a non-X-COP cluster release.
7. Run equivalence tests against MOND, refracted gravity, nonlocal gravity, and TeVeS.

The first concrete attempt is in
[SCREENED_COMPLETION_ATTEMPT_V0.md](SCREENED_COMPLETION_ATTEMPT_V0.md).

