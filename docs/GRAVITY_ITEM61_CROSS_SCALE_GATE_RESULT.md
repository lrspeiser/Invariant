# Item 61 cross-scale gate result

## Decision

`ITEM61_CROSS_SCALE_GATE_NOT_PASSED_EXACT_PARAMETERIZATION_RETAINED`

The unchanged Item 59 cross-scale-boundary law does not transfer to galaxies with its
cluster-selected universal coefficient `beta=1.5`. It remains a strong cluster-profile
lead, but it is not currently one law covering galaxies through clusters.

This result withholds universal promotion of the exact parameterization. It does not erase
the X-COP result, reject the boundary/nonlocal mechanism family, or use one galaxy as a
veto.

## Frozen transfer

The evaluator preserved the Item 59 formula, `beta=1.5`, the transition acceleration
`a0=1.2e-10 m/s^2`, and the logarithmic-radius kernel scale 0.25. It made no galaxy-,
sample-, or scale-specific fit.

For each galaxy it reconstructed baryonic acceleration from the already frozen baryonic
velocity components, evaluated the same radial kernel, and converted the resulting
acceleration to a rotation speed. It evaluated:

- 2,720 radii in 139 SPARC exploration galaxies;
- 199 radii in 11 independently reduced LITTLE THINGS exploration galaxies;
- the already frozen Item 59 result for 12 X-COP clusters, including 77 rows in four
  previously sealed confirmation clusters.

The sealed SPARC confirmation sample remained unopened. The galaxy response rows had
already been exposed before Item 61, so their result is a fixed-candidate transfer
diagnostic, not fresh confirmation.

## Results

Lower scores are better. Each galaxy score is its mean squared velocity residual in units
of the published uncertainty; the table averages equally across galaxies.

| Population | Cross-scale `beta=1.5` | Empirical RAR | Newtonian baryons |
|---|---:|---:|---:|
| SPARC, 139 galaxies | 625.410 | 33.556 | 378.452 |
| LITTLE THINGS, 11 galaxies | 71.039 | 15.841 | 50.974 |

The candidate beat the empirical RAR in only 7 of 139 SPARC galaxies and 1 of 11 LITTLE
THINGS galaxies. It beat Newtonian baryons in 39 SPARC and 5 LITTLE THINGS galaxies.

In contrast, the same parameterization passed the Item 59 X-COP gate, improving the
four-cluster confirmation score by at least 90.20% against every frozen comparator.

The contradiction is therefore not “the formula never works.” It is more specific and
more useful: the cluster-sized response supplied by `beta=1.5` is generally too strong in
disks and dwarfs.

## Counterexample interpretation

No individual galaxy is terminal. Every object-level score is retained with
`terminal_veto=false`. The cross-scale gate fails because two galaxy populations show a
broad aggregate pattern and because there is no authorized direct group/transition sample,
not because one measurement missed.

Even this pattern does not prune the family. Galaxy baryonic masses, distances,
inclinations, asymmetric-drift corrections, and disk geometries remain imperfect. The
proper conclusion is that this exact universal coefficient cannot be promoted now. A
descendant may survive if a response-blind physical variable—not a galaxy/cluster label—
derives the change in effective strength.

## Scientific next step

The next constructive target is a universal transition function such as

```text
beta_eff = beta_max * S(Z)
```

where `Z` is measured from baryonic geometry, compactness, boundary structure,
thermodynamic state, external field, or a dimensionless timescale. `S(Z)` must approach a
small value in disks and a larger value in clusters without knowing the object's class.

A real group and transition-regime sample is essential. Groups sit between galaxies and
clusters and can distinguish a smooth physical transition from a hidden binary label.
Any descendant must be frozen before that sample is opened.

## Reproduction

```powershell
python -m sigma_theory_compiler.gravity_item61_cross_scale_gate replay
python -m pytest tests/test_gravity_item61_cross_scale_gate.py -q
```

Paid model calls: zero. GPU use: none.
