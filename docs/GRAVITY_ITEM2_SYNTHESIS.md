# Gravity roadmap Item 2 synthesis

## Scoped decision

**`REJECT_ITEM2_TESTED_PROJECTED_SHAPE_FAMILIES_ADVANCE_ITEM3`**

Five target-blind real-data attempts now cover the Item 2 branches required before moving
on: global shape, two-dimensional multipoles, comparable stellar tracers, radially resolved
and nonlocal summaries, and an intermediate-scale group/filamentarity representation.

The synthesis receipt is `runs/gravity/roadmap/item-02-synthesis-v1.json`:

- file SHA-256: `285a18b887eb265e1bd6e1c10e40adb13fcf47f52c2bbf111d725e1933d674fc`
- content SHA-256: `fcccd1facb379e2bf27db05dea263169714cff7739b3c5de2e266400327b0bbf`

## What is rejected

Projected baryonic shape summaries in the five frozen model families are not a sufficient
universal hidden variable for the tested galaxy-to-group-to-cluster gravitational
responses.

| Attempt | Material repair or extension | Decisive result |
|---|---|---|
| 1 | Continuous projected axis ratio and concentration across SPARC/CLASH | Separates populations but has negative prediction within each and fails feature overlap. |
| 2 | Target-blind 2-D unWISE/X-ray multipoles | Loses to the support proxy within galaxies, clusters, overlap, and the independent bar-sign check. |
| 3 | Comparable stellar tracers in galaxies and clusters | The representation independently validates, but response prediction remains negative within both populations. |
| 4 | Radial/nonlocal MaNGA stellar shape with direct dynamics | Loses to mass/size and morphology controls and fails every response variant. |
| 5 | Real intermediate AXES groups and graph filamentarity | Adds no significant, robust increment beyond luminosity, size, richness, redshift, and environment. |

No attempt qualifies for confirmation.  Across the program there were zero SPARC, MaNGA,
or AXES confirmation-target accesses, zero direct-lensing likelihood evaluations, and zero
paid model calls.

## What is not rejected

This is deliberately a scoped failure-space result.  It does not exclude:

- intrinsic three-dimensional shape or velocity anisotropy not recoverable from the tested
  projections;
- action-level tensor, scalar/vector polarization, torsion, nonmetric, or other geometric
  theories;
- genuinely different nonlocal operators rather than rewrites of the tested summaries;
- filament dynamics or lensing responses not represented by cleaned group membership;
- surface density, volume density, pressure, thermodynamics, time/history, environment,
  running coupling, massive modes, or the later roadmap mechanisms.

It establishes no alternative to GR, removal of dark matter, or historical novelty.

## Why advancing is more creative

The failure database now characterizes a substantial region of equation space.  Further
regressions over the same projected multipoles and responses would be increasingly likely
to mine noise or rediscover an equivalent formula.  Advancing to surface-versus-volume
density introduces a materially new dimensionless cause while retaining shape as an
interaction term only if the new derivation requires it.

Item 3 must start from a frozen dimensionless derivation and then face a real
galaxy/group/cluster test.  It may not use object identity, observed dynamics as an input,
lensing-derived total mass, a fitted dark halo, or an object-specific gravitational
constant.

## Replay

```powershell
python -m sigma_theory_compiler.gravity_item2_synthesis --root . --check
python -m pytest tests/test_gravity_item2_synthesis.py -q
```
