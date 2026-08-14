# System 10 cylindrical sourced-constraint projection

The cylindrical `r=1` milestone previously closed 48 of the required 96 physical gravity rows:
four modified-harmonic gauge rows for each of 12 candidates. This follow-up resolves the next
coordinate convention without pretending that it resolves the remaining equations.

At `r=1`, the physical metric is `diag(-1,1,1,1)`, the future unit normal covector is
`n_mu=(-1,0,0,0)`, and the three coordinate tangents are orthonormal. In the registered symmetric
metric Euler row basis

```text
[E^00, sqrt(2)E^01, sqrt(2)E^02, sqrt(2)E^03,
 E^11, sqrt(2)E^12, sqrt(2)E^13, E^22, sqrt(2)E^23, E^33]
```

the physical projections are therefore exact:

```text
Hamiltonian_E_nn = row[0]
momentum_E_n1    = row[1] / sqrt(2)
momentum_E_n2    = row[2] / sqrt(2)
momentum_E_n3    = row[3] / sqrt(2)
```

The source convention remains the registered
`E_gf^mu_nu - T_total^mu_nu/2 = 0`. The physical constraints project the ungauged sourced action
Euler tensor; the modified-harmonic completion remains in the separate four gauge rows.

Five exact Einstein-Hilbert controls independently fix this projection. The flat cylindrical metric
gives four zeros. Mutating only a radial second jet gives Hamiltonian value `-1`; three mixed
time/spatial jet mutations independently give momentum values `1/2` in the radial, angular, and
axial slots. A wrong `sqrt(2)` normalization, a wrong matter-source sign, and partial advancement of
only one row are rejected.

## Honest boundary and resume point

The result binds the four projections to the sourced Euler and 85-state manifests of all 12
candidates, yielding 48 hash-bound **projection skeletons**. It does not count any skeleton as a
closed Hamiltonian/momentum coordinate-differential row. The measured gravity-row count therefore
remains 48/96.

The first missing primitive is exact and resumable:

```text
sourced_metric_euler_upper_row_0_as_
cylindrical_r1_85_state_spatial_differential_polynomial
```

It must prove cancellation of every `partial_0 v_A` acceleration and replace every
`partial_0 w_iA` by `partial_i v_A` before emitting a sparse polynomial over
`q_A, v_A, w_iA, partial_i(v_A), partial_i(w_jA)`. No absent coefficient may be inferred as zero.
One candidate advances only after its Hamiltonian row, all three momentum rows, and all controls
pass; the 12-candidate manifest cannot advance partially.

This specialization does not establish a general coordinate domain, sourced subsidiary
factorization, constraint propagation, H7, universal matter, or promotion.
