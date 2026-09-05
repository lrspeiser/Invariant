# Exterior source and potential join: retained results

No new gravity law is validated by these calculations. They repair the
Newtonian source derivatives needed by the current universal-length action.
No new gravity parameters, observational responses, raw data or reserved
confirmation products were opened or scored.

The original 80--120 kpc join failed its registered derivative targets. The
failure is retained. A higher-order exterior representation is a subsequent
numerical experiment; it does not retroactively pass the failed join.

## Exterior reference

| Run | Source | Highest order | Admission radius (kpc) | Uniform omitted-third-series bound | Maximum direct third-derivative difference | All exterior targets met |
| --- | --- | --- | --- | --- | --- | --- |
| exterior-moment-001 | primary | 64 | 80 | 4.16021e-10 | 3.24722e-14 | True |
| exterior-moment-001 | height_half | 64 | 80 | 1.7296e-11 | 3.06311e-14 | True |
| exterior-moment-002 | primary | 128 | 60 | 1.75418e-12 | 7.63428e-13 | True |
| exterior-moment-002 | height_half | 128 | 60 | 3.21241e-15 | 7.70985e-13 | True |

These exterior quantities use monopole units GM/r^(n+1), including the full
Cartesian tensor norms. The series bound is uniform over angle for the ideal
positive compact source; numerical moments and physical vertical tails are
separate checks. The direct comparison uses independent spatial quadrature
through infinite height. Its sampled agreement is not a uniform theorem for
the physical continuum tail. No gravitational-family rejection follows.

## First join failure

| Source | Largest force refinement | Largest third-tensor refinement | Near-field density-gradient identity error | Joined density-gradient identity error | All join targets met |
| --- | --- | --- | --- | --- | --- |
| primary | 4.31207e-07 | 0.0271342 | 0.0257245 | 0.018276 | False |
| height_half | 6.95633e-07 | 0.0541235 | 0.0513933 | 0.0365124 | False |

The fixed 80--120 kpc audit retains all 545 registered near-domain points and
152 join points per source. Its source-scaled tolerances were 0.0001 for force,
0.002 for Hessian and density, and 0.01 for the third tensor and density
gradient. These scales differ from the exterior monopole normalizers above.

The largest near-field density-gradient discrepancy occurred at R=96 kpc,
z=0 for both thicknesses. The largest third-tensor change was the cutoff
200-to-400 comparison at R=120 kpc, z=0. Radial integration refinement also
matters. Potential and force agreement alone did not diagnose these errors.
All derivatives of the join include the complete product rule, and neither
the trace nor its gradient is overwritten by the known source density.

The subsequent exterior experiment raises the expansion to order 128 and
tests admission from 60 kpc, retaining the original numerical tolerances.
That can support a new 60--80 kpc join, but the new joined potential has not
yet been audited. All old failures remain in the evidence tree.

## Independent checks

- exterior-verification-001: 36 execution snapshots checked; maximum fine-stencil scaled discrepancy 8.25772e-10; all registered verification checks passed: True.
- exterior-verification-002: 37 execution snapshots checked; maximum fine-stencil scaled discrepancy 8.18016e-10; all registered verification checks passed: True.

The verification uses checked execution snapshots, 80-digit recombination of
the stored source moment integrals, and fourth-order derivative stencils at
every registered exterior point. The sum check is not another quadrature
method. Symbolic synthetic tests separately differentiate a nonspherical
joined potential through third order, including axis and reflection cases.

## Next requirements

Construct and audit the revised matched potential with denser coverage across
the source taper and transition; preserve errors at every point. Validate a
production representation derived from one C3 potential, then use its field
and derivatives in the complete action flux and a separate Poisson solve.
Only after those numerical checks can we repeat the fixed-parameter galaxy
comparison. Direct outer-star dynamics, lensing, full Solar System predictions,
stability and untouched confirmation remain open scientific requirements.

The goal remains active. These numerical results do not alter the earlier
conditional cluster, galaxy or Solar System comparison counts.

## Result hashes

- `exterior-moment-001`: `744eeb06e59c38cbd0c7421092d81e98f813671db94c689c26c3eaa45c834a14`
- `exterior-verification-001`: `6a71f25b06a225edfb5648522a87c702dd8ccf6561d3a6c404094863e5a47c91`
- `potential-join-001`: `5e5563f0e21d544db3e43da01927f2857cf2eb97536e03180694489924d0c1df`
- `exterior-moment-002`: `284c32202b0c4950ba6b9320fdf915497023bb5616aba076ef822d793f1e7920`
- `exterior-verification-002`: `9410f3bb9a129315f74a26e9931a8aae58fbf37088b5f549c6f8b2943fefe9bd`
