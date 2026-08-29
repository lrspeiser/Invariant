# Item 57 independent-galaxy gate result

## Decision

`ITEM57_LITTLE_THINGS_REPLICATES_NEGATIVE_THINGS_SOURCE_LIMITED_EXACT_REPRESENTATION_RETIRED_IN_TESTED_SCOPE`

The unchanged Item 45 formula failed on the authorized LITTLE THINGS exploration
pipeline. This was a broad replicated failure, not a one-galaxy veto: it lost to the
empirical RAR comparator in all 11 evaluable galaxies, every loss persisted through all
eight frozen baryonic and geometry variants, and neither leave-one-out influence nor the
frozen symmetric trim changed the aggregate sign.

This retires only candidate 135082's exact high-amplitude representation in the tested
SPARC plus LITTLE THINGS scope. It does not prune the geometry-density interaction family,
establish GR or dark matter as uniquely correct, or prevent lower-amplitude, differently
normalized, or physically derived descendants from being generated and tested.

## What was tested

The formula and all parameters were frozen in commit `4c5159cd` before predictor
acquisition:

```text
I = geometry * tanh(2*density)
H = 0.5 + 0.5*tanh(2*I)
nu = 1 + 6*u^(-0.6)/(1 + u/100)*(0.05 + 0.95*H)
Vpred = sqrt(Vbar^2*nu)
```

No coefficient was fitted to LITTLE THINGS. The test used the 11 exploration galaxies
already opened by Item 5; the five LITTLE THINGS confirmation galaxies remained sealed.
The 35 SPARC confirmation galaxies also remained sealed.

The baryonic predictor was reconstructed without reading or using any observed-velocity,
velocity-error, fitted-halo, or dark-matter-residual column:

- gas gravity came from the Iorio/LITTLE THINGS H I surface-density profile through a
  deterministic axisymmetric softened-annulus quadrature;
- stellar mass came from Hunter V-band absolute magnitude with frozen mass-to-light
  brackets;
- stellar gravity used the analytic exponential-disk solution and the published Hunter
  disk scale;
- effective radius was fixed to `1.67834699 Rd`;
- only target radii within the published H I density support were evaluated.

The two post-freeze source repairs changed only VizieR's table identifier and catalog name
aliases. They did not change the sample, formula, coefficients, physical reconstruction,
systematics, thresholds, or scores.

## Numerical result

All 11 authorized galaxies passed the minimum per-object row floor. Of 255 existing
published target rows, 199 were inside the response-blind H I surface-density support and
were evaluated.

| Model | Equal-galaxy standardized loss | Median galaxy loss |
|---|---:|---:|
| Item 45 geometry-density candidate | 225.1771 | 147.7000 |
| Newtonian reconstructed baryons | 50.9742 | 29.5256 |
| Empirical RAR | 15.8414 | 5.9297 |

The candidate loss was 4.42 times the Newtonian-baryon loss and 14.21 times the RAR loss.
It was 341.75% worse than Newtonian baryons under the equal-galaxy score. It beat Newton
for only DDO 52 and beat RAR for none of the 11 galaxies.

All eight frozen systematic variants also lost to RAR for every galaxy:

| Variant | Candidate loss | RAR loss | Candidate galaxy wins |
|---|---:|---:|---:|
| neutral gas factor 1.20 | 211.65 | 14.89 | 0/11 |
| neutral gas factor 1.52 | 238.28 | 16.92 | 0/11 |
| V-band M/L 0.30 | 204.29 | 14.11 | 0/11 |
| V-band M/L 0.80 | 253.70 | 18.92 | 0/11 |
| `Rd-e_Rd` | 219.25 | 16.36 | 0/11 |
| `Rd+e_Rd` | 230.51 | 15.40 | 0/11 |
| gas softening `0.05 Rd` | 229.69 | 16.31 | 0/11 |
| gas softening `0.20 Rd` | 216.83 | 15.11 | 0/11 |

The counterexample policy classified this as
`REPLICATED_NEGATIVE_EVIDENCE_TESTED_REPRESENTATION`. Counts were 11 raw, 11
quality-verified, and 11 uncertainty-resolved counterexamples. No single object was used
as a veto, counterexample count alone was not decisive, and the formula family was not
pruned.

## THINGS lane

The primary de Blok et al. release contains 19 rotation curves as PostScript figures but no
machine-readable radial table. A 2026 Zenodo compilation with machine-readable
transcriptions was located, but its downloadable archive also contains response rows for
the sealed SPARC confirmation objects. No verified selective per-galaxy endpoint or safe
byte-range path was established before the freeze, so the archive was not downloaded and
no THINGS numeric claim was made.

The THINGS result is therefore
`SOURCE_FORMAT_LIMITATION_RETAINED_NO_NUMERIC_CLAIM`. This means the full two-pipeline
numeric gate did not pass, even though the LITTLE THINGS independent replication was
completed. The THINGS lane remains a data-procurement follow-up rather than being silently
replaced with digitized pixels or same-object SPARC velocities.

## Quality and interpretation

The numerical disk integrator is tested against the known analytic exponential-disk
rotation curve and agrees within the frozen 8% numerical tolerance. Its gas and stellar
contributions also pass exact mass-scaling tests.

The main remaining data caveats are that the rotation targets and H I surface-density
profiles are separate reductions of the same survey rather than instrument-independent
observations; published velocity errors lack a complete distance, inclination, beam, and
radial covariance matrix; and the stellar reconstruction assumes an exponential V-band
disk. Those limitations are why object-level records and the broader mechanism family are
retained.

The practical lesson is narrow but useful: the exact `amplitude=6`, `u^-0.6` response
over-enhances the inferred baryonic velocity on both SPARC and LITTLE THINGS. Future search
should learn from that failed region, not erase the underlying ideas of geometry-dependent
gating, nonlocality, memory, resonance, boundary response, or variable gravitational
strength.

## Reproduction

```powershell
python -m sigma_theory_compiler.gravity_item57_independent_galaxy_gate replay
python -m pytest tests/test_gravity_item57_independent_galaxy_gate.py -q
```

The replay regenerates and byte-compares the preflight, THINGS source audit, LITTLE THINGS
evaluation, and aggregate result. Paid model calls: zero. GPU use: none.
