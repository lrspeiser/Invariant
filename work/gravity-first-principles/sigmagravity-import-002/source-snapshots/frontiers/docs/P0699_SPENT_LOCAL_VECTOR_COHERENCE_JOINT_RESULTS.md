# P0699 spent local-vector-coherence joint screen

Frozen before candidate scores: 2026-08-02

Verdict: **fails jointly**, but produces the closest universal-setting raw
RX J2129 lens result in this branch while revealing that cell-by-cell source
coherence is the wrong granularity for an extended disk

## Exact candidate

P0699 uses the P0698 source-level equation without a transform or fitted
parameter:

\[
\mathcal C(\mathbf x)=
{\left|\sum_j\mathbf g_{N,j}(\mathbf x)\right|\over
\sum_j|\mathbf g_{N,j}(\mathbf x)|},
\]

\[
S_{\rm base}=\mathcal C S_{\rm coh}+(1-\mathcal C)S_{\rm local},
\qquad
\Phi_{\rm joint}=\Phi_{\rm base}
+e_{2D}(\Phi_{\rm route}-\Phi_{\rm local}).
\]

The same equation and constants are used for DDO154 and RX J2129.  There is no
per-object gravity setting, coherence threshold, exponent, smoothing scale,
gravity amplitude, photon amplitude, or fitted routing fraction.

## Joint result

| System | Metric | Candidate | Comparator or gate | Verdict |
|---|---|---:|---:|---|
| DDO154 | ordinary RMSE | `4.551 km/s` | algebraic MOND `2.916` | fail (`1.561x`) |
| DDO154 | weighted RMSE | `2.900 km/s` | algebraic MOND `1.226` | fail (`2.366x`) |
| DDO154 | ordinary RMSE | `4.551 km/s` | 3D QUMOND `3.936` | fail (`1.156x`) |
| DDO154 | mean bias | `+3.210 km/s` | absolute `<=3` | fail |
| RX J2129 | median physical deflection | `10.90 arcsec` | `1-20` | pass |
| RX J2129 | training / heldout roots | `15/15`, `7/7` | exact coverage | pass |
| RX J2129 | training / heldout RMS | `0.674 / 2.481 arcsec` | each `<=3` | pass |
| RX J2129 | heldout / compact halo | `0.978x` | `<=1.25x` | pass |
| RX J2129 | missing-multiplicity families | `1/7` | `0` | fail |
| RX J2129 | exact families | `6/7` | at least `5/7` | pass |
| RX J2129 | observable-surplus families | `0/7` | at most `2/7` | pass |
| RX J2129 | parity-diverse / critical families | `7/7`, `7/7` | `7/7`, `7/7` | pass |
| RX J2129 | nuisance parameters near bounds | `1` | `0` | fail |

All coherence, triangle-inequality, numerical residual, source identity,
boundary, curl, finite-field, parameter-accounting, root-convergence,
positional, compact-halo, parity, and critical-curve gates pass.  Four galaxy
gates, one multiplicity gate, and the nuisance-bound gate fail.  The exact
candidate is retired and does not advance.

## Strong cluster result, limited claim

The `2.481 arcsec` heldout error is 2.2% lower than the registered
object-specific compact-halo comparator (`2.536 arcsec`) and improves on the
P0693 parameter-free routing result (`2.670 arcsec`).  Six families have exact
global multiplicity, all seven have both parities and critical curves, and no
surplus images appear.  Family 3 still has only three roots for four observed
images, and one shear parameter reaches its bound, so this is not a topology
pass.

Because RX J2129 is fully spent and the formula was generated through repeated
work on this object, the result is mechanism evidence only—not independent
validation and not evidence against dark matter.

## Why the galaxy regressed

The frozen coherence diagnostic made a surprising but decisive prediction:

| Region | Median coherence |
|---|---:|
| DDO154 midplane over its twelve score radii | `0.337` |
| RX J2129 strong-lens annulus over all line-of-sight cells | `0.469` |

An extended disk contains source cells on all sides of a field point.  Their
individual vectors partially cancel even when the **net** field points cleanly
toward the galaxy center.  Cell-by-cell coherence therefore classifies the
disk as less coherent than the cluster lens region and suppresses the
successful P0697 coherent completion.  The failure is not repairable by a
post-hoc transform of `C`; P0699 explicitly forbids that.

## Next first-principles generator: barycentric radial alignment

The physically relevant disk property is not whether every source-cell vector
agrees, but whether the **summed field** agrees with the inward direction
defined by the baryonic center:

\[
\mathcal A_r(\mathbf x)=
\frac{\max[0,-\mathbf g_N(\mathbf x)\cdot\hat{\mathbf r}]}{|mathbf g_N(\mathbf x)|},
\qquad
\hat{\mathbf r}={\mathbf x-\mathbf x_c\over|\mathbf x-\mathbf x_c|}.
\]

This is parameter-free and bounded from zero to one.  A disk whose net field
is globally inward approaches one even when its individual mass-cell vectors
cancel.  A multi-center cluster can fall below one wherever local substructure
pulls away from the global barycentric direction.  It can gate the same
source-level equation without multiplying gauge-dependent potentials.

This is only a generator.  It must first pass a no-observation audit of the
spherical limit, two-center off-axis structure, rotations, translations,
resolution, center behavior, field residuals, curl, boundaries, and strong
gravity.  No galaxy or cluster score may be read during that audit.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/run_p0699_spent_local_vector_coherence_joint_screen.py
python -m pytest tests/test_local_vector_coherence.py tests/test_coherent_monopole.py tests/test_source_routing_qumond.py tests/test_p0635_ddo154_map_commissioning.py -q
```

Artifacts are in `results/p0699_spent_local_vector_coherence_joint_screen/`.

## Claim boundary

P0699 uses spent development systems and a diagnostic zero-slip photon rule.
P0633 and P0640 remain sealed.
