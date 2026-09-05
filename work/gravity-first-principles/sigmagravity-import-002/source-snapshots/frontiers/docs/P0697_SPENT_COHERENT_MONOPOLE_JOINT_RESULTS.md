# P0697 spent coherent-monopole joint screen

Frozen before candidate scores: 2026-08-02

Verdict: **fails jointly**; the coherent monopole is the strongest spent
DDO 154 galaxy result in this branch, but replacing the cluster's local base
destroys the previously successful RX J2129 image topology

## Exact candidate

P0697 combines the P0696 coherent-monopole potential with the unchanged
parameter-free projected routing correction:

\[
\Phi_{\rm joint}=\Phi_{\rm coh}
+e_{2D}(\Phi_{\rm route}-\Phi_{\rm local}),
\qquad
e_{2D}=1-\frac{\lambda_{\min}(C_{xy})}{\lambda_{\max}(C_{xy})}.
\]

There is no fitted gravity amplitude, photon amplitude, shell width,
interpolation exponent, routing fraction, or per-object gravity setting.  The
same equation, `a0`, `G`, and routing constants are used for the galaxy and
cluster.  The four bounded cluster nuisance parameters only align the fixed
field to the registered images.

## Joint result

| System | Metric | Candidate | Comparator or gate | Verdict |
|---|---|---:|---:|---|
| DDO154 | ordinary RMSE | `1.887 km/s` | algebraic MOND `2.916` | pass (`0.647x`) |
| DDO154 | weighted RMSE | `1.371 km/s` | algebraic MOND `1.226` | pass (`1.118x`) |
| DDO154 | ordinary RMSE | `1.887 km/s` | 3D QUMOND `3.936` | pass (`0.479x`) |
| DDO154 | mean bias | `+0.547 km/s` | absolute `<=3` | pass |
| RX J2129 | median physical deflection | `6.952 arcsec` | `1-20` | pass |
| RX J2129 | training / heldout roots | `14/15`, `6/7` | exact coverage | fail |
| RX J2129 | training / heldout RMS | undefined | every root required | fail |
| RX J2129 | missing-multiplicity families | `4/7` | `0` | fail |
| RX J2129 | exact or demagnified-only families | `3/7` | at least `5/7` | fail |
| RX J2129 | parity-diverse families | `4/7` | `7/7` | fail |
| RX J2129 | critical-curve families | `6/7` | `7/7` | fail |
| RX J2129 | nuisance parameters near bounds | `2` | `0` | fail |

All numerical residual, finite-field, potential-identity, boundary, curl,
parameter-accounting, and galaxy gates pass.  Ten cluster fit/topology gates
fail, so the exact equation is retired and does not advance to robustness,
Solar-System, or sealed tests.

## What changed between the galaxy and cluster

P0693 used

\[
\Phi_{0693}=\Phi_{\rm local}
+e_{2D}(\Phi_{\rm route}-\Phi_{\rm local}),
\]

whereas P0697 changes only the base:

\[
\Phi_{0697}=\Phi_{\rm coh}
+e_{2D}(\Phi_{\rm route}-\Phi_{\rm local}).
\]

That replacement cuts the DDO154 ordinary error by 52% (`3.943` to `1.887
km/s`) but cuts the RX J2129 median strong-lens deflection by 45% (`12.64` to
`6.95 arcsec`).  The cluster fit responds by pushing both shear components to
their bounds, yet still loses image multiplicity.  The failure is therefore
not a bad projected routing fraction: the unchanged correction is sitting on
a base field that is too globally monopolar and too weakly multi-centered.

The result isolates two useful mechanisms:

1. A shell-coherent radial completion can outperform both the existing fixed
   MOND comparator and 3D QUMOND on the spent gas-rich dwarf without a fitted
   galaxy parameter.
2. RX J2129 needs a base that retains local, multi-center field structure in
   addition to the routed topology correction.  One baryonic centroid and one
   global shell average erase too much of that structure.

## Next hypothesis generator: local vector coherence

The next candidate must choose between these mechanisms from the baryonic
field itself, not from an object label or a tuned cluster threshold.  A
parameter-free local coherence fraction follows directly from vector
addition:

\[
\mathcal C(\mathbf x)=
\frac{|\sum_j \mathbf g_{N,j}(\mathbf x)|}
{\sum_j |\mathbf g_{N,j}(\mathbf x)|},
\qquad 0\le\mathcal C\le1.
\]

The denominator is the scalar `1/r^2` convolution of baryonic source mass; the
numerator is the magnitude of the ordinary summed Newtonian vector.  An
isolated, directionally aligned source approaches one.  Conflicting source
directions in a multi-center field drive the fraction downward without a
fitted scale.

A candidate scalar potential is

\[
\Phi_{\rm base}=\Phi_{\rm local}
+\mathcal C(\mathbf x)[\Phi_{\rm coh}-\Phi_{\rm local}],
\qquad
\Phi_{\rm joint}=\Phi_{\rm base}
+e_{2D}(\Phi_{\rm route}-\Phi_{\rm local}).
\]

This is not yet an empirical candidate.  Multiplying potentials by a spatially
varying coherence field adds gradient terms and could produce artifacts.  It
must first pass a frozen no-observation audit of the `0-1` bound, spherical and
multi-center limits, rotations, translations, resolution, curl, strong-field
behavior, and boundary behavior.  Only then may it be exposed to DDO154 and RX
J2129.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/run_p0697_spent_coherent_monopole_joint_screen.py
python -m pytest tests/test_coherent_monopole.py tests/test_source_routing_qumond.py tests/test_p0635_ddo154_map_commissioning.py -q
```

Artifacts are in `results/p0697_spent_coherent_monopole_joint_screen/`.

## Claim boundary

Both systems are spent mechanism-development evidence, and the zero-slip
photon closure is not a relativistic metric theory.  P0697 is evidence for a
useful galaxy mechanism and against one global-centroid unifier—not validation
of new gravity.  P0633 and P0640 remain sealed.
