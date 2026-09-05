# P0698 local-vector-coherence mathematical audit

Frozen before synthetic metrics: 2026-08-02

Verdict: **all preregistered mathematical gates pass**; advance the exact
source-level operator to one separately frozen spent DDO154/RX J2129 screen

## Audited first-principles quantity

For baryonic source cells `j`, P0698 separates the ordinary Newtonian vector
sum from the amount of vector strength that was available before cancellation:

\[
\mathbf g_{N,{\rm direct}}(\mathbf x_i)
=\sum_{j\ne i}{Gm_j(\mathbf x_j-\mathbf x_i)\over
|\mathbf x_j-\mathbf x_i|^3},
\]

\[
A_N(\mathbf x_i)=\sum_{j\ne i}{Gm_j\over|\mathbf x_j-\mathbf x_i|^2},
\qquad
\mathcal C(\mathbf x_i)={|\mathbf g_{N,{\rm direct}}|\over A_N}.
\]

The triangle inequality gives `0 <= C <= 1`.  No softening length, smoothing
scale, exponent, threshold, or fitted normalization is used; the self pair is
zero in both sums.

To avoid the gauge dependence of multiplying potentials by a spatially
varying field, P0698 gates equation sources:

\[
S_{\rm base}=\mathcal C S_{\rm coh}+(1-\mathcal C)S_{\rm local},
\]

then solves one Poisson equation using the coherent potential as the finite
domain boundary.  The unchanged projected routing correction is added only
afterward:

\[
\Phi_{\rm joint}=\Phi_{\rm base}
+e_{2D}(\Phi_{\rm route}-\Phi_{\rm local}).
\]

## Frozen result

| Metric | Result | Gate | Verdict |
|---|---:|---:|---|
| full coherence range | `[4.0e-17, 0.9973]` | `[0,1]` | pass |
| raw triangle-inequality excess | `0` | `<=1e-12` | pass |
| single-center outer median coherence | `0.9758` | `>=0.9` | pass |
| equal two-center midpoint coherence | `5.37e-17` | `<=1e-12` | pass |
| two-center far-field median coherence | `0.9183` | `>=0.8` | pass |
| rotation coherence / potential / acceleration RMS | machine scale | `<=1e-10` | pass |
| two-cell translation coherence RMS | `2.99e-16` | `<=1e-5` | pass |
| 33-cubed to 49-cubed coherence RMS | `1.36e-4` | `<=0.08` | pass |
| coherent endpoint potential RMS | `7.78e-15` | `<=1e-10` | pass |
| maximum field residual | `1.46e-14` | `<=1e-10` | pass |
| maximum normalized curl | `1.02e-16` | `<=1e-10` | pass |
| base / routing boundary mismatch | `0 / 0` | `<=1e-14` | pass |
| high-acceleration error relative to Newtonian | `1.07e-7` | `<=1e-3` | pass |

The same dimensionless controller distinguishes the two geometries for which
it was designed.  It is nearly one around an isolated source, nearly zero at
the exact cancellation point between equal sources, and returns above 0.9 in
their common far field.  That distinction is stable under the tested grid
symmetries and resolution change.

The first run stopped before writing a report because the routing helper
requires positive extra source strength, which vanishes in the deliberately
Newtonian `a0 -> 0` control.  Commit `bd49e9bd` changed only that asymptotic
control to gate the coherent and Newtonian sources directly.  The candidate,
normal-acceleration fields, metrics, thresholds, and masks were unchanged.

## What this earns, and what it does not

P0698 earns one spent-data mechanism screen.  It does not demonstrate that
DDO154 is coherent where its measured rotation points matter or that RX J2129
is incoherent in the regions that create images.  Those are predictions of the
already frozen definition, not assumptions that may now be tuned.

The pairwise field is nonlocal, the finite-domain coherent boundary requires a
field-of-view robustness test, and no covariant action or causal propagation
law has been derived.  Even a spent joint pass would therefore precede—not
replace—robustness, Solar-System, relativistic-closure, and sealed-holdout
tests.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/run_p0698_local_vector_coherence_math_audit.py
python -m pytest tests/test_local_vector_coherence.py tests/test_coherent_monopole.py tests/test_source_routing_qumond.py tests/test_field_solvers.py -q
```

Artifacts are in `results/p0698_local_vector_coherence_math_audit/`.

## Claim boundary

No galaxy or cluster observation was loaded by P0698, and P0633/P0640 remain
sealed.
