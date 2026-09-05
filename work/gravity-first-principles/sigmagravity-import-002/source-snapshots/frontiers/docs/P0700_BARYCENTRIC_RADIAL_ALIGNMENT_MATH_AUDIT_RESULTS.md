# P0700 barycentric radial-alignment mathematical audit

Frozen before synthetic metrics: 2026-08-02

Verdict: **all preregistered mathematical gates pass**; advance the exact
source controller to one separately frozen spent DDO154/RX J2129 screen

## Audited controller

P0700 asks whether the already-summed Newtonian field points toward the
baryonic centroid:

\[
\mathbf x_c={\int\rho_b\mathbf x\,dV\over\int\rho_b\,dV},
\qquad
\hat{\mathbf r}={\mathbf x-\mathbf x_c\over|\mathbf x-\mathbf x_c|},
\]

\[
\mathcal A_r(\mathbf x)=
{\max[0,-\mathbf g_N(\mathbf x)\cdot\hat{\mathbf r}]
\over|\mathbf g_N(\mathbf x)|}.
\]

`A_r` is zero at the center or a zero field, exactly one for an inward radial
field, and zero for an outward or tangential field.  No exponent, threshold,
angle, smoothing scale, offset, or multiplier is introduced.

The gauge-safe field construction is unchanged:

\[
S_{\rm base}=\mathcal A_r S_{\rm coh}+(1-\mathcal A_r)S_{\rm local},
\]

\[
\Phi_{\rm joint}=\Phi_{\rm base}
+e_{2D}(\Phi_{\rm route}-\Phi_{\rm local}).
\]

## Frozen result

| Metric | Result | Gate | Verdict |
|---|---:|---:|---|
| alignment range | `[0,1]` | `[0,1]` | pass |
| explicit inward radial median | `1` | effectively `1` | pass |
| explicit outward / tangential maximum | `0 / 0` | `<=1e-14` | pass |
| center value | `0` | `<=1e-14` | pass |
| isolated-source outer median | `0.9999996` | `>=0.98` | pass |
| two-center structure cells below 0.9 | `30.87%` | `>=5%` | pass |
| rotation alignment / potential / acceleration RMS | machine scale | `<=1e-10` | pass |
| two-cell translation alignment RMS | `5.99e-8` | `<=1e-5` | pass |
| 33-cubed to 49-cubed alignment RMS | `2.75e-6` | `<=0.05` | pass |
| coherent endpoint potential RMS | `7.78e-15` | `<=1e-10` | pass |
| maximum field residual | `1.58e-14` | `<=1e-10` | pass |
| maximum normalized curl | `1.02e-16` | `<=1e-10` | pass |
| base / routing boundary mismatch | `0 / 0` | `<=1e-14` | pass |
| high-acceleration error relative to Newtonian | `1.23e-7` | `<=1e-3` | pass |

The controller fixes the conceptual granularity error exposed by P0699: an
isolated extended source remains radially coherent even though its individual
source-cell contributions can oppose one another.  The equal two-center field
still produces a sizable non-radial region, so the controller is not merely a
constant-one selector.

## What this earns, and what it does not

P0700 earns one spent mechanism screen with the unchanged galaxy and cluster
thresholds.  It does not show that the DDO154 disk is sufficiently aligned or
that RX J2129 substructure reduces alignment in the lens-producing cells.
Those are predictions that may not be transformed after the next run.

The barycentric direction is global and nonlocal, and the field has not been
derived from a covariant action or causal propagation law.  A merger with two
comparable baryonic centers may not admit one physically meaningful global
center.  Field-of-view, centering, multi-center, and Solar-System robustness
remain mandatory even if the spent joint screen passes.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/run_p0700_barycentric_radial_alignment_math_audit.py
python -m pytest tests/test_barycentric_radial_alignment.py tests/test_local_vector_coherence.py tests/test_coherent_monopole.py
```

Artifacts are in `results/p0700_barycentric_radial_alignment_math_audit/`.

## Claim boundary

No observational score was computed, and P0633/P0640 remain sealed.
