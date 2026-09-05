# P0696 coherent-monopole mathematical audit

Frozen before synthetic metrics: 2026-08-02

Verdict: **all preregistered mathematical gates pass**; advance the exact
operator to one separately frozen spent DDO 154/RX J2129 joint screen

## Audited operator

At the baryonic center, assign every native-grid cell to its nearest radial
shell.  The coherent Newtonian acceleration is the mean inward Newtonian field
on that shell:

\[
g_{N,0}(r_n)=\left\langle-\mathbf g_N\cdot\hat{\mathbf r}\right\rangle_n.
\]

Complete only that shared monopole with the simple algebraic low-acceleration
relation and integrate the difference into a scalar potential:

\[
g_{M,0}=\frac{g_{N,0}+\sqrt{g_{N,0}^2+4a_0g_{N,0}}}{2},
\qquad
\Delta\Phi_0(r)=\int_0^r[g_{M,0}(s)-g_{N,0}(s)]\,ds,
\]

\[
\Phi_{\rm coh}(\mathbf x)=\Phi_N(\mathbf x)+\Delta\Phi_0(|\mathbf x-\mathbf x_c|).
\]

The cluster-topology candidate remains a zero-boundary addition:

\[
\Phi_{\rm joint}=\Phi_{\rm coh}
+e_{2D}(\Phi_{\rm route}-\Phi_{\rm local}).
\]

This has no new fitted constant, smoothing scale, shell width, exponent, or
per-object adjustment.  Newtonian bars, clumps, and multipoles remain in
`Phi_N`; only the shell-shared inward component receives the completion.

## Frozen result

| Metric | Result | Gate | Verdict |
|---|---:|---:|---|
| spherical shell-mean radial relative RMS | `0.01659` | `<=0.03` | pass |
| spherical median absolute relative error | `0.001458` | `<=0.02` | pass |
| correction tangential/radial RMS | `0.002631` | `<=0.03` | pass |
| maximum correction angular scatter | `0.01151` | `<=0.05` | pass |
| high-acceleration correction/Newtonian RMS | `1.217e-7` | `<=1e-4` | pass |
| rotation potential relative RMS | `4.745e-16` | `<=1e-10` | pass |
| rotation acceleration relative RMS | `1.662e-15` | `<=1e-10` | pass |
| maximum normalized curl | `1.221e-16` | `<=1e-10` | pass |
| coherent / hybrid potential identity error | `0 / 0` | `<=1e-14` | pass |
| routing-correction boundary mismatch | `0` | `<=1e-14` | pass |

The coherent construction fixes the numerical defect that retired the
straight-ray version: its tangential leakage is about 25 times smaller than
the cubic ray implementation (`0.00263` versus `0.0661`) while retaining the
same broad physical motivation.  Its spherical shell means agree with the
declared completion at the 1.7% RMS level, and its strong-field correction is
only about 0.000012% of Newtonian gravity in the audit.

## Diagnostic correction record

The first execution incorrectly evaluated the strong-field diagnostic as
`RMS(correction - Newtonian) / RMS(Newtonian)`, which tends to one when the
correction correctly vanishes.  Commit `1e4337a3` changed only that diagnostic
to `RMS(correction) / RMS(Newtonian)`, matching the frozen gate's stated
quantity.  The equation, fields, grid, masks, thresholds, and every other
metric were unchanged.  The report hashes the corrected runner.

## What this earns, and what it does not

P0696 earns one spent-data observational screen.  It does not establish a
relativistic theory or an empirical success.  Shell averaging about a global
baryonic centroid is nonlocal and has not been derived from a covariant action
or a causal evolution equation.  A multi-center cluster could also make a
single global monopole physically inappropriate; the spent cluster topology
test is therefore especially discriminating.

P0633 galaxy kinematics and P0640 cluster lensing remain sealed.  They may be
opened only after the candidate passes the spent joint screen plus the already
declared robustness and Solar-System gates.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/run_p0696_coherent_monopole_math_audit.py
python -m pytest tests/test_coherent_monopole.py tests/test_source_routing_qumond.py tests/test_field_solvers.py -q
```

Artifacts are in `results/p0696_coherent_monopole_math_audit/`.
