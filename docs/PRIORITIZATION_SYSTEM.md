# Sigma formula discovery and prioritization system

## Purpose

This is a falsification-first work allocator for a finite gravity-action grammar. It exhaustively
enumerates the declared grammar, applies cheap hard gates before expensive ones, retains every
survivor losslessly, learns only admissible theory-side lessons from the Sigma archive, and creates
a multi-objective queue for subsequent derivations.

"Promising" means worth spending the next derivation or simulation budget on. It does not mean
probably true. Historical success cannot rescue a hard-gate failure, and Pareto position is not a
truth probability.

## Current measured result

- Frozen grammar: 50 basis terms, 1--6 distinct terms, coefficients `{-epsilon,+epsilon}`.
- Exhaustive actions: 1,088,651,720.
- Generator v2 sampled-static survivors: 17,540,440.
- Lossless export: 260 SHA-256-committed blocks, 420,982,000 bytes.
- Export audit: all 17,540,440 records passed count, format, order, size, and hash checks.
- RTX 5090 dense-static gate: 343 field/state points in 2.3 seconds.
- Dense-static rejections: 12,642,541.
- Dense-static passes: 4,897,899.
- Floating-point ambiguous: 0, using a `1e-10` guard around tolerance `1e-9`.
- Independent CPU/GPU check: 508/508 witnesses agree; all status hashes pass.
- Dense survivors: 36,047 structural families.
- Current discovery queue: 124 representatives in the first eight Pareto layers.

The 343-point test is sampled evidence, not a proof of global convexity.

## Evidence archive

The provenance database contains 2,957 test functions, 13,756 assertions, 743 protocols, 1,725
results, 546 research notes, 20,682 path/hash edges, and 128 historically named formulas. Of 446
normalized lessons, only 157 are admitted to priority signals. Sixteen mixed theory-side signals are
flagged for review; they are not automatically contradictions. Every ingested source is hashed, and
the source repository commit plus dirty paths are recorded.

## Evidence boundary

These inputs have zero priority weight and are retained only for historical audit:

- dark matter, invisible halos, NFW, or per-object halo fits;
- redshift-derived distance or cosmological position;
- supernova distance modulus or supernova-calibrated distance;
- GR/NFW-derived lensing acceleration presented as a direct measurement;
- observational fit quality used as a substitute for action health.

Redshift may later enter only as the directly measured wavelength ratio. Raw angles, detector
positions, spectra, Doppler measurements, clocks, ephemerides, and independently audited baryonic
inputs may enter only their designated measurement gates.

## Gate order

1. Grammar identity, dimensions, one universal law, and no private object parameters.
2. Newtonian/vacuum asymptotics and static convexity.
3. Kinetic rank, Hamiltonian boundedness, constraints, and physical degrees of freedom.
4. Principal symbols, hyperbolicity, characteristic cones, and covariance identities.
5. GR golden controls and Solar-System ephemeris/light/clock controls.
6. Audited galaxy, cluster, and raw-lensing measurements, followed by untouched holdouts.
7. Provenance completeness and reproducibility.

Every hard-gate rejection is terminal for that exact action.

## Use of historical tests

The archive never narrows the exhaustive grammar. After enumeration it identifies tested mechanism
families, carries forward known theoretical rejection burden, links generated families to prior
failure modes, exposes mixed signals, and orders expensive follow-up among surviving candidates.

Empirical composites, convenience switches, per-object fits, lensing-only closures, published
controls, explicit failures, and verdicts reporting worsening or null results are excluded from the
historical discovery queue.

## Pareto queue

There is no opaque total score. Non-dominated layers use parsimony, flux/gradient/measured-state
coverage, high-field robustness, and admissible theory-side pass/rejection history for matching
mechanisms. One-term corrections are retained as controls but separated from discovery candidates.
Every family keeps its exact survivor count and a deterministic representative ordinal, term list,
sign mask, and correction expression.

The downstream promotion registry has a separate deterministic dossier layer. It verifies the
pipeline hash, sampled-static lineage, every evaluator input/result/output hash, and all candidate
stage identities before explaining the first rejection or unresolved blocker. Terminally rejected
candidates never enter its work queue. Remaining candidates are Pareto-layered by exact gate depth,
exact evidence count, and term-count parsimony; these axes allocate follow-up derivation effort and
are explicitly not probabilities that a theory is true.

## Reproduce

Build Rust Generator v2, configure CuPy for CUDA, then run from the project root:

```powershell
scripts\run_priority_pipeline.ps1 `
  -Repo <path-to-source-repository>
```

The complete workflow takes about three minutes on this workstation. The script rebuilds the
knowledge graph, traverses the billion-action grammar, exports and audits every survivor, runs the
GPU gate, cross-checks it on CPU, and rebuilds the dense-survivor queue.

## Principal outputs

- `runs/knowledge-base/evidence.sqlite`: normalized provenance/evidence graph.
- `runs/knowledge-base/summary.json`: ingestion accounting and archive fingerprint.
- `runs/knowledge-base/formula-priority.json`: historical-formula work cues.
- `runs/generator-v2/billion-survivor-export.json`: exhaustive export manifest.
- `runs/generator-v2/billion-survivor-audit.json`: independent export audit.
- `runs/generator-v2/billion-dense-static-gpu.json`: dense GPU decisions.
- `runs/generator-v2/billion-dense-static-crosscheck.json`: CPU/hash cross-check.
- `runs/knowledge-base/generated-priority-dense.json`: current generated-family queue.

## Remaining theory gates

The 4.9 million dense-static survivors are not complete theories. The next tier must automate or
semi-automate covariant lifts, action variation, ADM/Dirac constraints, physical degree counts,
Hamiltonian checks, and characteristics. Only survivors should run through GR and Solar-System
golden controls. Galaxy, cluster, and raw-lensing tests come afterward and must use audited direct
observables under the evidence policy.
