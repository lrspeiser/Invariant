# Parallel execution profile

## Measured machine

- Intel Core Ultra 9 285K: 24 physical cores and 24 logical processors.
- 95.71 GiB physical RAM; 78.32 GiB was free during measurement.
- NVIDIA GeForce RTX 5090: 32,607 MiB VRAM and CUDA compute capability 12.0.

## Exact symbolic benchmark

The benchmark workload is one independent exact quartic-Horndeski 11-by-11 local principal
extraction plus its 22-by-22 generalized first-order pencil certificate. Every job rederived and
passed the certificate.

| Workers | Wall time for the same number of jobs | Jobs/second |
|---:|---:|---:|
| 1 | 8.442 s | 0.1185 |
| 8 | 9.051 s | 0.8839 |
| 12 | 9.750 s | 1.2307 |
| 16 | 10.503 s | 1.5234 |
| 20 | 11.141 s | 1.7951 |
| 22 | 12.687 s | 1.7341 |
| 24 | 13.545 s | 1.7718 |

Throughput peaks at 20 for this workload. The sustained production default is 16 symbolic workers,
leaving eight cores for Windows, SQLite/WAL, the GPU feeder, and light campaign work. Twenty is the
measured useful ceiling, not a universal guarantee for more memory-intensive Cadabra or simulation
jobs.

## GPU benchmark

The existing dense-static RTX 5090 run processed 17,540,440 candidates on 343 grid points each in
2.317 seconds. This is 7.57 million candidates per second, or 2.60 billion candidate-grid
evaluations per second. GPU work should use one owning process with large batches; multiple Python
GPU processes would duplicate contexts and VRAM without increasing useful throughput.

## Production lanes

- CPU symbolic: 16 sustained, 20 measured maximum useful.
- GPU dense: one process, millions of candidates per batch.
- LLM research: four concurrent calls initially. At the configured $2 reservation cap, at most $8
  is reserved simultaneously; the existing atomic $500 aggregate ledger remains authoritative.
- Database/policy/dossier housekeeping: up to four mostly light workers.

These counts are lane caps, not numbers to add blindly. A 20-worker symbolic peak should not run
beside another CPU-heavy simulation pool.

## Task-type isolation

Campaign workers now accept a task allowlist:

```powershell
python -m sigma_theory_compiler.campaign_cli run `
  --database runs/campaigns/campaign-v1-live.sqlite `
  --worker-id symbolic-01 `
  --duration 6h --follow `
  --task-types covariant_lift,symbolic_proxy,constraint_analysis,formal_reference_controls
```

The claim is enforced transactionally when a task is leased. A symbolic worker therefore cannot
consume an LLM task even if the LLM task has higher priority.

Launch a bounded filtered pool with:

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/run_campaign_pool.py `
  --database runs/campaigns/campaign-v1-live.sqlite `
  --workers 16 `
  --task-types covariant_lift,symbolic_proxy,constraint_analysis,formal_reference_controls `
  --duration 6h `
  --log-directory runs/campaigns/worker-logs
```

The launcher refuses more than the measured ceiling of 20, requires an explicit task-type lane,
runs workers without visible console windows on Windows, and writes separate stdout/stderr logs.

The measured artifacts are under `runs/benchmarks/parallel-symbolic-v1.json` through `v3.json`; the
machine-readable production caps are in `configs/resource_profile_5090.json`.
