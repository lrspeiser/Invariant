# Persistent Gravity Formula Engine

The persistent engine turns the deterministic formula generator and sampled-static screen into
a bounded local service. It is intended for searches that run for hours or days without losing
their queue, cursor, seeds, results, or provenance when a worker or the controlling terminal
stops.

It does **not** make a formula a physical theory. A passing result has survived only the declared
343-point sampled-static convexity screen. Covariant health, degrees of freedom, Solar-System
controls, galaxy observables, and independent validation remain separate gates.

## Safety and evidence contract

- Paid LLM calls are disabled and reported as zero. There is no API-key integration in this
  service.
- Observational data, dark-matter/halo inputs, and redshift-distance inputs must all remain
  disabled in the adapter. Startup fails before creating the search database if that contract is
  changed.
- Queue size, task count, wall time, service-disk use, CUDA batch size, retry count, lease time,
  process restarts, and JSONL telemetry bytes are bounded.
- One process owns CUDA. CPU workers do not duplicate CUDA state.
- Lane `utilization` means sampled **worker-lease occupancy**. The separate `hardware` record uses
  optional host CPU and NVML sensors to report instantaneous CPU, CUDA-core, memory-controller,
  VRAM, and power readings; unavailable sensors are labeled unavailable rather than inferred.
  `queue_starved` means a lane currently has no queued work;
  `workers_not_claiming` means work is queued but not all planned lane owners hold leases;
  `gpu:batch_underfilled` means queued GPU candidate count is below the planned batch.
- Results bind the generator config, source manifest/block where applicable, deterministic cursor,
  exact counts, and status-root hash. Binary blocks are size/SHA/header/accounting checked and
  sampled against the ordinal decoder.

## Start a service

Run from the repository root. `real` decodes a finite ordinal interval. `binary` consumes existing
hash-bound `SGSURV2` Rust survivor blocks.

```powershell
$env:PYTHONPATH = "src"
python -m sigma_theory_compiler.cli engine-start `
  --service-dir runs/engine/real-search `
  --mode real `
  --adapter-config configs/real_formula_execution_5090.json `
  --maximum-tasks 1000000 `
  --maximum-wall-seconds 1209600 `
  --maximum-disk-bytes 68719476736
```

The default starts a detached local worker and returns its PID. Add `--foreground` for a terminal
owned run, CI, or debugging. The execution and resource defaults are
`configs/persistent_parallel_search_5090.json` and `configs/resource_profile_5090.json`.

The service directory contains:

- `engine.sqlite`: durable queue, leases, results, events, checkpoints, and generator cursor;
- `execution-config.json`, `resource-profile.json`, `adapter-config.json`: immutable hash-bound
  startup snapshots;
- `telemetry.jsonl`: periodic execution/process snapshots;
- `status-summary.json` and `dashboard.html`: current queue, lane occupancy, starvation reasons,
  cursor, budget, and disk use;
- `last-run.json`: supervisor stop reason, restart/recovery counts, and utilization samples;
- `service.json`: lifecycle identity and PID; and
- `service.log`: detached-worker output.

Automatic refill runs at the configured refill interval while CPU workers and the single GPU owner
remain alive. This avoids recompiling the CUDA kernel or retransferring the frozen Hessian matrix
between bounded queue waves.

Worker occupancy and silicon utilization answer different questions. A `gpu` lane at utilization
`1.0` means its owner holds a batch continuously; a low simultaneous NVML percentage can still
expose host-side formula decoding between short kernels. The `real` ordinal adapter retains this
Python decoding cost. The verified `binary` adapter removes it by consuming Rust-produced
`SGSURV2` blocks and is therefore the preferred high-throughput path once a bounded, hash-verified
survivor corpus exists. CPU lanes should perform genuinely independent symbolic/formal work, not
duplicate a GPU screen merely to raise Task Manager percentages.

## Status, clean stop, and resume

```powershell
python -m sigma_theory_compiler.cli engine-status --service-dir runs/engine/real-search
python -m sigma_theory_compiler.cli engine-stop --service-dir runs/engine/real-search
python -m sigma_theory_compiler.cli engine-resume --service-dir runs/engine/real-search
```

`engine-stop` writes `stop.request`; it does not kill worker processes. The supervisor stops
admitting work, gives workers the configured shutdown grace period, checkpoints, and leaves queued
work and the source cursor intact. A worker still evaluating after that hard grace bound is
terminated; its lease is recovered on resume rather than treated as a result. `engine-resume`
removes the request and uses the same database,
deterministic seeds, attempts, and deadline. Expired leases from a crashed process are requeued up
to the configured attempt limit. The original execution deadline remains authoritative across
resumes.

Terminal stop reasons include `queue_drained`, `external_stop_requested`,
`task_budget_exhausted`, `execution_deadline_reached`, `disk_budget_exhausted`,
`run_wall_time_reached`, `no_workers_available`, and a fail-closed `refill_failed:...` reason.
`worker_restart_budget_exhausted` stops a crash loop after the configured process-restart cap.

## Export a machine-readable run summary

```powershell
python -m sigma_theory_compiler.cli engine-export `
  --service-dir runs/engine/real-search `
  --output runs/engine/real-search/export.json
```

The export contains the current service status, backend result counts, a root over all stored
status roots, last supervisor report, and the data-eligibility contract. It intentionally does not
rank a sampled-static pass as a successful gravity theory.

## Build and benchmark a bounded Rust corpus

```powershell
python -m sigma_theory_compiler.cli engine-corpus-build `
  --config configs/bounded_survivor_corpus_1m.json `
  --benchmark-cuda
```

The corpus builder hard-caps the declared run at one million formulas, preflights the disk budget,
resumes verified blocks, repairs incomplete/tampered blocks, and verifies Rust manifests and binary
records independently. The optional CUDA benchmark performs one initialization pass followed by
one to five cached measurements and requires CPU/GPU status-root equality. Throughput is a hardware
measurement, not scientific evidence.

## Stream a bounded Rust search directly into CUDA

The standalone streaming lifecycle generates independently recoverable Rust chunks, validates each
`SGSURV2` block, screens its survivors with one persistent cached CUDA owner, and retains portable
candidate lineage for promotion. The checked-in production template covers exactly
`[0, 1,000,000,000)` in 1,000 one-million-formula chunks, with a 64 GiB disk cap, a 14-day wall
cap, and paid LLM calls disabled. Confirm at least 64 GiB is free on the service drive first.

```powershell
$env:PYTHONPATH = "src"
cargo build --release --manifest-path generator-v2/Cargo.toml

python -m sigma_theory_compiler.rust_streaming_service start `
  --service-dir C:\gravity-engine-runs\rust-streaming-billion `
  --execution-config configs/persistent_parallel_search_5090.json `
  --resource-profile configs/resource_profile_5090.json `
  --stream-config configs/rust_streaming_service_billion.json

python -m sigma_theory_compiler.rust_streaming_service status `
  --service-dir C:\gravity-engine-runs\rust-streaming-billion
python -m sigma_theory_compiler.rust_streaming_service stop `
  --service-dir C:\gravity-engine-runs\rust-streaming-billion
python -m sigma_theory_compiler.rust_streaming_service resume `
  --service-dir C:\gravity-engine-runs\rust-streaming-billion
python -m sigma_theory_compiler.rust_streaming_service export `
  --service-dir C:\gravity-engine-runs\rust-streaming-billion `
  --output C:\gravity-engine-runs\rust-streaming-billion-export
```

The service preserves the exact ordinal, term IDs, sign mask, sampled-static status, and source
hashes for every pass or ambiguous result. Zero-survivor chunks remain valid, explicitly marked,
and hash-checked rather than being silently skipped. Physical NVML samples are reported separately
from scheduler occupancy: after the Rust gates reject most formulas, the surviving GPU batches can
be too short to saturate the 5090 even though no GPU work is waiting.

The first complete production run exhausted all one billion ordinals with 1,000/1,000 chunks
successful. Rust emitted 17,047,301 survivors for CUDA screening; the final sampled-static screen
retained 5,855 identities and zero ambiguous cases. The GPU owner spent only about 12.36 seconds in
the resumed 930-million-formula wave because it cleared survivors much faster than the single Rust
producer generated them. A restart-safe `rust_promotion_bridge.py` importer verified all 1,000
portable blocks and registered all 5,855 identities. The static covariant-lift gate rejected 5,785
for the forbidden baryonic action atom and blocked 70 for a missing exact nonlinear lift, leaving
zero candidates eligible for the downstream ADM/Dirac/principal evaluator.

## Current operational boundary

The service executes the existing deterministic grammar and sampled-static evaluator. A separate
promotion registry now has reviewed static covariant-lift, ADM/Dirac/principal-health, and sealed
Solar known-answer evaluators. They require exact candidate-to-action and action-to-weak-field
provenance and are not service queue stages yet. Candidate-specific Solar data and direct-observable
galaxy evaluators remain unimplemented and fail-closed. The engine also does not yet generate new
grammar productions from LLM
suggestions, distribute work across multiple machines, or enforce a separate byte quota on the
SQLite file. SQLite growth is nevertheless bounded by the configured task and wall limits, while
the whole service directory is monitored against the service disk cap.
