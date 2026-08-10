from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from sigma_theory_compiler.rust_parallel_streaming_search import (
    run_parallel_rust_streaming_search,
)
from sigma_theory_compiler.rust_streaming_search import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
MAXIMUM_BENCHMARK_FORMULAS = 32_000_000


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _execution(tasks: int) -> dict:
    config = _load(ROOT / "configs" / "persistent_parallel_search_5090.json")
    config["queue"] = {
        **config["queue"],
        "maximum_pending_work": max(tasks, 32),
        "lease_seconds": 60,
        "checkpoint_every_completions": max(1, tasks // 2),
    }
    config["budget"] = {
        **config["budget"],
        "maximum_tasks": tasks,
        "maximum_wall_seconds": 120,
    }
    config["cpu"] = {**config["cpu"], "maximum_workers": 1}
    config["supervisor"] = {
        **config["supervisor"],
        "cpu_workers": 0,
        "gpu_workers": 1,
        "worker_poll_seconds": 0.01,
        "telemetry_interval_seconds": 0.25,
        "refill_interval_seconds": 0.01,
        "maximum_telemetry_bytes": 8 * 1024 * 1024,
        "maximum_wall_seconds_per_run": 120,
        "maximum_process_restarts": 1,
        "shutdown_grace_seconds": 2,
    }
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded 5090 Rust producer scaling benchmark")
    parser.add_argument("--formula-count", type=int, default=16_000_000)
    parser.add_argument("--start-ordinal", type=int, default=64_000_000)
    parser.add_argument("--workers", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--output")
    args = parser.parse_args()
    if not 1_000_000 <= args.formula_count <= MAXIMUM_BENCHMARK_FORMULAS:
        raise ValueError("benchmark formula count must be between 1m and 32m")
    if args.formula_count % 1_000_000 or args.start_ordinal % 1_000_000:
        raise ValueError("benchmark range must align to one-million formula chunks")
    if any(worker not in (8, 12, 16) for worker in args.workers):
        raise ValueError("benchmark workers are restricted to 8, 12, or 16")
    tasks = args.formula_count // 1_000_000
    profile = _load(ROOT / "configs" / "resource_profile_5090.json")
    results = []
    with tempfile.TemporaryDirectory(prefix="sigma-rust-scaling-") as temporary:
        temporary_root = Path(temporary)
        for workers in args.workers:
            run_root = temporary_root / f"workers-{workers}"
            config = {
                "schema_version": "sigma-rust-parallel-streaming-1.0",
                "external_paid_llm_calls": False,
                "generator_config_path": str(ROOT / "configs" / "generator_v2_billion.json"),
                "generator_binary_path": str(
                    ROOT / "generator-v2" / "target" / "release" / "sigma-generator-v2.exe"
                ),
                "output_directory": str(run_root / "chunks"),
                "promotion_directory": str(run_root / "promotion"),
                "start_ordinal": args.start_ordinal,
                "formula_count": args.formula_count,
                "chunk_formula_count": 1_000_000,
                "maximum_formula_count": 1_000_000_000,
                "producer_workers": workers,
                "threads_per_producer": 1,
                "target_pending_chunks": workers * 2,
                "producer_chunk_lease_seconds": 60,
                "maximum_disk_bytes": 2 * 1024**3,
                "maximum_wall_seconds": 120,
                "equivalence_samples_per_chunk": 8,
                "ambiguity_guard": 1e-10,
                "data_eligibility": ELIGIBILITY,
            }
            report = run_parallel_rust_streaming_search(
                run_root / "stream.sqlite",
                _execution(tasks),
                profile,
                config,
                run_root / "telemetry.jsonl",
            )
            if not report["all_work_succeeded"]:
                raise RuntimeError(f"bounded {workers}-producer benchmark did not complete")
            gpu_stats = report["hardware"].get("gpu_utilization_percent") or {}
            cpu_stats = report["hardware"].get("cpu_utilization_percent") or {}
            results.append(
                {
                    "workers": workers,
                    "wall_seconds": report["combined"]["wall_seconds"],
                    "source_formulas_per_second": report["combined"][
                        "source_formulas_per_second"
                    ],
                    "producer_slot_utilization_fraction": report["producer"][
                        "wall_utilization_fraction"
                    ],
                    "producer_gpu_overlap_seconds": report["combined"][
                        "producer_consumer_overlap_seconds"
                    ],
                    "gpu_survivor_records_per_second": report["consumer"][
                        "records_per_second"
                    ],
                    "physical_gpu_mean_percent": gpu_stats.get("mean"),
                    "physical_gpu_peak_percent": gpu_stats.get("peak"),
                    "physical_cpu_mean_percent": cpu_stats.get("mean"),
                    "physical_cpu_peak_percent": cpu_stats.get("peak"),
                }
            )
    fastest = max(results, key=lambda result: result["source_formulas_per_second"])
    artifact = {
        "schema_version": "sigma-rust-parallel-scaling-benchmark-1.0",
        "formula_interval": {
            "start_ordinal": args.start_ordinal,
            "end_ordinal_exclusive": args.start_ordinal + args.formula_count,
            "formula_count": args.formula_count,
            "chunk_formula_count": 1_000_000,
        },
        "results": results,
        "fastest_workers": fastest["workers"],
        "selected_default_workers": fastest["workers"],
        "interpretation": (
            "Scheduler occupancy is not hardware utilization; physical GPU values are periodic "
            "NVML samples and may miss sub-sample kernel bursts."
        ),
        "data_eligibility": {**ELIGIBILITY, "paid_llm_calls": False, "passed": True},
    }
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
