from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _principal_job(_: int) -> dict[str, Any]:
    from sigma_theory_compiler.horndeski_principal import (
        quartic_horndeski_full_local_principal_control,
    )

    started = time.perf_counter()
    passed, evidence = quartic_horndeski_full_local_principal_control()
    return {
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "matrix_shape": evidence["matrix_shape"],
        "first_order_status": evidence["first_order_generalized_pencil"]["status"],
    }


def _gpu_inventory() -> dict[str, str] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    values = [item.strip() for item in completed.stdout.strip().split(",")]
    if len(values) != 3:
        return None
    return {"name": values[0], "memory_mib": values[1], "compute_capability": values[2]}


def benchmark(worker_counts: list[int]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for workers in worker_counts:
        started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_principal_job, range(workers)))
        wall = time.perf_counter() - started
        if not all(item["passed"] for item in results):
            raise RuntimeError(f"principal control failed at worker count {workers}")
        records.append(
            {
                "workers": workers,
                "jobs": workers,
                "wall_seconds": wall,
                "jobs_per_second": workers / wall,
                "median_job_seconds": statistics.median(
                    item["elapsed_seconds"] for item in results
                ),
                "maximum_job_seconds": max(item["elapsed_seconds"] for item in results),
                "all_passed": True,
            }
        )
    best = max(records, key=lambda item: item["jobs_per_second"])
    return {
        "schema_version": "sigma-parallel-lane-benchmark-1.0",
        "created_utc": datetime.now(UTC).isoformat(),
        "host": {
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "gpu": _gpu_inventory(),
        },
        "workload": (
            "one independent exact quartic-Horndeski 11-by-11 local principal extraction and "
            "22-by-22 generalized first-order pencil certificate per process"
        ),
        "records": records,
        "best_measured_worker_count": best["workers"],
        "best_measured_jobs_per_second": best["jobs_per_second"],
        "interpretation": (
            "This measures one representative exact symbolic workload, not every formal task. "
            "Production limits must also reserve RAM and keep GPU and LLM queues separately capped."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", default="1,2,4,6,8")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    worker_counts = sorted({int(item) for item in args.workers.split(",") if item.strip()})
    if not worker_counts or worker_counts[0] < 1:
        raise ValueError("worker counts must be positive")
    report = benchmark(worker_counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for record in report["records"]:
        print(
            f"workers={record['workers']} wall={record['wall_seconds']:.3f}s "
            f"throughput={record['jobs_per_second']:.4f}/s"
        )
    print(f"best_workers={report['best_measured_worker_count']}")
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
