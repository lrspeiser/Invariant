from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.persistent_parallel_search import PersistentParallelSearch
from sigma_theory_compiler.real_formula_execution import cuda_available
from sigma_theory_compiler.rust_parallel_streaming_search import (
    ParallelRustRangeScheduler,
    run_parallel_rust_streaming_search,
)
from sigma_theory_compiler.rust_streaming_search import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
EXECUTION = ROOT / "configs" / "persistent_parallel_search_5090.json"
PROFILE = ROOT / "configs" / "resource_profile_5090.json"
GENERATOR = ROOT / "configs" / "generator_v2_billion.json"
BINARY = ROOT / "generator-v2" / "target" / "release" / "sigma-generator-v2.exe"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _execution(tasks: int) -> dict:
    config = _load(EXECUTION)
    config["queue"] = {
        **config["queue"],
        "maximum_pending_work": max(8, tasks),
        "lease_seconds": 30,
        "checkpoint_every_completions": 1,
    }
    config["budget"] = {
        **config["budget"],
        "maximum_tasks": tasks,
        "maximum_wall_seconds": 30,
    }
    config["cpu"] = {**config["cpu"], "maximum_workers": 1}
    config["supervisor"] = {
        **config["supervisor"],
        "cpu_workers": 0,
        "gpu_workers": 1,
        "worker_poll_seconds": 0.01,
        "telemetry_interval_seconds": 0.2,
        "refill_interval_seconds": 0.01,
        "maximum_telemetry_bytes": 4 * 1024 * 1024,
        "maximum_wall_seconds_per_run": 30,
        "maximum_process_restarts": 1,
        "shutdown_grace_seconds": 2,
    }
    return config


def _parallel(
    output: Path,
    *,
    formula_count: int,
    chunk_size: int,
    workers: int,
    start: int = 60_000_000,
) -> dict:
    return {
        "schema_version": "sigma-rust-parallel-streaming-1.0",
        "external_paid_llm_calls": False,
        "generator_config_path": str(GENERATOR),
        "generator_binary_path": str(BINARY),
        "output_directory": str(output),
        "promotion_directory": str(output / "promotion"),
        "start_ordinal": start,
        "formula_count": formula_count,
        "chunk_formula_count": chunk_size,
        "maximum_formula_count": 1_000_000_000,
        "producer_workers": workers,
        "threads_per_producer": 1,
        "target_pending_chunks": max(workers * 2, workers),
        "producer_chunk_lease_seconds": 30,
        "maximum_disk_bytes": 512 * 1024 * 1024,
        "maximum_wall_seconds": 30,
        "equivalence_samples_per_chunk": 8,
        "ambiguity_guard": 1e-10,
        "data_eligibility": ELIGIBILITY,
    }


def _require_rust() -> None:
    if not BINARY.is_file():
        pytest.skip("release Rust generator is not built")


def test_disjoint_leases_and_verified_duplicate_replay(tmp_path: Path) -> None:
    _require_rust()
    config = _parallel(
        tmp_path / "chunks", formula_count=40_000, chunk_size=20_000, workers=2, start=0
    )
    execution = _execution(2)
    coordinator = PersistentParallelSearch(tmp_path / "parallel.sqlite", execution, _load(PROFILE))
    scheduler = ParallelRustRangeScheduler(coordinator, config, scheduler_id="first")
    first = scheduler._lease()
    second = scheduler._lease()
    assert first is not None and second is not None
    assert int(first[0]["sequence"]) == 0
    assert int(second[0]["sequence"]) == 1
    assert (
        int(first[0]["end_ordinal_exclusive"]), int(second[0]["start_ordinal"])
    ) == (20_000, 20_000)
    scheduler._generate(first[0], first[1])
    accepted, duplicate, _ = scheduler._enqueue_verified()
    assert (accepted, duplicate) == (1, 0)
    with coordinator.connect() as connection:
        connection.execute(
            "UPDATE rust_parallel_chunks SET state='verified' WHERE source_id=? AND sequence=0",
            (scheduler.source_id,),
        )
        connection.execute(
            "UPDATE rust_parallel_chunks SET lease_expires_utc='2000-01-01T00:00:00+00:00' "
            "WHERE source_id=? AND sequence=1",
            (scheduler.source_id,),
        )
    scheduler.close()

    resumed = ParallelRustRangeScheduler(coordinator, config, scheduler_id="resumed")
    accepted, duplicate, _ = resumed._enqueue_verified()
    assert (accepted, duplicate) == (0, 1)
    status = resumed.status()
    assert status["chunk_counts"] == {"available": 1, "enqueued": 1}
    assert status["next_contiguous_ordinal"] == 20_000
    resumed.close()


def test_four_real_rust_producers_overlap_one_cached_cuda_owner(tmp_path: Path) -> None:
    _require_rust()
    available, reason = cuda_available()
    if not available:
        pytest.skip(reason)
    baseline = run_parallel_rust_streaming_search(
        tmp_path / "baseline.sqlite",
        _execution(8),
        _load(PROFILE),
        _parallel(
            tmp_path / "baseline-chunks",
            formula_count=8_000_000,
            chunk_size=1_000_000,
            workers=1,
        ),
        tmp_path / "baseline-telemetry.jsonl",
    )
    report = run_parallel_rust_streaming_search(
        tmp_path / "parallel.sqlite",
        _execution(8),
        _load(PROFILE),
        _parallel(
            tmp_path / "chunks",
            formula_count=8_000_000,
            chunk_size=1_000_000,
            workers=4,
        ),
        tmp_path / "telemetry.jsonl",
    )
    assert report["all_work_succeeded"]
    assert report["cursor"]["exhausted"]
    assert report["cursor"]["next_contiguous_ordinal"] == 68_000_000
    assert report["producer"]["peak_active"] == 4
    assert report["consumer"]["backend_counts"] == {"gpu_cupy_binary_cached": 8}
    assert report["consumer"]["cache_reused_chunks"] >= 7
    assert report["combined"]["source_formulas_screened"] == 8_000_000
    assert report["combined"]["source_formulas_per_second"] > 0
    assert report["combined"]["producer_consumer_overlap_seconds"] > 0
    assert report["combined"]["producer_consumer_overlap_seconds"] <= report["combined"][
        "wall_seconds"
    ]
    assert report["hardware"]["sample_count"] >= 1
    speedup = baseline["combined"]["wall_seconds"] / report["combined"]["wall_seconds"]
    assert speedup > 1.2
    print(
        json.dumps(
            {
                "formula_count": 8_000_000,
                "single_producer_wall_seconds": baseline["combined"]["wall_seconds"],
                "four_producer_wall_seconds": report["combined"]["wall_seconds"],
                "wall_speedup": speedup,
                "single_source_formulas_per_second": baseline["combined"][
                    "source_formulas_per_second"
                ],
                "four_source_formulas_per_second": report["combined"][
                    "source_formulas_per_second"
                ],
                "four_producer_peak_active": report["producer"]["peak_active"],
                "four_producer_utilization_fraction": report["producer"][
                    "wall_utilization_fraction"
                ],
                "four_producer_gpu_overlap_seconds": report["combined"][
                    "producer_consumer_overlap_seconds"
                ],
                "gpu_survivor_records_per_second": report["consumer"][
                    "records_per_second"
                ],
                "physical_gpu_sample_count": report["hardware"]["gpu_available_samples"],
                "physical_gpu_mean_percent": (
                    report["hardware"]["gpu_utilization_percent"] or {}
                ).get("mean"),
                "physical_gpu_peak_percent": (
                    report["hardware"]["gpu_utilization_percent"] or {}
                ).get("peak"),
            },
            sort_keys=True,
        )
    )
    assert report["data_eligibility"] == {
        **ELIGIBILITY,
        "paid_llm_calls": False,
        "passed": True,
    }
