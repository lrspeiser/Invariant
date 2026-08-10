from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.bounded_survivor_corpus import verify_generated_manifest
from sigma_theory_compiler.persistent_parallel_search import PersistentParallelSearch
from sigma_theory_compiler.real_formula_execution import cuda_available
from sigma_theory_compiler.rust_streaming_search import (
    ELIGIBILITY,
    RustStreamingProducer,
    run_rust_streaming_search,
)

ROOT = Path(__file__).resolve().parents[1]
EXECUTION = ROOT / "configs" / "persistent_parallel_search_5090.json"
PROFILE = ROOT / "configs" / "resource_profile_5090.json"
GENERATOR = ROOT / "configs" / "generator_v2_billion.json"
BINARY = ROOT / "generator-v2" / "target" / "release" / "sigma-generator-v2.exe"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _execution(maximum_tasks: int = 2) -> dict:
    config = _load(EXECUTION)
    config["queue"] = {
        **config["queue"],
        "maximum_pending_work": maximum_tasks,
        "lease_seconds": 20,
        "checkpoint_every_completions": 1,
    }
    config["budget"] = {
        **config["budget"],
        "maximum_tasks": maximum_tasks,
        "maximum_wall_seconds": 20,
    }
    config["cpu"] = {**config["cpu"], "maximum_workers": 1}
    config["supervisor"] = {
        **config["supervisor"],
        "cpu_workers": 0,
        "gpu_workers": 1,
        "worker_poll_seconds": 0.01,
        "telemetry_interval_seconds": 0.02,
        "refill_interval_seconds": 0.01,
        "maximum_telemetry_bytes": 4 * 1024 * 1024,
        "maximum_wall_seconds_per_run": 20,
        "maximum_process_restarts": 1,
        "shutdown_grace_seconds": 2,
    }
    return config


def _stream(output: Path, formula_count: int = 40_000, chunk_size: int = 20_000) -> dict:
    return {
        "schema_version": "sigma-rust-streaming-search-1.0",
        "external_paid_llm_calls": False,
        "generator_config_path": str(GENERATOR),
        "generator_binary_path": str(BINARY),
        "output_directory": str(output),
        "start_ordinal": 0,
        "formula_count": formula_count,
        "chunk_formula_count": chunk_size,
        "maximum_formula_count": 1_000_000,
        "threads": 4,
        "target_pending_chunks": 2,
        "producer_lease_seconds": 20,
        "maximum_disk_bytes": 8 * 1024 * 1024,
        "maximum_wall_seconds": 20,
        "equivalence_samples_per_chunk": 8,
        "ambiguity_guard": 1e-10,
        "data_eligibility": ELIGIBILITY,
    }


def _require_local_runtime() -> None:
    if not BINARY.is_file():
        pytest.skip("bounded release Rust generator is not built")


def test_streaming_producer_is_single_owner_restart_safe_and_budgeted(tmp_path: Path) -> None:
    _require_local_runtime()
    execution = _execution(1)
    coordinator = PersistentParallelSearch(
        tmp_path / "restart.sqlite", execution, _load(PROFILE)
    )
    config = _stream(tmp_path / "chunks", formula_count=2_000, chunk_size=2_000)
    config["target_pending_chunks"] = 1
    first = RustStreamingProducer(coordinator, config, owner_id="first")
    second = RustStreamingProducer(coordinator, config, owner_id="second")
    first._acquire_owner()
    with pytest.raises(RuntimeError, match="another Rust streaming producer"):
        second.refill()
    first.release_owner()

    produced = first.refill()
    assert produced["accepted_chunks"] == 1
    assert produced["cursor"]["exhausted"]
    with coordinator.connect() as connection:
        connection.execute(
            "UPDATE rust_stream_source SET next_ordinal=0,sequence=0 WHERE source_id=?",
            (first.source_id,),
        )
        connection.execute(
            "UPDATE rust_stream_chunks SET state='generating' WHERE source_id=? AND sequence=0",
            (first.source_id,),
        )
    first.release_owner()

    recovered = RustStreamingProducer(coordinator, config, owner_id="recovered")
    assert recovered.status()["chunk_counts"] == {"verified": 1}
    lease = coordinator.claim("gpu", "crash-window", lease_seconds=20)
    assert lease is not None
    assert coordinator.finish(lease, "crash-window", {"bounded_test": True})
    replay = recovered.refill()
    assert replay["duplicate_chunks"] == 1
    assert replay["cursor"]["exhausted"]

    invalid = dict(config)
    invalid["producer_lease_seconds"] = 0
    with pytest.raises(ValueError, match="lease must be positive"):
        RustStreamingProducer(coordinator, invalid)


def test_real_rust_stream_overlaps_cached_single_gpu_owner(tmp_path: Path) -> None:
    _require_local_runtime()
    available, reason = cuda_available()
    if not available:
        pytest.skip(reason)
    report = run_rust_streaming_search(
        tmp_path / "stream.sqlite",
        _execution(),
        _load(PROFILE),
        _stream(tmp_path / "chunks"),
        tmp_path / "telemetry.jsonl",
        output_report=tmp_path / "report.json",
    )
    assert report["formula_count"] == 40_000
    assert report["chunk_count"] == 2
    assert report["exact_interval"] == {
        "start_ordinal": 0,
        "end_ordinal_exclusive": 40_000,
        "complete": True,
    }
    assert len(report["provenance"]["generator_config_sha256"]) == 64
    assert len(report["provenance"]["generator_binary_sha256"]) == 64
    assert len(report["provenance"]["verified_chunk_chain_sha256"]) == 64
    assert report["cursor"]["next_ordinal"] == 40_000
    assert report["cursor"]["exhausted"]
    assert report["all_work_succeeded"]
    assert report["cpu_gpu_equivalence_passed"]
    assert report["consumer"]["backend_counts"] == {"gpu_cupy_binary_cached": 2}
    assert report["consumer"]["cached_chunks"] >= 1
    assert report["producer"]["formulas_per_second"] > 0
    assert report["consumer"]["records_per_second"] > 0
    assert report["combined"]["formulas_per_second"] > 0
    assert report["combined"]["overlap_observed"]
    assert report["data_eligibility"] == {
        **ELIGIBILITY,
        "paid_llm_calls": False,
        "passed": True,
    }
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved == report


def test_zero_survivor_rust_chunk_is_strictly_normalized_and_restartable(
    tmp_path: Path,
) -> None:
    _require_local_runtime()
    execution = _execution(1)
    coordinator = PersistentParallelSearch(
        tmp_path / "zero.sqlite", execution, _load(PROFILE)
    )
    config = _stream(tmp_path / "chunks", formula_count=1_000_000, chunk_size=1_000_000)
    config.update(
        {
            "start_ordinal": 70_000_000,
            "target_pending_chunks": 1,
            "maximum_disk_bytes": 64 * 1024 * 1024,
        }
    )
    producer = RustStreamingProducer(coordinator, config)
    generated = producer.refill()
    assert generated["accepted_chunks"] == 1
    assert generated["cursor"]["next_ordinal"] == 71_000_000
    assert generated["cursor"]["exhausted"]
    with coordinator.connect() as connection:
        row = connection.execute(
            "SELECT * FROM rust_stream_chunks WHERE source_id=? AND sequence=0",
            (producer.source_id,),
        ).fetchone()
        payload = json.loads(
            connection.execute("SELECT payload_json FROM work").fetchone()[0]
        )
    assert row["state"] == "enqueued"
    assert row["record_count"] == 0
    assert row["block_size_bytes"] == 44
    manifest = json.loads(Path(row["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["survivor_count"] == 0
    assert "survive_sampled_static" not in manifest["gate_counts"]
    assert payload["manifest_verification_normalization"] == (
        "omitted-zero-survive-sampled-static-gate"
    )
    with pytest.raises(ValueError, match="corpus accounting mismatch"):
        verify_generated_manifest(
            Path(row["manifest_path"]),
            Path(config["output_directory"]),
            GENERATOR,
            expected_start=70_000_000,
            expected_end=71_000_000,
            equivalence_samples_per_block=8,
        )
