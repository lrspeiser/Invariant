from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.persistent_parallel_search import PersistentParallelSearch
from sigma_theory_compiler.real_formula_execution import cuda_available
from sigma_theory_compiler.rust_streaming_search import ELIGIBILITY, RustStreamingProducer
from sigma_theory_compiler.rust_streaming_service import (
    export_service,
    initialize_service,
    resume_service,
    service_status,
)

ROOT = Path(__file__).resolve().parents[1]
EXECUTION = ROOT / "configs" / "persistent_parallel_search_5090.json"
PROFILE = ROOT / "configs" / "resource_profile_5090.json"
GENERATOR = ROOT / "configs" / "generator_v2_billion.json"
BINARY = ROOT / "generator-v2" / "target" / "release" / "sigma-generator-v2.exe"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _execution(path: Path) -> Path:
    config = _load(EXECUTION)
    config["queue"] = {
        **config["queue"],
        "maximum_pending_work": 3,
        "lease_seconds": 30,
        "checkpoint_every_completions": 1,
    }
    config["budget"] = {
        **config["budget"],
        "maximum_tasks": 3,
        "maximum_wall_seconds": 30,
    }
    config["cpu"] = {**config["cpu"], "maximum_workers": 1}
    config["supervisor"] = {
        **config["supervisor"],
        "cpu_workers": 0,
        "gpu_workers": 1,
        "worker_poll_seconds": 0.01,
        "telemetry_interval_seconds": 0.1,
        "refill_interval_seconds": 0.01,
        "maximum_telemetry_bytes": 4 * 1024 * 1024,
        "maximum_wall_seconds_per_run": 30,
        "maximum_process_restarts": 1,
        "shutdown_grace_seconds": 2,
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _stream(path: Path) -> Path:
    config = {
        "schema_version": "sigma-rust-streaming-search-1.0",
        "external_paid_llm_calls": False,
        "generator_config_path": str(GENERATOR),
        "generator_binary_path": str(BINARY),
        "output_directory": "replaced-by-service",
        "start_ordinal": 0,
        "formula_count": 60_000,
        "chunk_formula_count": 20_000,
        "maximum_formula_count": 1_000_000_000,
        "threads": 4,
        "target_pending_chunks": 2,
        "producer_lease_seconds": 30,
        "maximum_disk_bytes": 16 * 1024 * 1024,
        "maximum_wall_seconds": 30,
        "equivalence_samples_per_chunk": 8,
        "ambiguity_guard": 1e-10,
        "data_eligibility": ELIGIBILITY,
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_resumed_multichunk_service_overlaps_cuda_and_exports_lineage(
    tmp_path: Path,
) -> None:
    if not BINARY.is_file():
        pytest.skip("bounded release Rust generator is not built")
    available, reason = cuda_available()
    if not available:
        pytest.skip(reason)
    service_root = tmp_path / "service"
    initialize_service(
        service_root,
        _execution(tmp_path / "execution.json"),
        PROFILE,
        _stream(tmp_path / "stream.json"),
    )
    execution = _load(service_root / "execution-config.json")
    resource = _load(service_root / "resource-profile.json")
    stream = _load(service_root / "stream-config.json")
    coordinator = PersistentParallelSearch(service_root / "stream.sqlite", execution, resource)
    crashed = RustStreamingProducer(coordinator, stream, owner_id="pre-crash-producer")
    first = crashed.refill()
    assert first["accepted_chunks"] == 1
    assert first["cursor"]["next_ordinal"] == 20_000
    crashed.release_owner()
    with coordinator.connect() as connection:
        connection.execute(
            "UPDATE rust_stream_source SET owner_id='producer-999999-dead',"
            "owner_lease_expires_utc='2999-01-01T00:00:00+00:00'"
        )
    service_state = _load(service_root / "service.json")
    service_state["state"] = "running"
    service_state["pid"] = 999999
    (service_root / "service.json").write_text(json.dumps(service_state), encoding="utf-8")

    resumed = resume_service(service_root, foreground=True)
    run = resumed["run"]
    streaming = run["streaming"]
    assert streaming["formula_count"] == 60_000
    assert streaming["chunk_count"] == 3
    assert streaming["exact_interval"]["complete"]
    assert streaming["cursor"]["exhausted"]
    assert streaming["all_work_succeeded"]
    assert streaming["cpu_gpu_equivalence_passed"]
    assert streaming["consumer"]["cached_chunks"] >= 2
    assert streaming["combined"]["overlap_observed"]
    assert streaming["combined"]["formulas_per_second"] > 0
    assert run["hardware"]["sample_count"] >= 1
    assert run["hardware"]["semantics"].startswith("periodic physical sensor")
    status = service_status(service_root)
    assert status["state"] == "completed"
    assert not status["alive"]
    assert status["source"]["exhausted"]

    output = tmp_path / "promotion-export.json"
    exported = export_service(service_root, output, maximum_export_bytes=8 * 1024 * 1024)
    assert exported["block_count"] == 3
    assert exported["survivor_identity_count"] == (
        exported["pass_count"] + exported["ambiguous_count"]
    )
    assert exported["survivor_identity_count"] > 0
    assert exported["ambiguous_count"] >= 0
    assert len(exported["blocks_root_sha256"]) == 64
    for block in exported["blocks"]:
        artifact = output.parent / block["file"]
        assert artifact.is_file()
        assert artifact.stat().st_size > 0
        assert block["source_block_sha256"]
        assert block["result_status_root_sha256"]
    assert json.loads(output.read_text(encoding="utf-8")) == exported


def test_service_rejects_unsealed_or_underbudgeted_sources(tmp_path: Path) -> None:
    if not BINARY.is_file():
        pytest.skip("bounded release Rust generator is not built")
    execution = _execution(tmp_path / "execution.json")
    stream_path = _stream(tmp_path / "stream.json")
    stream = _load(stream_path)
    stream["data_eligibility"] = {**ELIGIBILITY, "observational_data_opened": True}
    stream_path.write_text(json.dumps(stream), encoding="utf-8")
    with pytest.raises(ValueError, match="eligibility"):
        initialize_service(tmp_path / "unsealed", execution, PROFILE, stream_path)

    stream_path = _stream(tmp_path / "stream-budget.json")
    stream = _load(stream_path)
    stream["maximum_disk_bytes"] = 1024
    stream_path.write_text(json.dumps(stream), encoding="utf-8")
    with pytest.raises(ValueError, match="disk budget"):
        initialize_service(tmp_path / "underbudgeted", execution, PROFILE, stream_path)
