from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY as PROMOTION_ELIGIBILITY
from sigma_theory_compiler.promotion_orchestrator import PromotionOrchestrator
from sigma_theory_compiler.real_formula_execution import cuda_available
from sigma_theory_compiler.rust_promotion_service_stage import RustPromotionServiceStage
from sigma_theory_compiler.rust_streaming_search import ELIGIBILITY
from sigma_theory_compiler.rust_streaming_service import service_status, start_service

ROOT = Path(__file__).resolve().parents[1]
EXECUTION = ROOT / "configs" / "persistent_parallel_search_5090.json"
PROFILE = ROOT / "configs" / "resource_profile_5090.json"
GENERATOR = ROOT / "configs" / "generator_v2_billion.json"
PIPELINE = ROOT / "configs" / "promotion_pipeline_fail_closed.json"
BINARY = ROOT / "generator-v2" / "target" / "release" / "sigma-generator-v2.exe"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _execution(path: Path) -> Path:
    config = _load(EXECUTION)
    config["queue"] = {
        **config["queue"],
        "maximum_pending_work": 2,
        "lease_seconds": 30,
        "checkpoint_every_completions": 1,
    }
    config["budget"] = {
        **config["budget"],
        "maximum_tasks": 2,
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
        "maximum_telemetry_bytes": 2 * 1024 * 1024,
        "maximum_wall_seconds_per_run": 30,
        "maximum_process_restarts": 1,
        "shutdown_grace_seconds": 2,
    }
    return _write(path, config)


def _parallel(path: Path) -> Path:
    return _write(
        path,
        {
            "schema_version": "sigma-rust-parallel-streaming-1.0",
            "external_paid_llm_calls": False,
            "generator_config_path": str(GENERATOR),
            "generator_binary_path": str(BINARY),
            "output_directory": "service-owned",
            "start_ordinal": 0,
            "formula_count": 10_000,
            "chunk_formula_count": 5_000,
            "maximum_formula_count": 1_000_000_000,
            "producer_workers": 2,
            "threads_per_producer": 1,
            "target_pending_chunks": 4,
            "producer_chunk_lease_seconds": 30,
            "maximum_disk_bytes": 8 * 1024 * 1024,
            "maximum_wall_seconds": 30,
            "equivalence_samples_per_chunk": 8,
            "ambiguity_guard": 1e-10,
            "data_eligibility": ELIGIBILITY,
        },
    )


def _stage(path: Path) -> Path:
    reviewed = _load(ROOT / "configs" / "rust_parallel_promotion_stage_fail_closed.json")[
        "reviewed_evaluator_descriptors"
    ]
    return _write(
        path,
        {
            "schema_version": "sigma-rust-promotion-service-stage-1.0",
            "enabled": True,
            "external_paid_llm_calls": False,
            "pipeline_config_path": str(PIPELINE),
            "generator_config_path": str(GENERATOR),
            "reviewed_evaluator_descriptors": reviewed,
            "maximum_records_per_run": 5000,
            "maximum_blocks_per_run": 4,
            "maximum_orchestrator_tasks_per_run": 100,
            "maximum_total_candidates": 5000,
            "maximum_disk_bytes": 32 * 1024 * 1024,
            "maximum_wall_seconds": 30,
            "data_eligibility": PROMOTION_ELIGIBILITY,
        },
    )


def _work_root(database: Path) -> str:
    connection = sqlite3.connect(database)
    rows = connection.execute(
        "SELECT work_id,state,payload_json,result_json,error_text FROM work ORDER BY work_id"
    ).fetchall()
    connection.close()
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def test_automatic_bridge_is_bounded_idempotent_and_cannot_mutate_upstream(
    tmp_path: Path,
) -> None:
    if not BINARY.is_file():
        pytest.skip("release Rust generator is unavailable")
    available, reason = cuda_available()
    if not available:
        pytest.skip(reason)
    service_root = tmp_path / "service"
    result = start_service(
        service_root,
        _execution(tmp_path / "execution.json"),
        PROFILE,
        _parallel(tmp_path / "parallel.json"),
        foreground=True,
        promotion_stage_config_path=_stage(tmp_path / "stage.json"),
    )
    downstream = result["run"]["automatic_promotion"]
    assert downstream["outcome"] == "completed"
    assert downstream["bridge_run"]["snapshot_exhausted"]
    assert downstream["candidate_count_after"] > 0
    assert downstream["paid_llm_spend_usd"] == 0.0
    assert len(downstream["provenance"]["combined_root_sha256"]) == 64
    assert len(downstream["reviewed_evaluator_registry"]) == 4
    assert len(downstream["evaluator_registry_root_sha256"]) == 64
    assert downstream["orchestrator_run"]["evaluated"] >= downstream["candidate_count_after"]
    assert downstream["orchestrator"]["stages"]["covariant_symbolic_health"]["counts"]
    assert downstream["orchestrator"]["stages"]["adm_dirac_principal_health"]["counts"][
        "blocked"
    ] > 0
    upstream_root = _work_root(service_root / "stream.sqlite")

    stage = RustPromotionServiceStage(
        service_root / "downstream", _load(service_root / "promotion-stage-config.json")
    )
    replay = stage.run(service_root / "downstream" / "source-promotion-export.json")
    assert replay["bridge_run"]["consumed_records"] == 0
    assert replay["bridge_run"]["registered_candidates"] == 0
    assert replay["provenance"]["combined_root_sha256"] == downstream["provenance"][
        "combined_root_sha256"
    ]
    assert replay["evaluator_registry_root_sha256"] == downstream[
        "evaluator_registry_root_sha256"
    ]
    assert _work_root(service_root / "stream.sqlite") == upstream_root
    status = service_status(service_root)
    assert status["state"] == "completed"
    assert status["automatic_promotion"]["run_count"] == 2
    assert status["automatic_promotion"]["data_eligibility"] == {
        **PROMOTION_ELIGIBILITY,
        "passed": True,
    }

    limited_config = _load(service_root / "promotion-stage-config.json")
    limited_config["maximum_records_per_run"] = 1
    limited_config["maximum_total_candidates"] = 1
    limited = RustPromotionServiceStage(tmp_path / "limited-downstream", limited_config)
    first_limited = limited.run(
        service_root / "downstream" / "source-promotion-export.json"
    )
    assert first_limited["candidate_count_after"] == 1
    backpressured = limited.run(
        service_root / "downstream" / "source-promotion-export.json"
    )
    assert backpressured["outcome"] == "backpressured"
    assert backpressured["bridge_run"] is None


def test_stage_rejects_observational_unsealing(tmp_path: Path) -> None:
    config = _load(_stage(tmp_path / "stage.json"))
    config["data_eligibility"] = {
        **PROMOTION_ELIGIBILITY,
        "observational_data_opened": True,
    }
    with pytest.raises(ValueError, match="eligibility"):
        RustPromotionServiceStage(tmp_path / "downstream", config)


def test_descriptor_tampering_and_unlisted_registry_are_rejected(tmp_path: Path) -> None:
    config = _load(_stage(tmp_path / "stage.json"))
    config["reviewed_evaluator_descriptors"][0]["descriptor_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="descriptor file hash"):
        RustPromotionServiceStage(tmp_path / "tampered", config)

    config = _load(_stage(tmp_path / "binding-stage.json"))
    config["reviewed_evaluator_descriptors"][0]["required_binding_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="binding does not match"):
        RustPromotionServiceStage(tmp_path / "binding-tampered", config)

    config = _load(_stage(tmp_path / "clean-stage.json"))
    descriptor_allowlist = config["reviewed_evaluator_descriptors"]
    config["reviewed_evaluator_descriptors"] = []
    stage = RustPromotionServiceStage(tmp_path / "unlisted", config)
    orchestrator = PromotionOrchestrator(stage.orchestrator_database, stage.pipeline)
    descriptor = _load(Path(descriptor_allowlist[0]["descriptor_path"]))
    descriptor["artifact_path"] = str((ROOT / descriptor["artifact_path"]).resolve())
    orchestrator.register_evaluator(descriptor)
    with pytest.raises(ValueError, match="unlisted evaluator"):
        stage._register_reviewed_evaluators(orchestrator)
