import json
import multiprocessing
import time
from pathlib import Path

import pytest

from sigma_theory_compiler.continuous_formula_formal_backend import (
    MANIFEST_SCHEMA,
    build_formal_evidence,
    load_backend_config,
)
from sigma_theory_compiler.continuous_formula_formal_backend import (
    _sealed as backend_sealed,
)
from sigma_theory_compiler.continuous_scientific_pipeline_service import (
    _sealed as service_sealed,
)
from sigma_theory_compiler.continuous_scientific_pipeline_service import (
    acquire_lease,
    apply_action,
    build_execution_result,
    build_readiness,
    initial_queue,
    load_service_config,
    release_lease,
    run_bounded_service,
    validate_execution_result,
    validate_readiness,
)
from sigma_theory_compiler.high_throughput import (
    build_basis,
    candidate_id,
    correction_expression,
    decode_ordinal,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/continuous_scientific_pipeline_service.json"
ARTIFACT = ROOT / "runs/engine/continuous-scientific-pipeline-service-readiness.json"
RESULT = ROOT / "runs/engine/continuous-scientific-pipeline-service-result.json"


def _deadline_sleeping_worker(output: object) -> None:
    del output
    time.sleep(60)


def _deadline_success_worker(output: object) -> None:
    output.put({"ok": True, "result": {"bounded": True}})


def _deadline_error_worker(output: object) -> None:
    output.put({"ok": False, "error": "ValueError: exact formal failure"})


def _copy_backend_dependencies(tmp_path: Path, service_config: dict) -> None:
    backend_path = Path(service_config["formal_backend_config_path"])
    backend = json.loads((ROOT / backend_path).read_text())
    for relative in (
        backend_path,
        Path(backend["generator_config_path"]),
        Path(backend["grammar_path"]),
        Path(backend["field_contract_path"]),
        Path(backend["formal_controls_path"]),
        Path(backend["candidate_mapper_source_path"]),
        Path(backend["action_health_source_path"]),
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())


def _receipt(count: int = 1024) -> dict:
    import hashlib

    body = {
        "candidate_root_sha256": "a" * 64,
        "screen_decision": "pass",
        "unique_formula_count": count,
        "theory_pass_claimed": False,
        "observations_opened": False,
        "rank_eligible": False,
    }
    body["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return body


def _generation(count: int = 1024) -> dict:
    receipt = _receipt(count)
    generator = json.loads((ROOT / "configs/generator_v2_billion.json").read_text())
    decoded = decode_ordinal(generator["basis_count"], generator["max_action_terms"], 7)
    manifest = backend_sealed(
        {
            "schema_version": MANIFEST_SCHEMA,
            "batch": {
                "start_ordinal": 0,
                "end_ordinal_exclusive": count,
                "candidate_count": count,
            },
            "candidate_root_sha256": receipt["candidate_root_sha256"],
            "screen_counts": {"reject": 0, "pass": count, "ambiguous": 0},
            "all_survivor_ordinals_root_sha256": "b" * 64,
            "survivor_records": [
                {
                    "candidate_id": candidate_id(generator["protocol_version"], decoded),
                    "ordinal": 7,
                    "term_ids": decoded["term_ids"],
                    "signs": decoded["signs"],
                    "correction_expression": correction_expression(
                        decoded, build_basis(generator["basis_count"])
                    ),
                    "sampled_static_margin": 1.0,
                }
            ],
            "survivor_record_count": count,
            "sample_complete": False,
            "observations_opened": False,
            "forbidden_target_inputs_opened": False,
        }
    )
    return {"receipt": receipt, "manifest": manifest}


def test_readiness_exact_and_not_started() -> None:
    value = build_readiness(ROOT, CONFIG)
    assert value == json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_readiness(value, ROOT, CONFIG)
    assert value["execution_state"] == {
        "service_started_at_preregistration": False,
        "cycles_executed_at_preregistration": 0,
        "queue_created_at_preregistration": False,
        "current_runtime_status_claimed": False,
        "live_SQLite_accessed": False,
    }
    assert value["snapshot_scope"] == {
        "artifact_role": "preexecution_preregistration",
        "runtime_status_asserted": False,
        "completed_execution_reported_separately": True,
        "completed_execution_result_path": (
            "runs/engine/continuous-scientific-pipeline-service-result.json"
        ),
    }
    assert value["service_contract"]["hard_owned_child_formal_timeout"] is True
    assert not any(value["seals"].values())


def test_completed_execution_result_is_exact_and_fail_closed() -> None:
    value = build_execution_result(ROOT, CONFIG)
    assert value == json.loads(RESULT.read_text(encoding="utf-8"))
    validate_execution_result(value, ROOT, CONFIG)
    assert value["coverage"]["unique_formula_count"] == 3_932_160
    assert value["coverage"]["real_CPU_batches"] == 8
    assert len(value["completed_receipt_bindings"]) == 8
    assert (
        value["completed_receipt_bindings"][0]["ordinal_interval"]["start_ordinal"]
        == (value["coverage"]["start_ordinal"])
    )
    assert (
        value["completed_receipt_bindings"][-1]["ordinal_interval"]["stop_ordinal_exclusive"]
        == value["coverage"]["stop_ordinal_exclusive"]
    )
    assert (
        value["terminal_runtime_archive"]["queue"]["content_sha256"]
        == value["runtime_binding"]["queue_content_sha256"]
    )
    assert value["outcomes"] == {
        "sampled_static_reject_batches": 5,
        "sampled_static_pass_batches": 3,
        "formal_receipts": 3,
        "formal_blocks": 3,
        "formal_passes": 0,
        "leaderboard_rebuild_requests": 0,
        "rank_assignments": 0,
    }
    assert not any(value["seals"].values())


def test_completed_result_validation_does_not_read_mutable_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = build_execution_result(ROOT, CONFIG)
    config = load_service_config(ROOT, CONFIG)
    runtime = (ROOT / config["runtime_directory"]).resolve()
    original = Path.read_text

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        resolved = path.resolve()
        if resolved == runtime or runtime in resolved.parents:
            pytest.fail("immutable result validation read mutable runtime")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    validate_execution_result(value, ROOT, CONFIG)


@pytest.mark.parametrize(
    "field",
    [
        "candidate_root",
        "ordinal_interval",
        "formal_receipt",
        "runtime_queue",
        "terminal_queue",
        "replay_dependency",
        "interpretation",
    ],
)
def test_execution_result_receipt_and_runtime_tamper_fails_closed(field: str) -> None:
    value = json.loads(RESULT.read_text(encoding="utf-8"))
    value.pop("content_sha256")
    if field == "candidate_root":
        value["completed_receipt_bindings"][0]["generated_receipt"]["candidate_root_sha256"] = (
            "0" * 64
        )
    elif field == "ordinal_interval":
        value["completed_receipt_bindings"][1]["ordinal_interval"]["start_ordinal"] += 1
    elif field == "formal_receipt":
        value["completed_receipt_bindings"][5]["formal_receipt"]["decision"] = "pass"
    elif field == "runtime_queue":
        value["runtime_binding"]["queue_content_sha256"] = "0" * 64
    elif field == "terminal_queue":
        value["terminal_runtime_archive"]["queue"]["next_ordinal"] -= 1
    elif field == "replay_dependency":
        value["replay_dependencies"]["replay_dependency_root_sha256"] = "0" * 64
    else:
        value["interpretation"] = "No formal pass."
    with pytest.raises(ValueError, match="execution result"):
        validate_execution_result(service_sealed(value), ROOT, CONFIG)


def test_owned_child_deadline_terminates_only_campaign_child() -> None:
    from sigma_theory_compiler.continuous_scientific_pipeline_service import _run_owned_child

    before = {child.pid for child in multiprocessing.active_children()}
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="hard action deadline"):
        _run_owned_child(
            _deadline_sleeping_worker,
            (),
            maximum_seconds=0.5,
            action_name="formal validation",
        )
    assert time.monotonic() - started < 1.5
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_owned_child_result_and_error_are_fail_closed() -> None:
    from sigma_theory_compiler.continuous_scientific_pipeline_service import _run_owned_child

    assert _run_owned_child(
        _deadline_success_worker,
        (),
        maximum_seconds=2,
        action_name="formal validation",
    ) == {"bounded": True}
    with pytest.raises(RuntimeError, match="exact formal failure"):
        _run_owned_child(
            _deadline_error_worker,
            (),
            maximum_seconds=2,
            action_name="formal validation",
        )


def test_lease_is_exclusive_and_identity_bound(tmp_path: Path) -> None:
    config = load_service_config(ROOT, CONFIG)
    first, owned = acquire_lease(tmp_path, config, ["python", "--run"])
    with pytest.raises(RuntimeError, match="already exists"):
        acquire_lease(tmp_path, config, ["python", "--run"])
    tampered = json.loads(first.read_text())
    tampered["pid"] += 1
    first.write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError, match="ownership changed"):
        release_lease(first, owned)


def test_receipt_tamper_and_formal_stage_fail_closed(tmp_path: Path) -> None:
    config = load_service_config(ROOT, CONFIG)
    queue = initial_queue(config)
    with pytest.raises(ValueError, match="receipt hash"):
        payload = _generation()
        payload["receipt"]["unique_formula_count"] = 7
        apply_action(queue, "generate_and_screen", payload)
    queue = apply_action(queue, "generate_and_screen", _generation())
    backend = load_backend_config(ROOT, ROOT / config["formal_backend_config_path"])
    receipt, evidence = build_formal_evidence(
        queue["generated_receipt"],
        queue["generation_manifest"],
        backend,
        root=ROOT,
        output_root=tmp_path / "formal",
    )
    queue = apply_action(queue, "formal_validate", {"receipt": receipt, "evidence": evidence})
    assert queue["formal_receipt"]["decision"] == "block"
    assert queue["leaderboard_rebuild_requests"] == []
    queue = apply_action(queue, "wait", None)
    assert queue["generated_receipt"] is None
    assert len(queue["completed_action_receipts"]) == 1
    with pytest.raises(ValueError, match="state content hash"):
        apply_action({**queue, "invented": True}, "wait", None)


def test_bounded_resume_stop_and_resource_backoff(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    config["runtime_directory"] = "runtime"
    config["maximum_cycles"] = 1
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    admission = json.loads((ROOT / config["admission_config_path"]).read_text())
    (tmp_path / "configs").mkdir()
    (tmp_path / config["admission_config_path"]).write_text(json.dumps(admission))
    _copy_backend_dependencies(tmp_path, config)
    calls = []
    result = run_bounded_service(
        tmp_path,
        path,
        resource_probe=lambda: (92.0, 65536),
        generation_executor=lambda q, c: calls.append(1) or _generation(),
        formal_executor=lambda r, q, c: pytest.fail("formal executor should not run"),
        argv=["test"],
    )
    assert result["cycles"] == 1 and calls == []
    assert (
        json.loads((tmp_path / "runtime/queue.json").read_text())["next_ordinal"]
        == config["start_ordinal"]
    )
    (tmp_path / "runtime/stop.request").write_text("")
    result = run_bounded_service(
        tmp_path,
        path,
        resource_probe=lambda: (0.0, 65536),
        generation_executor=lambda q, c: _generation(),
        formal_executor=lambda r, q, c: pytest.fail("formal executor should not run"),
        argv=["test"],
    )
    assert result["state"] == "stop_requested"
