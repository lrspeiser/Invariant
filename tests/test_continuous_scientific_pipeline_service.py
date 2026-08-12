import json
from pathlib import Path

import pytest

from sigma_theory_compiler.continuous_scientific_pipeline_service import (
    acquire_lease,
    apply_action,
    build_readiness,
    initial_queue,
    load_service_config,
    release_lease,
    run_bounded_service,
    validate_readiness,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/continuous_scientific_pipeline_service.json"
ARTIFACT = ROOT / "runs/engine/continuous-scientific-pipeline-service-readiness.json"


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


def test_readiness_exact_and_not_started() -> None:
    value = build_readiness(ROOT, CONFIG)
    assert value == json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_readiness(value, ROOT, CONFIG)
    assert value["execution_state"] == {
        "service_started": False,
        "cycles_executed": 0,
        "queue_created": False,
        "live_SQLite_accessed": False,
    }
    assert not any(value["seals"].values())


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


def test_receipt_tamper_and_formal_stage_fail_closed() -> None:
    config = load_service_config(ROOT, CONFIG)
    queue = initial_queue(config)
    with pytest.raises(ValueError, match="receipt hash"):
        apply_action(queue, "generate_and_screen", {**_receipt(), "unique_formula_count": 7})
    queue = apply_action(queue, "generate_and_screen", _receipt())
    queue = apply_action(queue, "formal_validate", None)
    assert queue["formal_receipt"]["decision"] == "block"
    assert queue["leaderboard_rebuild_requests"] == []
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
    calls = []
    result = run_bounded_service(
        tmp_path,
        path,
        resource_probe=lambda: (92.0, 65536),
        generation_executor=lambda q, c: calls.append(1) or _receipt(),
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
        generation_executor=lambda q, c: _receipt(),
        argv=["test"],
    )
    assert result["state"] == "stop_requested"
