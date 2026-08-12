import itertools
import json
from pathlib import Path

import pytest

import sigma_theory_compiler.continuous_scientific_pipeline_epoch as epoch_module
from sigma_theory_compiler.continuous_scientific_pipeline_epoch import (
    _genesis_checkpoint,
    build_epoch_genesis,
    load_epoch_config,
    materialize_epoch_runtime,
    normalize_bounded_pause,
    run_epoch_once,
    validate_epoch_genesis,
)
from sigma_theory_compiler.continuous_scientific_pipeline_service import (
    _sealed as service_sealed,
)
from sigma_theory_compiler.continuous_scientific_pipeline_service import initial_queue

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/continuous_scientific_pipeline_epoch_003.json"
ARTIFACT = ROOT / "runs/engine/continuous-scientific-pipeline-epoch-003-genesis.json"


def test_epoch_genesis_is_exact_disjoint_and_not_executed() -> None:
    value = build_epoch_genesis(ROOT, CONFIG)
    assert value == json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_epoch_genesis(value, ROOT, CONFIG)
    predecessor = json.loads((ROOT / value["predecessor"]["path"]).read_text(encoding="utf-8"))
    assert value["coverage"]["start_ordinal"] == predecessor["coverage"]["stop_ordinal_exclusive"]
    assert value["coverage"]["unique_formula_count"] == 3_932_160
    assert value["coverage"]["batch_count"] == 8
    intervals = value["coverage"]["intervals"]
    assert intervals[0]["start_ordinal"] == value["coverage"]["start_ordinal"]
    assert intervals[-1]["stop_ordinal_exclusive"] == value["coverage"]["stop_ordinal_exclusive"]
    assert all(
        left["stop_ordinal_exclusive"] == right["start_ordinal"]
        for left, right in itertools.pairwise(intervals)
    )
    assert value["execution_state"] == {
        "runtime_materialized": False,
        "formulas_evaluated": 0,
        "formal_receipts": 0,
        "epoch_complete": False,
    }
    assert not any(value["promotion_contract"].values())
    assert not any(value["seals"].values())


def test_epoch_predecessor_and_genesis_tamper_fail_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["predecessor_result"]["file_sha256"] = "0" * 64
    path = tmp_path / "epoch.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="binding file hash"):
        load_epoch_config(ROOT, path)

    value = build_epoch_genesis(ROOT, CONFIG)
    value["coverage"]["start_ordinal"] += 1
    with pytest.raises(ValueError, match="content hash"):
        validate_epoch_genesis(value, ROOT, CONFIG)


def test_runtime_genesis_is_atomic_idempotent_and_mismatch_closed(tmp_path: Path) -> None:
    value = build_epoch_genesis(ROOT, CONFIG)
    runtime = materialize_epoch_runtime(tmp_path, value)
    assert materialize_epoch_runtime(tmp_path, value) == runtime
    service_config = value["derived_service_config"]
    queue = json.loads((runtime / service_config["queue_name"]).read_text(encoding="utf-8"))
    checkpoint = json.loads(
        (runtime / service_config["checkpoint_name"]).read_text(encoding="utf-8")
    )
    persistent_config = json.loads((runtime / "service-config.json").read_text(encoding="utf-8"))
    assert queue == value["initial_queue"]
    assert checkpoint == _genesis_checkpoint(service_config, queue)
    assert checkpoint["state"] == "bounded_pause"
    assert persistent_config["service_config"] == service_config
    assert persistent_config["service_config_sha256"] == queue["service_config_sha256"]
    queue["next_ordinal"] += 1
    (runtime / service_config["queue_name"]).write_text(json.dumps(queue), encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-matching persistent state"):
        materialize_epoch_runtime(tmp_path, value)


def test_incomplete_false_completion_is_normalized_to_resumable_pause(tmp_path: Path) -> None:
    value = build_epoch_genesis(ROOT, CONFIG)
    runtime = materialize_epoch_runtime(tmp_path, value)
    service_config = value["derived_service_config"]
    checkpoint_path = runtime / service_config["checkpoint_name"]
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint_body = {key: item for key, item in checkpoint.items() if key != "content_sha256"}
    checkpoint_body["state"] = "bounded_complete"
    checkpoint_path.write_text(json.dumps(service_sealed(checkpoint_body)), encoding="utf-8")
    normalized = normalize_bounded_pause(runtime, service_config)
    assert normalized["state"] == "bounded_pause"
    assert normalized["queue_content_sha256"] == value["initial_queue"]["content_sha256"]


def test_exact_terminal_queue_is_normalized_to_complete(tmp_path: Path) -> None:
    value = build_epoch_genesis(ROOT, CONFIG)
    runtime = materialize_epoch_runtime(tmp_path, value)
    service_config = value["derived_service_config"]
    queue_path = runtime / service_config["queue_name"]
    checkpoint_path = runtime / service_config["checkpoint_name"]
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue_body = {key: item for key, item in queue.items() if key != "content_sha256"}
    queue_body["next_ordinal"] = queue_body["stop_ordinal_exclusive"]
    queue = service_sealed(queue_body)
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint_body = {key: item for key, item in checkpoint.items() if key != "content_sha256"}
    checkpoint_body["queue_content_sha256"] = queue["content_sha256"]
    checkpoint_body["state"] = "bounded_pause"
    checkpoint_path.write_text(json.dumps(service_sealed(checkpoint_body)), encoding="utf-8")
    assert normalize_bounded_pause(runtime, service_config)["state"] == "bounded_complete"


def test_one_bounded_invocation_persists_pause_and_removes_transient_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    genesis = build_epoch_genesis(ROOT, CONFIG)
    service_config = {**genesis["derived_service_config"], "runtime_directory": "runtime/epoch-003"}
    genesis_body = {key: item for key, item in genesis.items() if key != "content_sha256"}
    genesis_body["derived_service_config"] = service_config
    genesis_body["initial_queue"] = initial_queue(service_config)
    genesis = epoch_module._sealed(genesis_body)

    def fake_bounded_service(root: Path, config_path: Path, **kwargs: object) -> dict[str, object]:
        del kwargs
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        assert loaded == service_config
        runtime = root / service_config["runtime_directory"]
        checkpoint_path = runtime / service_config["checkpoint_name"]
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        body = {key: item for key, item in checkpoint.items() if key != "content_sha256"}
        body["cycles"] = 1
        body["state"] = "bounded_complete"
        rewritten = service_sealed(body)
        checkpoint_path.write_text(json.dumps(rewritten), encoding="utf-8")
        return rewritten

    monkeypatch.setattr(epoch_module, "build_epoch_genesis", lambda root, path: genesis)
    monkeypatch.setattr(epoch_module, "run_bounded_service", fake_bounded_service)
    checkpoint = run_epoch_once(
        tmp_path,
        tmp_path / "unused.json",
        resource_probe=lambda: (92.0, 65536),
    )
    assert checkpoint["cycles"] == 1
    assert checkpoint["state"] == "bounded_pause"
    runtime = tmp_path / service_config["runtime_directory"]
    assert not (runtime / ".active-service-config.json").exists()
