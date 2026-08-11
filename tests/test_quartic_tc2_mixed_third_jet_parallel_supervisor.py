from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_mixed_third_jet_continuation_service import (
    _with_hash as continuation_with_hash,
)
from sigma_theory_compiler.quartic_tc2_mixed_third_jet_parallel_supervisor import (
    FALSE_CLAIMS,
    QuarticTC2MixedThirdJetParallelSupervisorError,
    _with_hash,
    export_supervisor,
    request_stop,
    run_supervisor,
    supervisor_status,
)

ROOT = Path(__file__).resolve().parents[1]
EPOCH_CONFIG = (
    ROOT
    / "configs/backgrounds/quartic_tc2_mixed_third_jet_parallel_continuation_service.json"
)
SUPERVISOR_CONFIG = (
    ROOT / "configs/backgrounds/quartic_tc2_mixed_third_jet_parallel_supervisor.json"
)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint(*, offset: int = 320, remaining: int = 11_980) -> dict[str, object]:
    return continuation_with_hash(
        {
            "schema_version": "sigma-quartic-tc2-mixed-third-jet-continuation-checkpoint-1.0",
            "config_content_sha256": "epoch-config",
            "initial_prior_file_sha256": "0" * 64,
            "initial_prior_content_sha256": "1" * 64,
            "initial_chunk_config_file_sha256": "2" * 64,
            "initial_chunk_config_content_sha256": "3" * 64,
            "next_offset": offset,
            "remaining_mixed_triples": remaining,
            "prior_resume_sha256": f"{offset:064x}",
            "current_artifact_path": "chunks/latest.json",
            "current_artifact_content_sha256": "4" * 64,
            "current_artifact_file_sha256": "5" * 64,
            "completed_chunks": 0,
            "permanently_stopped": False,
            "stop_reason": None,
            "history": [],
            "claims": dict(FALSE_CLAIMS),
        }
    )


def _fixture(tmp_path: Path, *, maximum_epochs: int = 2) -> tuple[Path, Path, Path]:
    epoch_config = tmp_path / "configs/backgrounds/epoch.json"
    epoch_config.parent.mkdir(parents=True, exist_ok=True)
    epoch_config.write_bytes(EPOCH_CONFIG.read_bytes())
    output = tmp_path / "runs/epoch"
    checkpoint = output / "checkpoint.json"
    _write_json(checkpoint, _checkpoint())

    source = json.loads(SUPERVISOR_CONFIG.read_text(encoding="utf-8"))
    source.update(
        {
            "epoch_config_path": epoch_config.relative_to(tmp_path).as_posix(),
            "epoch_output_path": output.relative_to(tmp_path).as_posix(),
            "supervisor_output_path": "runs/supervisor",
            "maximum_epochs_per_run": maximum_epochs,
            "poll_interval_seconds": 0,
        }
    )
    source = _with_hash(source)
    config = tmp_path / "configs/backgrounds/supervisor.json"
    _write_json(config, source)
    return config, output, checkpoint


def _fake_epoch_runner(
    project_root: Path, *, config_path: Path, output_path: Path
) -> dict[str, object]:
    del project_root, config_path
    checkpoint_path = output_path / "checkpoint.json"
    before = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    record = {
        "offset": before["next_offset"],
        "next_offset": before["next_offset"] + 64,
        "processed_count": 64,
    }
    after = continuation_with_hash(
        {
            **{key: value for key, value in before.items() if key != "content_sha256"},
            "next_offset": before["next_offset"] + 64,
            "remaining_mixed_triples": before["remaining_mixed_triples"] - 64,
            "prior_resume_sha256": f"{before['next_offset'] + 64:064x}",
            "completed_chunks": before["completed_chunks"] + 1,
            "history": [*before["history"], record],
        }
    )
    _write_json(checkpoint_path, after)
    return {
        "status": "checkpointed",
        "reason": "chunk_limit",
        "chunks_advanced": 1,
        "next_offset": after["next_offset"],
    }


def test_supervisor_advances_exactly_bounded_epochs_and_resumes(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path, maximum_epochs=2)
    state = run_supervisor(
        tmp_path,
        config,
        epoch_runner=_fake_epoch_runner,
        sleep=lambda _: None,
    )
    assert state["state"] == "stopped"
    assert state["stop_reason"] == "epoch_limit"
    assert state["epochs_completed"] == 2
    assert state["chunks_advanced"] == 2
    assert state["next_offset"] == 448
    assert state["remaining_mixed_triples"] == 11_852
    assert state["claims"] == FALSE_CLAIMS

    resumed = run_supervisor(
        tmp_path,
        config,
        epoch_runner=_fake_epoch_runner,
        sleep=lambda _: None,
    )
    assert resumed["epochs_completed"] == 4
    assert resumed["chunks_advanced"] == 4
    assert resumed["next_offset"] == 576
    status = supervisor_status(tmp_path, config)
    assert status["next_offset"] == 576
    assert status["alive"] is False
    assert all(value is False for value in status["claims"].values())
    exported = export_supervisor(
        tmp_path, config, tmp_path / "runs/engine/portable-supervisor.json"
    )
    assert exported["status"] == "portable_stopped_checkpoint"
    assert exported["lifecycle"]["next_offset"] == 576
    assert exported["lifecycle"]["resume_available"] is True
    assert exported["parallel_contract"]["parallel_worker_count"] == 8
    assert all(value is False for value in exported["claims"].values())


def test_supervisor_stop_request_is_graceful_and_nonmutating(tmp_path: Path) -> None:
    config, _, checkpoint = _fixture(tmp_path)
    before = checkpoint.read_bytes()
    requested = request_stop(tmp_path, config)
    assert requested["state"] == "stop_requested"

    called = False

    def forbidden(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        nonlocal called
        called = True
        raise AssertionError("epoch runner must not be called")

    state = run_supervisor(tmp_path, config, epoch_runner=forbidden)
    assert state["stop_reason"] == "external_stop_requested"
    assert called is False
    assert checkpoint.read_bytes() == before


def test_supervisor_stops_before_history_budget_is_exceeded(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path, maximum_epochs=2)
    data = json.loads(config.read_text(encoding="utf-8"))
    data["maximum_history_records"] = 1
    _write_json(config, _with_hash(data))
    state = run_supervisor(
        tmp_path,
        config,
        epoch_runner=_fake_epoch_runner,
        sleep=lambda _: None,
    )
    assert state["epochs_completed"] == 1
    assert state["chunks_advanced"] == 1
    assert state["stop_reason"] == "supervisor_history_limit"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("parallel_worker_count", 7, "safety contract"),
        ("parallel_execution_policy", "unordered", "safety contract"),
        ("external_paid_llm_calls", True, "safety contract"),
        ("full_tube_Sylvester_identity", "pass", "safety contract"),
        ("epoch_config_file_sha256", "0" * 64, "epoch config binding"),
    ],
)
def test_supervisor_rejects_contract_tamper(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    config, _, _ = _fixture(tmp_path)
    data = json.loads(config.read_text(encoding="utf-8"))
    data[field] = value
    _write_json(config, _with_hash(data))
    with pytest.raises(
        QuarticTC2MixedThirdJetParallelSupervisorError, match=message
    ):
        run_supervisor(tmp_path, config, epoch_runner=_fake_epoch_runner)


def test_supervisor_rejects_checkpoint_and_live_lock_tamper(tmp_path: Path) -> None:
    config, _, checkpoint = _fixture(tmp_path)
    data = json.loads(checkpoint.read_text(encoding="utf-8"))
    data["remaining_mixed_triples"] -= 1
    _write_json(checkpoint, data)
    with pytest.raises(
        QuarticTC2MixedThirdJetParallelSupervisorError,
        match="checkpoint contract",
    ):
        run_supervisor(tmp_path, config, epoch_runner=_fake_epoch_runner)

    _write_json(checkpoint, _checkpoint())
    lock = tmp_path / "runs/supervisor/supervisor.lock"
    _write_json(lock, {"pid": os.getpid()})
    with pytest.raises(
        QuarticTC2MixedThirdJetParallelSupervisorError,
        match="already running",
    ):
        run_supervisor(tmp_path, config, epoch_runner=_fake_epoch_runner)
