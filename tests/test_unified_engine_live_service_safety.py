from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path

import pytest

import sigma_theory_compiler.unified_engine_live_service_safety as safety

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/unified_engine_live_service_safety.json"
ARTIFACT = ROOT / "runs/engine/unified-engine-live-service-safety-readiness.json"


def _config() -> dict:
    return safety.load_safety_config(ROOT, CONFIG)


def _checkpoint(config: dict) -> dict:
    value = {
        "schema_version": safety.CHECKPOINT_SCHEMA,
        "runtime_epoch": config["runtime_epoch"],
        "runtime_directory": config["runtime_directory"],
        "config_file_sha256": "a" * 64,
        "worker_argv_sha256": "b" * 64,
        "state": "running",
        "pid": 1,
        "refresh_count": 19,
        "consecutive_failures": 2,
        "reload_count": 3,
        "last_reload_utc": None,
        "last_error": None,
        "last_refresh": None,
        "stop_reason": None,
        "updated_utc": "2026-08-11T00:00:00+00:00",
    }
    return json.loads(safety._checkpoint_payload(value))


def _runtime_paths(tmp_path: Path) -> dict[str, Path]:
    runtime = tmp_path / "runtime"
    lease = tmp_path / "cutover.lease.json"
    return {
        "runtime": runtime,
        "checkpoint": runtime / "checkpoint.json",
        "snapshot": runtime / "status.json",
        "dashboard": runtime / "dashboard.html",
        "stop": runtime / "stop.request",
        "log": runtime / "service.log",
        "lease": lease,
        "lease_recovery": tmp_path / "cutover.lease.json.recovery",
        "legacy_checkpoint": tmp_path / "legacy" / "checkpoint.json",
        "legacy_snapshot": tmp_path / "legacy" / "unified-engine-status-live.json",
        "checked_snapshot": tmp_path / "checked" / "unified-engine-status.json",
    }


def _write_checkpoint(path: Path, config: dict, *, state: str = "starting") -> None:
    checkpoint = _checkpoint(config)
    checkpoint.update(state=state, pid=None if state == "starting" else os.getpid())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(safety._checkpoint_payload(checkpoint))


def _write_worker_lease(path: Path, config: dict, identity: str) -> None:
    now = "2026-08-11T00:00:00+00:00"
    lease = {
        "schema_version": safety.LEASE_SCHEMA,
        "runtime_epoch": config["runtime_epoch"],
        "runtime_directory": config["runtime_directory"],
        "worker_argv_sha256": identity,
        "owner_kind": "worker",
        "owner_pid": os.getpid(),
        "owner_argv_sha256": identity,
        "created_utc": now,
        "updated_utc": now,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(safety._lease_payload(lease))


def _seed_fixture(
    tmp_path: Path, *, roots: list[str] | None = None
) -> tuple[Path, dict, Path]:
    root = tmp_path / "seed-root"
    unified = root / "configs" / "unified.json"
    unified.parent.mkdir(parents=True)
    unified.write_text(json.dumps({"sources": []}), encoding="utf-8")
    config = copy.deepcopy(_config())
    config["unified_config_path"] = "configs/unified.json"
    roots = roots or ["a" * 64]
    history = [
        {"category_roots": {"solar": "b" * 64}, "leaderboard_root_sha256": item}
        for item in roots
    ]
    leaderboard = {
        "schema_version": safety.EXPECTED_LEADERBOARD_SCHEMA,
        "leaderboard_root_sha256": roots[-1],
        "history": history,
    }
    leaderboard["content_sha256"] = safety._content_sha(leaderboard)
    core = {
        "scientific_leaderboards": leaderboard,
        "source_revisions": {
            "fixture": {
                "file_sha256": "d" * 64,
                "content_sha256": "e" * 64,
            }
        },
    }
    snapshot = {"core": core, "core_content_sha256": safety._sha(core)}
    path = root / "runs" / "seed.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    return root, config, path


def test_config_is_fixed_epoch_bounded_and_predecessor_bound() -> None:
    config = _config()
    assert config["runtime_directory"] == safety.EXPECTED_RUNTIME_DIRECTORY
    assert config["legacy_runtime_directory"] == safety.EXPECTED_LEGACY_RUNTIME_DIRECTORY
    assert config["cutover_lease_path"].endswith(".lease.json")
    assert config["startup_identity_timeout_seconds"] == 10
    assert config["control_poll_interval_seconds"] == 0.25
    assert config["maximum_refreshes"] == 4032
    assert config["data_seals"] == safety.EXPECTED_SEALS
    drift = copy.deepcopy(config)
    drift["runtime_directory"] = "runs/engine/new-reset-epoch"
    path = ROOT / "configs/unified_engine_live_service_safety.runtime-drift.test.json"
    try:
        path.write_text(json.dumps(drift), encoding="utf-8")
        with pytest.raises(ValueError, match="runtime_directory epoch drift"):
            safety.load_safety_config(ROOT, path)
    finally:
        path.unlink(missing_ok=True)


def test_checkpoint_hash_epoch_and_counters_fail_closed() -> None:
    config = _config()
    checkpoint = _checkpoint(config)
    safety._validate_checkpoint(checkpoint, config)
    checkpoint["refresh_count"] = 0
    with pytest.raises(ValueError, match="content hash"):
        safety._validate_checkpoint(checkpoint, config)
    checkpoint = _checkpoint(config)
    checkpoint["runtime_epoch"] = "different-epoch"
    checkpoint["content_sha256"] = safety._sha(
        {key: item for key, item in checkpoint.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="epoch mismatch"):
        safety._validate_checkpoint(checkpoint, config)


def test_compatible_reload_preserves_epoch_and_all_counters(tmp_path: Path) -> None:
    current = _config()
    reloaded = copy.deepcopy(current)
    reloaded["refresh_interval_seconds"] = 301
    path = tmp_path / "reload.json"
    path.write_text(json.dumps(reloaded), encoding="utf-8")
    checkpoint = _checkpoint(current)
    before = {key: checkpoint[key] for key in ("refresh_count", "consecutive_failures")}
    new_config, config_sha = safety._reload_preserving_epoch(
        ROOT, path, current, checkpoint
    )
    assert new_config["refresh_interval_seconds"] == 301
    assert checkpoint["refresh_count"] == before["refresh_count"]
    assert checkpoint["consecutive_failures"] == before["consecutive_failures"]
    assert checkpoint["reload_count"] == 4
    assert checkpoint["config_file_sha256"] == config_sha


def test_worker_argv_preserves_windows_spaced_tokens(tmp_path: Path) -> None:
    root = tmp_path / "project with spaces"
    config = root / "configs" / "live safety config.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")
    command = safety._worker_command(root, config)
    assert command[0] == os.fspath(Path(safety.sys.executable))
    assert command[command.index("--project-root") + 1] == str(root.resolve())
    assert command[command.index("--config") + 1] == "configs/live safety config.json"
    assert safety._argv_identity(command) == safety._argv_identity(["launcher", *command[1:]])
    assert "project with spaces" not in " ".join(command[:4])


def test_pid_identity_rejects_live_unrelated_process(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = safety._argv_identity(safety._worker_command(ROOT, CONFIG))

    class Process:
        def __init__(self, pid: int) -> None:
            assert pid == 42

        def cmdline(self) -> list[str]:
            return ["python", "-m", "unrelated.module", "worker"]

    import psutil

    monkeypatch.setattr(safety, "pid_alive", lambda pid: pid == 42)
    monkeypatch.setattr(psutil, "Process", Process)
    assert safety._pid_matches_worker(42, expected) is False


def test_secure_atomic_write_ignores_predictable_temp_symlink(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint.json"
    predictable = tmp_path / "checkpoint.json.tmp"
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"safe")
    try:
        predictable.symlink_to(victim)
    except OSError:
        pytest.skip("symlink creation unavailable")
    safety._safe_atomic_write(target, b"new", 1024)
    assert target.read_bytes() == b"new"
    assert victim.read_bytes() == b"safe"
    assert predictable.is_symlink()
    target.unlink()
    target.symlink_to(victim)
    with pytest.raises(RuntimeError, match="target symlink"):
        safety._safe_atomic_write(target, b"bad", 1024)
    assert victim.read_bytes() == b"safe"


def test_secure_atomic_write_uses_exclusive_unpredictable_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "status.json"
    predictable = tmp_path / "status.json.tmp"
    predictable.write_bytes(b"attacker-controlled")
    created: list[str] = []
    original = safety.tempfile.mkstemp

    def recording_mkstemp(*, prefix: str, suffix: str, dir: Path):
        descriptor, name = original(prefix=prefix, suffix=suffix, dir=dir)
        created.append(name)
        return descriptor, name

    monkeypatch.setattr(safety.tempfile, "mkstemp", recording_mkstemp)
    safety._safe_atomic_write(target, b"published", 1024)
    assert target.read_bytes() == b"published"
    assert predictable.read_bytes() == b"attacker-controlled"
    assert len(created) == 1
    assert Path(created[0]).name != predictable.name


def test_interruptible_wait_observes_control_without_full_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def guard() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise safety.ControlSignal("external_stop_requested")

    monkeypatch.setattr(safety.time, "sleep", lambda seconds: None)
    with pytest.raises(safety.ControlSignal, match="external_stop_requested"):
        safety._interruptible_wait(300.0, 0.25, guard)
    assert calls == 3


def test_stale_projection_manifest_aborts_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    paths = _runtime_paths(tmp_path)
    seed = tmp_path / "seed.json"
    seed.write_text("{}", encoding="utf-8")
    manifests = iter(
        [
            {"file_sha256": {"a": "1"}, "manifest_sha256": "1" * 64},
            {"file_sha256": {"a": "2"}, "manifest_sha256": "2" * 64},
        ]
    )
    monkeypatch.setattr(safety, "_paths", lambda root, value: paths)
    monkeypatch.setattr(safety, "_dependency_manifest", lambda *args: next(manifests))
    monkeypatch.setattr(safety, "load_config", lambda path: {})
    monkeypatch.setattr(safety, "load_leaderboard_config", lambda path: {})
    monkeypatch.setattr(
        safety,
        "_select_previous_leaderboard",
        lambda *args: (
            {"history": []},
            {
                "path": "seed.json",
                "file_sha256": safety._file_sha(seed),
                "source_kind": "fixture",
            },
        ),
    )
    snapshot = {
        "core": {"data_seals": {}},
        "volatile": {"sampled_at_utc": "fixture"},
    }
    with pytest.raises(safety.ControlSignal, match="projection_inputs_changed"):
        safety.safe_refresh_once(
            tmp_path,
            config,
            lambda: None,
            snapshot_builder=lambda *args: copy.deepcopy(snapshot),
            leaderboard_builder=lambda *args: {},
            dashboard_renderer=lambda value: "dashboard",
        )
    assert not paths["snapshot"].exists()
    assert not paths["dashboard"].exists()


def test_checked_leaderboard_seed_validation_fails_closed_on_tamper_schema_and_size(
    tmp_path: Path,
) -> None:
    root, config, path = _seed_fixture(tmp_path, roots=["a" * 64, "c" * 64])
    leaderboard, receipt = safety._validated_snapshot_leaderboard(
        root, config, path, "checked_snapshot"
    )
    assert len(leaderboard["history"]) == 2
    assert receipt["history_entry_count"] == 2
    valid = json.loads(path.read_text(encoding="utf-8"))

    tampered = copy.deepcopy(valid)
    tampered["core"]["scientific_leaderboards"]["history"][0][
        "leaderboard_root_sha256"
    ] = "d" * 64
    tampered["core_content_sha256"] = safety._sha(tampered["core"])
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash mismatch"):
        safety._validated_snapshot_leaderboard(
            root, config, path, "checked_snapshot"
        )

    incompatible = copy.deepcopy(valid)
    incompatible["core"]["scientific_leaderboards"]["schema_version"] = "other"
    incompatible["core"]["scientific_leaderboards"][
        "content_sha256"
    ] = safety._content_sha(incompatible["core"]["scientific_leaderboards"])
    incompatible["core_content_sha256"] = safety._sha(incompatible["core"])
    path.write_text(json.dumps(incompatible), encoding="utf-8")
    with pytest.raises(ValueError, match="schema incompatible"):
        safety._validated_snapshot_leaderboard(
            root, config, path, "checked_snapshot"
        )

    oversized = copy.deepcopy(valid)
    oversized_history = [
        {
            "category_roots": {"solar": "b" * 64},
            "leaderboard_root_sha256": f"{index:064x}",
        }
        for index in range(safety.MAXIMUM_SEED_HISTORY_ENTRIES + 1)
    ]
    oversized["core"]["scientific_leaderboards"]["history"] = oversized_history
    oversized["core"]["scientific_leaderboards"][
        "leaderboard_root_sha256"
    ] = oversized_history[-1]["leaderboard_root_sha256"]
    oversized["core"]["scientific_leaderboards"][
        "content_sha256"
    ] = safety._content_sha(oversized["core"]["scientific_leaderboards"])
    oversized["core_content_sha256"] = safety._sha(oversized["core"])
    path.write_text(json.dumps(oversized), encoding="utf-8")
    with pytest.raises(ValueError, match="history oversized"):
        safety._validated_snapshot_leaderboard(
            root, config, path, "checked_snapshot"
        )

    path.unlink()
    with pytest.raises(ValueError, match="seed missing"):
        safety._validated_snapshot_leaderboard(
            root, config, path, "checked_snapshot"
        )


def test_truncated_runtime_history_is_replaced_by_compatible_checked_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    paths = _runtime_paths(tmp_path)
    checked = {"history": [{"id": 1}, {"id": 2}, {"id": 3}]}
    runtime = {"history": [{"id": 3}]}
    checked_receipt = {"source_kind": "checked_snapshot"}
    runtime_receipt = {"source_kind": "runtime_snapshot"}
    monkeypatch.setattr(
        safety,
        "_checked_or_legacy_leaderboard_seed",
        lambda *args: (checked, checked_receipt),
    )
    monkeypatch.setattr(
        safety, "_runtime_leaderboard_seed", lambda *args: (runtime, runtime_receipt)
    )
    selected, receipt = safety._select_previous_leaderboard(ROOT, config, paths)
    assert selected is checked
    assert receipt is checked_receipt

    runtime["history"] = [{"id": 2}, {"id": 3}, {"id": 4}]
    selected, receipt = safety._select_previous_leaderboard(ROOT, config, paths)
    assert selected is runtime
    assert receipt is runtime_receipt

    runtime["history"] = [{"id": 9}]
    with pytest.raises(ValueError, match="histories are incompatible"):
        safety._select_previous_leaderboard(ROOT, config, paths)


def test_checked_and_legacy_seed_missing_fails_closed(tmp_path: Path) -> None:
    config = _config()
    paths = _runtime_paths(tmp_path)
    with pytest.raises(ValueError, match="checked and legacy leaderboard seeds missing"):
        safety._checked_or_legacy_leaderboard_seed(ROOT, config, paths)


def test_seed_hash_change_aborts_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    paths = _runtime_paths(tmp_path)
    seed = tmp_path / "seed.json"
    seed.write_text("seed", encoding="utf-8")
    manifest = {"file_sha256": {"a": "1"}, "manifest_sha256": "1" * 64}
    monkeypatch.setattr(safety, "_paths", lambda root, value: paths)
    monkeypatch.setattr(safety, "_dependency_manifest", lambda *args: manifest)
    monkeypatch.setattr(safety, "load_config", lambda path: {})
    monkeypatch.setattr(safety, "load_leaderboard_config", lambda path: {})
    monkeypatch.setattr(
        safety,
        "_select_previous_leaderboard",
        lambda *args: (
            {"history": []},
            {
                "path": "seed.json",
                "file_sha256": safety._file_sha(seed),
                "source_kind": "fixture",
            },
        ),
    )

    def mutate_seed(value: dict) -> str:
        seed.write_text("changed", encoding="utf-8")
        return "dashboard"

    with pytest.raises(safety.ControlSignal, match="seed_changed_during_refresh"):
        safety.safe_refresh_once(
            tmp_path,
            config,
            lambda: None,
            snapshot_builder=lambda *args: {"core": {}, "volatile": {}},
            leaderboard_builder=lambda *args: {},
            dashboard_renderer=mutate_seed,
        )
    assert not paths["snapshot"].exists()
    assert not paths["dashboard"].exists()


def test_reload_failure_is_written_to_final_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    paths = _runtime_paths(tmp_path)
    identity = safety._argv_identity(safety._worker_command(ROOT, CONFIG))
    _write_checkpoint(paths["checkpoint"], config)
    _write_worker_lease(paths["lease"], config, identity)
    monkeypatch.setattr(safety, "load_safety_config", lambda root, path: config)
    monkeypatch.setattr(safety, "_paths", lambda root, value: paths)
    monkeypatch.setattr(safety, "_assert_no_legacy_worker", lambda *args: None)
    monkeypatch.setattr(
        safety,
        "safe_refresh_once",
        lambda *args, **kwargs: (_ for _ in ()).throw(safety.ControlSignal("configuration_changed")),
    )
    monkeypatch.setattr(
        safety,
        "_reload_preserving_epoch",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad reload")),
    )
    safety.run_worker(ROOT, CONFIG)
    checkpoint = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
    assert checkpoint["state"] == "stopped"
    assert checkpoint["stop_reason"] == "configuration_reload_failed"
    assert checkpoint["last_error"] == "ValueError: bad reload"
    assert not paths["lease"].exists()


def test_legacy_worker_present_rejects_start_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    paths = _runtime_paths(tmp_path)
    monkeypatch.setattr(safety, "load_safety_config", lambda root, path: config)
    monkeypatch.setattr(safety, "_paths", lambda root, value: paths)
    monkeypatch.setattr(safety, "_current_exact_argv_identity", lambda: "c" * 64)
    monkeypatch.setattr(safety, "_worker_pids", lambda identity: [42])
    monkeypatch.setattr(
        safety.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("worker spawn must not occur"),
    )
    with pytest.raises(RuntimeError, match=r"legacy .* worker present: \[42\]"):
        safety.start_service(ROOT, CONFIG)
    assert not paths["lease"].exists()
    assert not paths["checkpoint"].exists()


def test_exclusive_lease_prevents_repeated_concurrent_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    paths = _runtime_paths(tmp_path)
    launches: list[list[str]] = []

    class Process:
        pid = 4242

    def popen(command: list[str], **kwargs):
        checkpoint = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
        assert checkpoint["state"] == "starting"
        assert checkpoint["pid"] is None
        assert paths["lease"].is_file()
        launches.append(command)
        return Process()

    monkeypatch.setattr(safety, "load_safety_config", lambda root, path: config)
    monkeypatch.setattr(safety, "_paths", lambda root, value: paths)
    monkeypatch.setattr(safety, "_current_exact_argv_identity", lambda: "c" * 64)
    monkeypatch.setattr(safety, "_assert_no_legacy_worker", lambda *args: None)
    monkeypatch.setattr(safety, "_open_log", lambda path: io.BytesIO())
    monkeypatch.setattr(safety.subprocess, "Popen", popen)
    monkeypatch.setattr(safety, "_wait_for_worker_identity", lambda *args: None)
    monkeypatch.setattr(safety, "_lease_owner_state", lambda lease: "match")
    first = safety.start_service(ROOT, CONFIG)
    assert first["pid"] == 4242
    with pytest.raises(RuntimeError, match="lease already active"):
        safety.start_service(ROOT, CONFIG)
    assert len(launches) == 1


def test_stale_lease_recovery_requires_verified_nonmatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    paths = _runtime_paths(tmp_path)
    identity = "d" * 64
    stale = {
        "schema_version": safety.LEASE_SCHEMA,
        "runtime_epoch": config["runtime_epoch"],
        "runtime_directory": config["runtime_directory"],
        "worker_argv_sha256": identity,
        "owner_kind": "worker",
        "owner_pid": 42,
        "owner_argv_sha256": identity,
        "created_utc": "2026-08-11T00:00:00+00:00",
        "updated_utc": "2026-08-11T00:00:00+00:00",
    }
    paths["lease"].write_bytes(safety._lease_payload(stale))
    monkeypatch.setattr(safety, "_current_exact_argv_identity", lambda: "e" * 64)
    monkeypatch.setattr(safety, "_lease_owner_state", lambda lease: "mismatch")
    recovered = safety._acquire_start_lease(paths, config, identity)
    assert recovered["owner_kind"] == "starter"
    assert recovered["owner_pid"] == os.getpid()
    assert not paths["lease_recovery"].exists()
    paths["lease"].write_bytes(safety._lease_payload(stale))
    monkeypatch.setattr(safety, "_lease_owner_state", lambda lease: "unverifiable")
    with pytest.raises(RuntimeError, match="lease already active: unverifiable"):
        safety._acquire_start_lease(paths, config, identity)
    assert paths["lease"].is_file()


def test_first_refresh_sees_running_checkpoint_and_stop_allows_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    paths = _runtime_paths(tmp_path)
    identity = safety._argv_identity(safety._worker_command(ROOT, CONFIG))
    _write_checkpoint(paths["checkpoint"], config)
    _write_worker_lease(paths["lease"], config, identity)
    observed: dict[str, object] = {}

    def refresh(root: Path, value: dict, guard) -> dict:
        status = safety.service_status(root, CONFIG)
        observed.update(status)
        stop_status = safety.stop_service(root, CONFIG)
        assert stop_status["state"] == "running"
        assert paths["lease"].is_file()
        guard()
        raise AssertionError("stop request was not observed")

    monkeypatch.setattr(safety, "load_safety_config", lambda root, path: config)
    monkeypatch.setattr(safety, "_paths", lambda root, value: paths)
    monkeypatch.setattr(safety, "_assert_no_legacy_worker", lambda *args: None)
    monkeypatch.setattr(safety, "_pid_matches_worker", lambda pid, expected: pid == os.getpid())
    monkeypatch.setattr(safety, "safe_refresh_once", refresh)
    assert safety.run_worker(ROOT, CONFIG) == 0
    assert observed["state"] == "running"
    assert observed["alive"] is True
    stopped = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
    assert stopped["state"] == "stopped"
    assert stopped["pid"] is None
    assert stopped["stop_reason"] == "external_stop_requested"
    assert not paths["lease"].exists()

    monkeypatch.setattr(safety, "_current_exact_argv_identity", lambda: "f" * 64)
    rollback_lease = safety._acquire_start_lease(paths, config, identity)
    assert rollback_lease["owner_kind"] == "starter"


def test_worker_releases_lease_even_if_final_checkpoint_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    paths = _runtime_paths(tmp_path)
    identity = safety._argv_identity(safety._worker_command(ROOT, CONFIG))
    _write_checkpoint(paths["checkpoint"], config)
    _write_worker_lease(paths["lease"], config, identity)
    original_write = safety._safe_atomic_write

    def controlled_write(path: Path, payload: bytes, maximum: int) -> None:
        if path == paths["checkpoint"]:
            checkpoint = json.loads(payload)
            if checkpoint["state"] == "stopped":
                raise OSError("final checkpoint fixture failure")
        original_write(path, payload, maximum)

    monkeypatch.setattr(safety, "load_safety_config", lambda root, path: config)
    monkeypatch.setattr(safety, "_paths", lambda root, value: paths)
    monkeypatch.setattr(safety, "_assert_no_legacy_worker", lambda *args: None)
    monkeypatch.setattr(
        safety,
        "safe_refresh_once",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            safety.ControlSignal("external_stop_requested")
        ),
    )
    monkeypatch.setattr(safety, "_safe_atomic_write", controlled_write)
    with pytest.raises(OSError, match="final checkpoint fixture failure"):
        safety.run_worker(ROOT, CONFIG)
    assert not paths["lease"].exists()


def test_readiness_is_deterministic_and_does_not_open_runtime_state() -> None:
    rebuilt = safety.build_safety_readiness(ROOT, CONFIG)
    stored = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert rebuilt == stored
    assert rebuilt["live_database_opened_by_readiness"] is False
    assert rebuilt["supervisor_outputs_opened_by_readiness"] is False
    assert rebuilt["service_started"] is False
    assert rebuilt["safety_contract"]["stale_projection_publication_allowed"] is False
    assert rebuilt["safety_contract"]["leaderboard_history_seed_checked_snapshot"] == (
        safety.EXPECTED_CHECKED_SNAPSHOT
    )
    assert rebuilt["safety_contract"]["leaderboard_history_seed_pre_and_post_hash_guarded"] is True
    assert rebuilt["safety_contract"]["maximum_seed_history_entries"] == 64
