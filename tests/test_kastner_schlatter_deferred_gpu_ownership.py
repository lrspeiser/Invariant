from __future__ import annotations

import json
from pathlib import Path

import pytest

import sigma_theory_compiler.kastner_schlatter_deferred_gpu_ownership as deferred

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_deferred_gpu_ownership.json"
READINESS = ROOT / "runs/engine/kastner-schlatter-deferred-gpu-ownership-readiness.json"


def _sample(utilization: int, free_mib: int) -> dict:
    return {
        "device_index": 0,
        "device_name": "NVIDIA GeForce RTX 5090",
        "gpu_utilization_percent": utilization,
        "memory_used_mib": 32607 - free_mib,
        "memory_free_mib": free_mib,
        "memory_total_mib": 32607,
        "power_watts": 100.0,
        "sampled_utc": "synthetic-control",
        "scope": "synthetic NVML control",
    }


def test_readiness_validates_and_did_not_start_waiter() -> None:
    result = json.loads(READINESS.read_text(encoding="utf-8"))
    deferred.validate_readiness(result, CONFIG)
    assert result["execution_state"] == {
        "waiter_started_by_readiness": False,
        "gpu_owner_reserved_by_readiness": False,
        "existing_gpu_process_signaled": False,
        "sqlite_accessed": False,
    }
    assert result["ownership_contract"]["required_consecutive_safe_samples"] == 3
    assert result["ownership_contract"]["handoff_to_scheduler_automatic"] is False


def test_three_consecutive_safe_samples_reserve_then_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequence = iter([_sample(10, 9000), _sample(15, 8500), _sample(5, 10000)])
    monkeypatch.setattr(deferred, "sample_nvml", lambda: next(sequence))
    monkeypatch.setattr(deferred.time, "sleep", lambda _: None)
    token = deferred.wait_for_ownership(CONFIG, runtime_override=tmp_path, maximum_polls_override=3)
    assert token.checkpoint["state"] == "reserved"
    assert token.checkpoint["consecutive_safe_samples"] == 3
    assert token.lease_path.exists()
    assert not list(tmp_path.glob("*.sqlite*"))
    token.release()
    assert not token.lease_path.exists()
    assert token.checkpoint["state"] == "released"


def test_unsafe_sample_resets_consecutive_counter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequence = iter(
        [
            _sample(5, 9000),
            _sample(99, 9000),
            _sample(5, 9000),
            _sample(5, 9000),
            _sample(5, 9000),
        ]
    )
    monkeypatch.setattr(deferred, "sample_nvml", lambda: next(sequence))
    monkeypatch.setattr(deferred.time, "sleep", lambda _: None)
    token = deferred.wait_for_ownership(CONFIG, runtime_override=tmp_path, maximum_polls_override=5)
    assert token.checkpoint["polls"] == 5
    token.release()


def test_bounded_timeout_releases_lease(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deferred, "sample_nvml", lambda: _sample(100, 1000))
    monkeypatch.setattr(deferred.time, "sleep", lambda _: None)
    with pytest.raises(TimeoutError, match="bounded"):
        deferred.wait_for_ownership(CONFIG, runtime_override=tmp_path, maximum_polls_override=4)
    assert not (tmp_path / "deferred-gpu-owner.lease.json").exists()
    checkpoint = json.loads((tmp_path / "deferred-gpu-owner-checkpoint.json").read_text())
    assert checkpoint["state"] == "timed_out"
    assert checkpoint["polls"] == 4


def test_active_duplicate_rejected_and_stale_lease_recovered(tmp_path: Path) -> None:
    config, _ = deferred.load_config(CONFIG)
    lease_path, owned = deferred._acquire_waiter(tmp_path, config)
    with pytest.raises(RuntimeError, match="already active"):
        deferred._acquire_waiter(tmp_path, config)
    lease_path.unlink()
    stale_body = {
        "schema_version": "sigma-deferred-gpu-owner-lease-1.0",
        "mechanism_id": config["mechanism_id"],
        "role": "waiting",
        "pid": 2_147_483_647,
        "process_argv_sha256": "0" * 64,
        "polls": 7,
        "updated_utc": "stale-control",
    }
    stale = {**stale_body, "content_sha256": deferred._content_sha(stale_body)}
    lease_path.write_text(json.dumps(stale), encoding="utf-8")
    recovered_path, _ = deferred._acquire_waiter(tmp_path, config)
    assert recovered_path.exists()
    assert (tmp_path / config["recovery_name"]).exists()
    recovered_path.unlink()
    assert owned["role"] == "waiting"


def test_sqlite_contamination_and_live_runtime_override_fail(tmp_path: Path) -> None:
    config, _ = deferred.load_config(CONFIG)
    (tmp_path / "forbidden.sqlite").write_bytes(b"not opened")
    with pytest.raises(RuntimeError, match="contains SQLite"):
        deferred._acquire_waiter(tmp_path, config)
    with pytest.raises(ValueError, match="live campaign runtime override"):
        deferred.wait_for_ownership(
            CONFIG,
            runtime_override=ROOT / "runs/campaigns/deferred-forbidden",
            maximum_polls_override=1,
        )


def test_config_and_readiness_tamper_fail(tmp_path: Path) -> None:
    result = json.loads(READINESS.read_text(encoding="utf-8"))
    result["scientific_test_pass"] = True
    result["content_sha256"] = deferred._content_sha(result)
    with pytest.raises(ValueError, match="claim"):
        deferred.validate_readiness(result, CONFIG)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["required_consecutive_safe_samples"] = 1
    bad = tmp_path / "configs" / CONFIG.name
    bad.parent.mkdir()
    bad.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises((FileNotFoundError, ValueError)):
        deferred.load_config(bad)


def test_oversize_lease_and_widened_poll_bound_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = deferred.load_config(CONFIG)
    lease = tmp_path / config["lease_name"]
    lease.write_bytes(b"x" * (int(config["maximum_checkpoint_bytes"]) + 1))
    with pytest.raises(RuntimeError, match="exceeds bound"):
        deferred._acquire_waiter(tmp_path, config)
    lease.unlink()
    monkeypatch.setattr(deferred, "sample_nvml", lambda: _sample(1, 10_000))
    with pytest.raises(ValueError, match="poll bound"):
        deferred.wait_for_ownership(CONFIG, runtime_override=tmp_path, maximum_polls_override=722)


def test_source_has_no_sqlite_subprocess_or_signal_surface() -> None:
    source = (
        ROOT / "src/sigma_theory_compiler/kastner_schlatter_deferred_gpu_ownership.py"
    ).read_text()
    lowered = source.lower()
    assert "import sqlite" not in lowered
    assert "import subprocess" not in lowered
    assert "os.kill" not in lowered
    assert "terminate(" not in lowered
    assert "popen(" not in lowered
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["seals"]["sqlite_access"] is False
    assert config["seals"]["existing_gpu_process_signaled"] is False
