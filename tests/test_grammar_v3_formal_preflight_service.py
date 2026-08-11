from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.grammar_v3_formal_preflight_service import (
    GrammarV3FormalPreflightService,
    portable_status,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "grammar_v3_formal_preflight_service.json"
ARTIFACT = ROOT / "runs" / "engine" / "grammar-v3-formal-preflight-status.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _enabled() -> dict:
    config = _load(CONFIG)
    config["execution_enabled"] = True
    return config


def _run(directory: Path) -> tuple[GrammarV3FormalPreflightService, dict]:
    service = GrammarV3FormalPreflightService(directory, _enabled(), ROOT)
    admission = service.enqueue()
    assert admission["accepted"] == 163
    result = service.run_bounded()
    assert result["executed"] == 163
    return service, result["status"]


def test_checked_in_service_is_disabled_and_registry_is_exact(tmp_path: Path) -> None:
    service = GrammarV3FormalPreflightService(tmp_path / "disabled", _load(CONFIG), ROOT)
    assert service.status()["execution_enabled"] is False
    assert len(service.work_items) == 163
    assert service.candidate_registry_root_sha256 == (
        "3e2b33ced97fe135bdf2c3eda55e08ec64ff87e213e43f26c9a1dd19e546d504"
    )
    with pytest.raises(RuntimeError, match="disabled by config"):
        service.enqueue()


def test_real_reviewed_preflight_counts_seals_and_replay(tmp_path: Path) -> None:
    directory = tmp_path / "real"
    service, status = _run(directory)
    assert status["work_state_counts"] == {"succeeded": 163}
    assert status["decision_counts"] == {"blocked": 1, "pass": 162}
    assert status["family_decision_counts"] == {
        "AETHER_K1234_PARAMETER_CELL": {"pass": 128},
        "CONFORMAL_G4_PHI_SCALAR_TENSOR": {"blocked": 1},
        "CUBIC_HORNDESKI_G3_WEAK_CELL": {"pass": 32},
        "KESSENCE_G2_CONVEX": {"pass": 2},
    }
    assert status["gate_counts"] == {
        "family_prerequisite": {"blocked": 1, "pass": 162},
        "receipt_binding": {"pass": 163},
    }
    assert status["expensive_adm_or_global_energy_run"] is False
    assert status["observational_data_opened"] is False
    assert status["data_eligibility"] == {**ELIGIBILITY, "passed": True}
    assert status["paid_llm_spend_usd"] == 0.0
    assert status["disk_bytes"] <= service.config["budget"]["maximum_disk_bytes"]

    resumed = GrammarV3FormalPreflightService(directory, _enabled(), ROOT)
    assert resumed.enqueue()["duplicate"] == 163
    replay = resumed.run_bounded()
    assert replay["executed"] == 0
    assert portable_status(replay["status"]) == portable_status(status)


def test_expired_lease_recovery_and_missing_adapter_block(tmp_path: Path) -> None:
    directory = tmp_path / "recover"
    unavailable = frozenset({"CUBIC_HORNDESKI_G3_WEAK_CELL"})
    service = GrammarV3FormalPreflightService(
        directory, _enabled(), ROOT, unavailable_families=unavailable
    )
    assert service.enqueue()["accepted"] == 163
    abandoned = service.coordinator.claim("cpu", "crashed", lease_seconds=-1)
    assert abandoned is not None
    resumed = GrammarV3FormalPreflightService(
        directory, _enabled(), ROOT, unavailable_families=unavailable
    )
    result = resumed.run_bounded()
    assert result["recovered"] == {"recovered": 1, "failed": 0}
    assert result["status"]["decision_counts"] == {"blocked": 33, "pass": 130}
    assert result["status"]["family_decision_counts"][
        "CUBIC_HORNDESKI_G3_WEAK_CELL"
    ] == {"blocked": 32}


def test_binding_and_payload_tamper_fail_closed(tmp_path: Path) -> None:
    config = _enabled()
    config["compilation_campaign"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="file hash mismatch"):
        GrammarV3FormalPreflightService(tmp_path / "binding", config, ROOT)

    directory = tmp_path / "tamper"
    service = GrammarV3FormalPreflightService(directory, _enabled(), ROOT)
    service.enqueue()
    with service.coordinator.connect() as connection:
        row = connection.execute(
            "SELECT work_id,payload_json FROM work ORDER BY ordinal LIMIT 1"
        ).fetchone()
        payload = json.loads(row["payload_json"])
        payload["typed_action_ir_sha256"] = "0" * 64
        connection.execute(
            "UPDATE work SET payload_json=? WHERE work_id=?",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")), row["work_id"]),
        )
    with pytest.raises(ValueError, match="payload was tampered"):
        service.status()


def test_committed_portable_status_is_exact(tmp_path: Path) -> None:
    _, status = _run(tmp_path / "artifact")
    assert portable_status(status) == _load(ARTIFACT)
