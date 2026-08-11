from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.grammar_v3_promotion_admission_service import (
    GrammarV3PromotionAdmissionService,
    portable_status,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "grammar_v3_promotion_admission_service.json"
ARTIFACT = ROOT / "runs" / "engine" / "grammar-v3-promotion-admission-status.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _enabled() -> dict:
    config = _load(CONFIG)
    config["execution_enabled"] = True
    return config


def _run(directory: Path) -> tuple[GrammarV3PromotionAdmissionService, dict]:
    service = GrammarV3PromotionAdmissionService(directory, _enabled(), ROOT)
    assert service.enqueue()["accepted"] == 162
    result = service.run_bounded()
    assert result["executed"] == 162
    return service, result["status"]


def test_checked_in_service_disabled_and_only_preflight_passes_are_present(
    tmp_path: Path,
) -> None:
    service = GrammarV3PromotionAdmissionService(tmp_path / "disabled", _load(CONFIG), ROOT)
    assert len(service.preflight.work_items) == 163
    assert len(service.work_items) == 162
    assert all(
        item["family_id"] != "CONFORMAL_G4_PHI_SCALAR_TENSOR"
        for item in service.work_items
    )
    assert service.status()["execution_enabled"] is False
    with pytest.raises(RuntimeError, match="disabled by config"):
        service.enqueue()


def test_exact_family_queue_admission_replay_and_seals(tmp_path: Path) -> None:
    directory = tmp_path / "real"
    service, status = _run(directory)
    assert status["preflight_candidate_count"] == 163
    assert status["preflight_status_binding"] == service.config["preflight_status"]
    assert status["preflight_config_binding"] == service.config["preflight_config"]
    assert status["preflight_pass_count"] == 162
    assert status["preflight_blocked_excluded_count"] == 1
    assert status["eligible_candidate_count"] == 162
    assert status["work_state_counts"] == {"succeeded": 162}
    assert status["decision_counts"] == {"pass": 162}
    assert status["family_decision_counts"] == {
        "AETHER_K1234_PARAMETER_CELL": {"pass": 128},
        "CUBIC_HORNDESKI_G3_WEAK_CELL": {"pass": 32},
        "KESSENCE_G2_CONVEX": {"pass": 2},
    }
    assert status["target_queue_counts"] == {
        "grammar_v3_aether_candidate_adm_formal": 128,
        "grammar_v3_g2_candidate_adm_formal": 2,
        "grammar_v3_g3_candidate_adm_formal": 32,
    }
    assert set(status["target_queue_registry_roots"]) == set(
        status["target_queue_counts"]
    )
    assert all(len(root) == 64 for root in status["target_queue_registry_roots"].values())
    assert status["downstream_expensive_execution_started"] is False
    assert status["observational_data_opened"] is False
    assert status["data_eligibility"] == {**ELIGIBILITY, "passed": True}
    assert status["paid_llm_spend_usd"] == 0.0
    assert status["disk_bytes"] <= service.config["budget"]["maximum_disk_bytes"]

    resumed = GrammarV3PromotionAdmissionService(directory, _enabled(), ROOT)
    assert resumed.enqueue()["duplicate"] == 162
    replay = resumed.run_bounded()
    assert replay["executed"] == 0
    assert portable_status(replay["status"]) == portable_status(status)


def test_missing_admission_adapter_and_expired_lease_fail_closed(tmp_path: Path) -> None:
    directory = tmp_path / "missing"
    unavailable = frozenset({"AETHER_K1234_PARAMETER_CELL"})
    service = GrammarV3PromotionAdmissionService(
        directory,
        _enabled(),
        ROOT,
        unavailable_admission_families=unavailable,
    )
    assert service.enqueue()["accepted"] == 162
    abandoned = service.coordinator.claim("cpu", "crashed", lease_seconds=-1)
    assert abandoned is not None
    resumed = GrammarV3PromotionAdmissionService(
        directory,
        _enabled(),
        ROOT,
        unavailable_admission_families=unavailable,
    )
    result = resumed.run_bounded()
    assert result["recovered"] == {"recovered": 1, "failed": 0}
    assert result["status"]["decision_counts"] == {"blocked": 128, "pass": 34}
    assert result["status"]["family_decision_counts"][
        "AETHER_K1234_PARAMETER_CELL"
    ] == {"blocked": 128}
    assert "grammar_v3_aether_candidate_adm_formal" not in result["status"][
        "target_queue_counts"
    ]


def test_artifact_binding_and_persisted_payload_tamper_refused(tmp_path: Path) -> None:
    config = _enabled()
    config["preflight_status"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content hash mismatch"):
        GrammarV3PromotionAdmissionService(tmp_path / "binding", config, ROOT)

    directory = tmp_path / "tamper"
    service = GrammarV3PromotionAdmissionService(directory, _enabled(), ROOT)
    service.enqueue()
    with service.coordinator.connect() as connection:
        row = connection.execute(
            "SELECT work_id,payload_json FROM work ORDER BY ordinal LIMIT 1"
        ).fetchone()
        payload = json.loads(row["payload_json"])
        payload["preflight_result_sha256"] = "0" * 64
        connection.execute(
            "UPDATE work SET payload_json=? WHERE work_id=?",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")), row["work_id"]),
        )
    with pytest.raises(ValueError, match="payload was tampered"):
        service.status()


def test_committed_portable_status_is_exact(tmp_path: Path) -> None:
    _, status = _run(tmp_path / "artifact")
    assert portable_status(status) == _load(ARTIFACT)
