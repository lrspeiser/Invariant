from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.grammar_v3_g2_candidate_formal_service import (
    GrammarV3G2CandidateFormalService,
    portable_status,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "grammar_v3_g2_candidate_formal_service.json"
ARTIFACT = ROOT / "runs" / "engine" / "grammar-v3-g2-candidate-formal-status.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _enabled() -> dict:
    config = _load(CONFIG)
    config["execution_enabled"] = True
    return config


def _run(directory: Path) -> tuple[GrammarV3G2CandidateFormalService, dict]:
    service = GrammarV3G2CandidateFormalService(directory, _enabled(), ROOT)
    assert service.enqueue()["accepted"] == 2
    result = service.run_bounded()
    assert result["executed"] == 2
    return service, result["status"]


def test_checked_in_disabled_and_two_exact_candidates_are_bound(tmp_path: Path) -> None:
    service = GrammarV3G2CandidateFormalService(tmp_path / "disabled", _load(CONFIG), ROOT)
    assert len(service.work_items) == 2
    assert {item["quadratic_coefficient"] for item in service.work_items} == {
        "1/8",
        "1/4",
    }
    assert all(item["preflight_result_sha256"] for item in service.work_items)
    assert all(item["admission_result_sha256"] for item in service.work_items)
    assert service.status()["execution_enabled"] is False
    with pytest.raises(RuntimeError, match="disabled"):
        service.enqueue()


def test_strongest_reviewed_g2_gates_and_global_block_are_exact(tmp_path: Path) -> None:
    directory = tmp_path / "real"
    _, status = _run(directory)
    assert status["work_state_counts"] == {"succeeded": 2}
    assert status["decision_counts"] == {"blocked": 2}
    assert status["blocker_counts"] == {
        "hash_bound_general_nonmaximal_positive_mass_theorem": 2
    }
    pass_gates = {
        "candidate_action_preflight_admission_binding",
        "exact_polynomial_predecessor_equivalence",
        "covariant_variation_noether",
        "coupled_adm_primary_and_legendre",
        "candidate_local_dirac_pair",
        "principal_symbol",
        "common_time_cone",
        "pointwise_hamiltonian",
        "dominant_energy_condition",
        "explicit_asymptotically_flat_contract",
        "scalar_boundary_flux",
        "restricted_maximal_slice_positive_mass",
    }
    blocked_gates = {
        "general_nonmaximal_positive_mass",
        "global_positive_energy",
        "full_formal_completion",
    }
    assert all(status["gate_counts"][gate] == {"pass": 2} for gate in pass_gates)
    assert all(status["gate_counts"][gate] == {"blocked": 2} for gate in blocked_gates)
    assert status["general_nonmaximal_global_positive_mass_proved"] is False
    assert status["full_formal_pass_count"] == 0
    assert status["observational_data_opened"] is False
    assert status["data_eligibility"] == {**ELIGIBILITY, "passed": True}
    assert status["paid_llm_spend_usd"] == 0.0

    resumed = GrammarV3G2CandidateFormalService(directory, _enabled(), ROOT)
    assert resumed.enqueue()["duplicate"] == 2
    replay = resumed.run_bounded()
    assert replay["executed"] == 0
    assert portable_status(replay["status"]) == portable_status(status)


def test_missing_adapter_and_expired_lease_remain_blocked(tmp_path: Path) -> None:
    missing = frozenset({"generic_kessence_timelike_principal_hamiltonian"})
    directory = tmp_path / "missing"
    service = GrammarV3G2CandidateFormalService(
        directory, _enabled(), ROOT, missing_adapter_ids=missing
    )
    service.enqueue()
    abandoned = service.coordinator.claim("cpu", "crashed", lease_seconds=-1)
    assert abandoned is not None
    resumed = GrammarV3G2CandidateFormalService(
        directory, _enabled(), ROOT, missing_adapter_ids=missing
    )
    result = resumed.run_bounded()
    assert result["recovered"] == {"recovered": 1, "failed": 0}
    status = result["status"]
    assert status["decision_counts"] == {"blocked": 2}
    assert status["gate_counts"]["principal_symbol"] == {"blocked": 2}
    assert next(iter(status["blocker_counts"])).startswith(
        "reviewed_local_g2_adapter_missing"
    )


def test_upstream_hash_and_persisted_payload_tamper_fail_closed(tmp_path: Path) -> None:
    config = _enabled()
    config["positive_mass_audit"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content hash mismatch"):
        GrammarV3G2CandidateFormalService(tmp_path / "binding", config, ROOT)

    directory = tmp_path / "tamper"
    service = GrammarV3G2CandidateFormalService(directory, _enabled(), ROOT)
    service.enqueue()
    with service.coordinator.connect() as connection:
        row = connection.execute(
            "SELECT work_id,payload_json FROM work ORDER BY ordinal LIMIT 1"
        ).fetchone()
        payload = json.loads(row["payload_json"])
        payload["admission_result_sha256"] = "0" * 64
        connection.execute(
            "UPDATE work SET payload_json=? WHERE work_id=?",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")), row["work_id"]),
        )
    with pytest.raises(ValueError, match="payload was tampered"):
        service.status()


def test_committed_portable_status_is_exact(tmp_path: Path) -> None:
    _, status = _run(tmp_path / "artifact")
    assert portable_status(status) == _load(ARTIFACT)
